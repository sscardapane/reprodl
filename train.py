#!/usr/bin/env python
# coding: utf-8

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import torch, torchaudio
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as ptl
from pytorch_lightning.metrics.functional import accuracy

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

import wandb
from pytorch_lightning.loggers import WandbLogger

logger = logging.getLogger(__name__)

hyperparameter_defaults = dict(
    sample_rate = 8000,
    lr = 1e-4,
)

class CS50Dataset(torch.utils.data.Dataset):

    def __init__(self, datapath : Path, folds, sample_rate=8000):
        
        self.datapath = datapath
        self.csv = pd.read_csv(datapath / Path('meta/esc50.csv'))
        self.csv = self.csv[self.csv['fold'].isin(folds)]
        self.resample = torchaudio.transforms.Resample(orig_freq=44100, new_freq=sample_rate)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)
        self.power_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        
    def __getitem__(self, index):
        
        xb, sample_rate = torchaudio.load(self.datapath / 'audio' / f'{self.csv.iloc[index, 0]}')
        yb = self.csv.iloc[index, 2]

        sound = self.resample(xb)
        sound = self.mel(sound)
        return self.power_to_db(sound), yb
    
    def __len__(self):
        return len(self.csv)


class AudioNet(ptl.LightningModule):
    
    def __init__(self, hparams : DictConfig):
        super().__init__()
        self.hparams = hparams
        self.conv1 = nn.Conv2d(1, 128, 11, padding=5)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(512, hparams.model.n_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool2(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.fc1(x[:, :, 0, 0])
        return x
    
    def training_step(self, batch, batch_idx):
        # Can basically copy from here for the moment:
        # https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html
        xb, yb = batch
        y_pred = self(xb)
        loss = F.cross_entropy(y_pred, yb)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        y_pred = self(xb)
        loss = F.cross_entropy(y_pred, yb)
        y_hat = torch.argmax(y_pred, dim=1)
        acc = accuracy(y_hat, yb)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)
        self.log('valid_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optim.lr)
        return optimizer


@hydra.main(config_path="configs", config_name="default")
def train(cfg: DictConfig):

    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
    
    wandb.init(config=hyperparameter_defaults)
    to_merge = OmegaConf.create(wandb.config._as_dict())

    # Terrible hack :-(
    cfg.data.sample_rate = to_merge.sample_rate
    cfg.optim.lr = to_merge.lr

    wandb_logger = WandbLogger(project='reprodl')

    ptl.seed_everything(1)

    traindata = CS50Dataset(Path(get_original_cwd()) / Path(cfg.data.path), cfg.data.train_folds, cfg.data.sample_rate)
    testdata = CS50Dataset(Path(get_original_cwd()) / Path(cfg.data.path), cfg.data.test_folds, cfg.data.sample_rate)
    train_loader = torch.utils.data.DataLoader(traindata, batch_size=cfg.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testdata, batch_size=cfg.batch_size, shuffle=False)

    audionet = AudioNet(cfg)
    trainer = ptl.Trainer(**cfg.trainer, logger=wandb_logger)
    trainer.fit(audionet, train_loader, test_loader)

if __name__ == '__main__':
    train()