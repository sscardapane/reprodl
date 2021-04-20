import unittest
import torch
import pytorch_lightning as ptl
from train import AudioNet
import omegaconf

class TestModel(unittest.TestCase):

    def setUp(self):
        self.cfg = omegaconf.OmegaConf.load('./configs/default.yaml')

    def test_model_creation(self):
        with torch.no_grad():
            net = AudioNet(self.cfg)
            yb = net(torch.randn((1, 1, 200, 100)))
        self.assertEqual(yb.shape, torch.Size([1, 50]))

    #def test_model_overfit(self):
    #    ptl.seed_everything(1234)
    #    data = torch.utils.data.TensorDataset(torch.randn(5, 1, 200, 100), torch.randint(0, 10, size=(5,)))
    #    net = AudioNet(self.cfg)
    #    self.cfg.trainer.max_epochs = 100
    #    trainer = ptl.Trainer(**self.cfg.trainer, overfit_batches=1)
    #    trainer.fit(net, torch.utils.data.DataLoader(data))
    #    self.assertLessEqual(trainer.logged_metrics['train_loss'].item(), 0.05)

if __name__ == '__main__':
    unittest.main()