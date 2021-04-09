FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

RUN apt-get update && \
    apt-get upgrade -y

WORKDIR /reprodl
COPY . /reprodl

RUN pip install -r requirements.txt
RUN apt-get install -y libsndfile1-dev # torchaudio

RUN pip install dvc boto3 --ignore-installed ruamel.yaml
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
RUN dvc pull