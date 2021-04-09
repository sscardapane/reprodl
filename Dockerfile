FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN git clone https://github.com/sscardapane/reprodl
WORKDIR /reprodl

RUN pip install -r requirements.txt

RUN pip install dvc boto3 --ignore-installed ruamel.yaml
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
RUN dvc remote add -d minio s3://dvc/
RUN dvc remote modify minio endpointurl http://141.108.25.49:9000
RUN dvc pull