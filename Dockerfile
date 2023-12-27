ARG AARCH64_BASE_IMAGE=nvidia/cuda:11.8.0-devel-ubuntu22.04
FROM ${AARCH64_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_CROSS_VERSION=11-8
ENV CUDA_CROSS_VERSION_DOT=11.8


RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y git
RUN apt install -y python3 python3-pip

RUN pip3 install --upgrade pip

RUN apt-get -y install nvidia-cuda-toolkit
RUN apt install ninja-build -y
RUN pip install boto3 fastapi uvicorn

RUN pip3  install torch==2.0.1 deepspeed==0.10 tensorboard transformers datasets sentencepiece accelerate ray==2.7


ADD / .

EXPOSE 8000

CMD python3 -u api.py
