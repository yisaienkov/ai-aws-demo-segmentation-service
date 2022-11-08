FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y curl ca-certificates sudo git bzip2 && \
    rm -rf /var/lib/apt/lists/*

ENV TZ=Europe/Kiev

COPY requirements.txt ./requirements.txt

RUN python -m pip install -U pip && \
    python -m pip install -r requirements.txt && \
    python -m pip cache purge

COPY ./ /app/

WORKDIR /app

CMD uvicorn src.main:app --host=0.0.0.0
