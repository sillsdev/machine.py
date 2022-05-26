#compatability with Tensorflow 2.6.0 as per https://www.tensorflow.org/install/source#gpu
ARG PYTHON_VERSION=3.8
ARG UBUNTU_VERSION=focal
ARG POETRY_VERSION=1.1.13
ARG CUDA_VERSION=11.2.2-cudnn8-runtime-ubuntu20.04

# Install silnlp
FROM nvidia/cuda:$CUDA_VERSION
ARG PYTHON_VERSION
ARG POETRY_VERSION

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y \
    curl \
    build-essential \
    autoconf \
    libtool \
    pkg-config \
    python$PYTHON_VERSION \
    python3-distutils \
    python3-apt \
    python3-dev \
    python3-pip \
    gcc

# make some useful symlinks that are expected to exist
RUN ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python3  & \
    ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN pip install "poetry==$POETRY_VERSION"
WORKDIR /code
COPY . /code
RUN poetry config virtualenvs.create false && poetry install --no-dev --no-interaction --no-ansi

CMD ["bash"]