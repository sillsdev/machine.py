#compatability with Tensorflow 2.6.0 as per https://www.tensorflow.org/install/source#gpu
ARG PYTHON_VERSION=3.8
ARG UBUNTU_VERSION=focal
ARG POETRY_VERSION=1.2.0
ARG CUDA_VERSION=11.2.2-cudnn8-runtime-ubuntu20.04

# Install silnlp
FROM nvidia/cuda:$CUDA_VERSION
ARG PYTHON_VERSION
ARG POETRY_VERSION

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y \
    python$PYTHON_VERSION \
    python3-pip \
    python3-venv

# make some useful symlinks that are expected to exist
RUN ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python3  & \
    ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python

ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Add `poetry` to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /code

# Install dependencies
COPY . /code
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-cache --without dev

CMD ["bash"]
