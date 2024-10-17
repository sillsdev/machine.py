#compatability with Tensorflow 2.6.0 as per https://www.tensorflow.org/install/source#gpu
ARG PYTHON_VERSION=3.12
ARG UBUNTU_VERSION=noble
ARG POETRY_VERSION=1.6.1
ARG CUDA_VERSION=12.6.1-base-ubuntu24.04

FROM python:$PYTHON_VERSION-slim as builder
ARG POETRY_VERSION

ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Add `poetry` to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /src
COPY poetry.lock pyproject.toml /src
RUN poetry export --with=gpu --without-hashes -f requirements.txt > requirements.txt


FROM nvidia/cuda:$CUDA_VERSION
ARG PYTHON_VERSION

ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /root

RUN apt-get update && \
    apt-get install --no-install-recommends -y software-properties-common && \
    apt-get update && \
    apt-get install --no-install-recommends -y \
    curl \
    python$PYTHON_VERSION \
    # these are needed for ClearML
    git libsm6 libxext6 libxrender-dev libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python$PYTHON_VERSION

# make some useful symlinks that are expected to exist
RUN ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python3  & \
    ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python

COPY --from=builder /src/requirements.txt .
RUN --mount=type=cache,target=/root/.cache \
    pip install --no-cache-dir -r requirements.txt && rm requirements.txt

COPY . .
RUN pip install --no-deps . && rm -r /root/*

CMD ["bash"]
