#compatability with Tensorflow 2.6.0 as per https://www.tensorflow.org/install/source#gpu
ARG PYTHON_VERSION=3.8
ARG UBUNTU_VERSION=focal
ARG POETRY_VERSION=1.2.0
ARG CUDA_VERSION=11.2.2-cudnn8-runtime-ubuntu20.04

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
COPY . /src
RUN poetry build
RUN poetry export --without-hashes -f requirements.txt > requirements.txt


FROM nvidia/cuda:$CUDA_VERSION
ARG PYTHON_VERSION

ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    python$PYTHON_VERSION \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# make some useful symlinks that are expected to exist
RUN ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python3  & \
    ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python

COPY --from=builder /src/requirements.txt .
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt && rm requirements.txt

COPY --from=builder /src/dist/*.whl .
RUN pip install --no-deps *.whl && rm *.whl

CMD ["bash"]
