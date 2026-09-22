# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.12.17

FROM ghcr.io/astral-sh/uv:$UV_VERSION AS uv

FROM python:$PYTHON_VERSION-slim-bookworm AS builder
COPY --from=uv /uv /bin/uv

WORKDIR /src
COPY uv.lock pyproject.toml /src/
# dev dependencies are left out of the image
RUN uv export --frozen --no-default-groups --group gpu --all-extras --no-hashes --no-emit-project -o requirements.txt

FROM python:$PYTHON_VERSION-slim-bookworm

ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /root

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    # these are needed for ClearML
    git libsm6 libxext6 libxrender-dev libglib2.0-0 build-essential

# get rid of all distro python3 packages - they cause conflicts and we don't need them.
RUN apt list | grep ^python3- | sed 's|/.*||' | xargs apt remove -y
# but we at least need pip
RUN apt-get install --no-install-recommends -y python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

COPY --from=builder /src/requirements.txt .
COPY --exclude=.* . .

# We are installing these python packages globally, so we need to break the system packages
RUN mkdir -p ~/.config/pip && printf "[global]\nbreak-system-packages = true" > ~/.config/pip/pip.conf

RUN --mount=type=cache,target=/root/.cache \
    python -m pip install --no-cache-dir -r requirements.txt && rm requirements.txt && \
    # these are needed for clearml to run
    python -m pip install --no-cache-dir clearml-agent setuptools
RUN python -m pip install --no-deps . && rm -r /root/*
ENV CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1

CMD ["bash"]
