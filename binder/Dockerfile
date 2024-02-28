# This dockerfile is used to set up a plain Linux environment into which we can
# install all dependencies with pip, and avoid conda. The documentation about
# using Dockerfiles with mybinder can be found here:

# https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html#preparing-your-dockerfile

FROM ubuntu:22.04

# Install needed system packages

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libcairo2-dev \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip

# Set up user as required by mybinder docs:

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Copy data over.

COPY . ${HOME}/
USER root
RUN chown -R ${NB_UID} ${HOME}

# Remaining, unprivileged setup

USER ${NB_USER}
WORKDIR ${HOME}
ENV PATH="${HOME}/.local/bin:${PATH}"
ENV JUPYTER_CONFIG_DIR="${HOME}/.local/etc/jupyter/"
RUN binder/setup.sh
