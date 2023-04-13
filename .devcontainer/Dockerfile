# syntax=docker/dockerfile:1.5

ARG CUDA=11.8
ARG LLVM=16
ARG RAPIDS=23.06
ARG DISTRO=ubuntu22.04
ARG REPO=rapidsai/devcontainers

ARG PYTHON_PACKAGE_MANAGER=conda

FROM ${REPO}:${RAPIDS}-cpp-llvm${LLVM}-cuda${CUDA}-${DISTRO} as pip-base

FROM ${REPO}:${RAPIDS}-cpp-llvm${LLVM}-cuda${CUDA}-mambaforge-${DISTRO} as conda-base

FROM ${PYTHON_PACKAGE_MANAGER}-base

ARG PYTHON_PACKAGE_MANAGER
ENV PYTHON_PACKAGE_MANAGER="${PYTHON_PACKAGE_MANAGER}"

USER coder

# COPY --chown=coder:coder opt/cuspatial /opt/cuspatial

RUN /bin/bash -c 'mkdir -m 0755 -p ~/.{aws,cache,conda,config/pip,local}'

WORKDIR /home/coder/

ENV PYTHONSAFEPATH="1"
ENV PYTHONUNBUFFERED="1"
ENV PYTHONDONTWRITEBYTECODE="1"

ENV SCCACHE_REGION="us-east-2"
ENV SCCACHE_BUCKET="rapids-sccache-devs"
ENV VAULT_HOST="https://vault.ops.k8s.rapids.ai"

ENV HISTFILE="/home/coder/.cache/._bash_history"
