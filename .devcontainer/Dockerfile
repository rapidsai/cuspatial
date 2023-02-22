# syntax=docker/dockerfile:1.5

ARG CUDA=11.8
ARG LLVM=15
ARG RAPIDS=23.04
ARG DISTRO=ubuntu22.04
ARG REPO=rapidsai/devcontainers
ARG FROM_IMAGE=${REPO}:${RAPIDS}-cpp-llvm${LLVM}-cuda${CUDA}-mambaforge-${DISTRO}

FROM ${FROM_IMAGE}

COPY --chown=root:root opt/cuspatial /opt/cuspatial

WORKDIR /home/coder/

ENV PYTHONDONTWRITEBYTECODE="1"

ENV SCCACHE_REGION="us-east-2"
ENV SCCACHE_BUCKET="rapids-sccache-devs"
ENV VAULT_HOST="https://vault.ops.k8s.rapids.ai"

ENV HISTFILE="/home/coder/.cache/._bash_history"
