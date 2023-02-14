ARG CUDA=11.8
ARG LLVM=15
ARG RAPIDS=23.04
ARG DISTRO=ubuntu22.04
ARG REPO=rapidsai/devcontainers
ARG FROM_IMAGE=${REPO}:${RAPIDS}-cpp-llvm${LLVM}-cuda${CUDA}-mambaforge-${DISTRO}

FROM ${FROM_IMAGE}

RUN /opt/conda/bin/pip install rapids-dependency-file-generator conda-merge

COPY --chown=coder:coder home/coder/workspace.code-workspace /home/coder/

COPY --chown=root:root opt/rapids /opt/rapids

WORKDIR /home/coder/

ENV RUSTC_WRAPPER=sccache
ENV CMAKE_C_COMPILER_LAUNCHER=sccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=sccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=sccache
ENV HISTFILE=/home/coder/.cache/._bash_history
ENV SCCACHE_S3_KEY_PREFIX=cuspatial-dev
