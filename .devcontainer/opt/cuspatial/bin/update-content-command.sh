#! /usr/bin/env bash

mkdir -m 0755 -p ~/{.aws,.cache,.conda,.config};

rapids-make-vscode-workspace > ~/workspace.code-workspace;
