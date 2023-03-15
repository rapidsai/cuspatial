#! /usr/bin/env bash

mkdir -m 0755 -p ~/.{aws,cache,config/clangd,conda,local};

cp /etc/skel/.config/clangd/config.yaml ~/.config/clangd/config.yaml;

rapids-make-vscode-workspace --update;
