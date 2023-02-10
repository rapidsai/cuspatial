#! /usr/bin/env bash

skbuild_dir="$(python -c 'import skbuild; print(skbuild.constants.SKBUILD_DIR())')";

mkdir -p ~/.config/clangd;

cat <<EOF >  ~/.config/clangd/config.yaml
$(cat /opt/rapids/.clangd)
---
If:
  PathMatch: $HOME/rmm/.*
CompileFlags:
  CompilationDatabase: $HOME/rmm/build
---
If:
  PathMatch: $HOME/cudf/.*
CompileFlags:
  CompilationDatabase: $HOME/cudf/cpp/build
---
If:
  PathMatch: $HOME/cuspatial/.*
CompileFlags:
  CompilationDatabase: $HOME/cuspatial/cpp/build
EOF
