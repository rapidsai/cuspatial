#! /usr/bin/env bash

if [[ ! -d ~/cudf/.git ]]; then
    echo "Cloning cuDF" 1>&2;
    /opt/devcontainer/bin/github/repo/clone.sh "rapidsai" "cudf" "cudf";
fi
