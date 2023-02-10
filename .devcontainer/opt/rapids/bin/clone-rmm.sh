#! /usr/bin/env bash

if [[ ! -d ~/rmm/.git ]]; then
    echo "Cloning RMM" 1>&2;
    /opt/devcontainer/bin/github/repo/clone.sh "rapidsai" "rmm" "rmm";
fi
