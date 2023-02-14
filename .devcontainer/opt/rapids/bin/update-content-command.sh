#! /usr/bin/env bash

mkdir -m 0755 -p ~/{.aws,.cache,.conda,.config,rmm,cudf};

cat <<"EOF" >> ~/.bashrc
if [[ "$PATH" != *"/opt/rapids/bin"* ]]; then
    export PATH="$PATH:/opt/rapids/bin";
fi
EOF
