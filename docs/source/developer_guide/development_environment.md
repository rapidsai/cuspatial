# Creating a Development Environment

cuSpatial recommends using [Dev Containers](https://containers.dev/) to setup the development environment.
Install `devcontainers` extension in vscode, and open this repo in vscode. When promted to open the repo
in devcontainer, click proceed.

## From Bare Metal

RAPIDS keeps a single source of truth for library dependencies in `dependencies.yaml`. This file divides
the dependencies into several dimensions: building, testing, documentations, notebooks etc. As a developer,
you generally want to generate an environment recipe that includes everything that the library *may* use.

To do so, install the rapids-dependency-file-generator via pip:
```shell
pip install rapids-dependency-file-generator
```

And run under the repo root:
```shell
rapids-dependency-file-generator --clean
```

The environment recipe is generated within the `conda/environments` directory. To continue the next step of building,
see the [build page](https://docs.rapids.ai/api/cuspatial/stable/developer_guide/build.html).
