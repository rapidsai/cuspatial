# How to Contribute to cuSpatial

`cuSpatial` is a part of the RAPIDS community. When contributing to cuSpatial, developers should
follow the RAPIDS contribution guidelines. The RAPIDS documentation
[contributing section](https://docs.rapids.ai/contributing) walks through the process of identifying
an issue, submitting and merging a PR.

## Directory structure and file naming

The `cuspatial` package comprises several subpackages.

- `core` contains the main components of cuspatial
- `io` contains I/O functions for reading and writing external data objects
- `tests` contains unit tests for cuspatial
- `utils` contains utility functions
- `_lib` contains Cython APIs that wrap the C++ `libcuspatial` backend.

[`library_design`](library_design.md) further discusses high-level library design of `cuspatial`.

### Cython code

The `_lib` folder contains all cython code. Each feature in `libcuspatial` exposed to
`cuspatial` should have two Cython files:

1. A `pxd` file declaring C++ APIs so that they may be used in Cython, and
2. A `pyx` file containing Cython functions that wrap those C++ APIs so that they can be called from Python.

`pyx` files are organized under the root of `_lib`. `pxd` files are under `_lib/cpp`.
`pxd` files should mirror the file hierarchy of `cpp/include` in `libcuspatial`.

For more information see [the Cython layer design documentation](./library_design.md#cython-layer).

## Code style

cuSpatial employs a number of linters to ensure consistent style across the code base, and manages
them using [`pre-commit`](https://pre-commit.com/). Developers are strongly recommended to set up
`pre-commit` prior to any development. The `.pre-commit-config.yaml` file at the root of the repo is
the primary source of truth for linting.

To install pre-commit, install via conda/pip:

```bash
# conda
conda install -c conda-forge pre-commit
```
```bash
# pip
pip install pre-commit
```

Then run pre-commit hooks before committing code:
```bash
pre-commit run
```

Optionally, you may set up the pre-commit hooks to run automatically when you make a git commit. This can be done by running the following command in cuspatial repository:

```bash
pre-commit install
```

Now code linters and formatters will be run each time you commit changes.

You can skip these checks with `git commit --no-verify` or with the short version `git commit -n`.

### Linter Details

Specifically, cuSpatial uses the following tools:

- [`flake8`](https://github.com/pycqa/flake8) checks for general code formatting compliance. 
- [`black`](https://github.com/psf/black) is an automatic code formatter.
- [`isort`](https://pycqa.github.io/isort/) ensures imports are sorted consistently.

Linter config data is stored in a number of files. cuSpatial generally uses `pyproject.toml` over
`setup.cfg` and avoids project-specific files (e.g. `setup.cfg` > `python/cudf/setup.cfg`). However,
differences between tools and the different packages in the repo result in the following caveats:

- `flake8` has no plans to support `pyproject.toml`, so it must live in `setup.cfg`.
- `isort` must be configured per project to set which project is the "first party" project.

Additionally, cuSpatial's use of `versioneer` means that each project must have a `setup.cfg`.
As a result, cuSpatial currently maintains both root and project-level `pyproject.toml` and
`setup.cfg` files.

## Writing tests

Every new feature contributed to cuspatial should include unit tests. The unit test file should be
added to the `tests` folder. In general, the `tests` folder mirrors the folder hierarchy of the
`cuspatial` package. At the lowest level, each module expands into a folder that contains specific
test files for features in the module.

cuSpatial uses [`pytest`](https://docs.pytest.org/) as the unit testing framework. `conftest.py`
contains useful fixtures that can be shared across different test functions. Reusing these fixtures
reduces redundancy in test code.

cuspatial compute APIs should strive to reach result parity with its host (CPU) equivalent. For 
`GeoSeries` and `GeoDataFrame` features, unit tests should compare results with
corresponding `geopandas` functions.
