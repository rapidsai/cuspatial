# Benchmarking cuDF

The goal of the benchmarks in this repository is to measure the performance of various cuDF APIs.
Benchmarks in cuDF are written using the
[`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/index.html) plugin to the
[`pytest`](https://docs.pytest.org/en/latest/) Python testing framework.
Using `pytest-benchmark` provides a seamless experience for developers familiar with `pytest`.
We include benchmarks of both public APIs and internal functions.
The former give us a macro view of our performance, especially vis-à-vis geopandas.
The latter help us quantify and minimize the overhead of our Python bindings.

```{note}
Our current benchmarks focus entirely on measuring run time.
However, minimizing memory footprint can be just as important for some cases.
In the future, we may update our benchmarks to also include memory usage measurements.
```

## Benchmark organization

At the top level benchmarks are divided into `internal` and `API` directories.
API benchmarks are for public features that we expect users to consume.
Internal benchmarks capture the performance of cuspatial internals that have no stability guarantees.

Within each directory, benchmarks are organized based on the type of function.
Functions in cuspatial generally fall into two groups:

1. Methods of classes like `GeoDataFrame` or `GeoSeries`.
2. Free functions operating on the above classes like `cuspatial.from_geopandas`.

The former should be organized into files named `bench_class.py`.
For example, benchmarks of `GeoDataFrame.sjoin` belong in `API/bench_geodataframe.py`.
Benchmarks should be written at the highest level of generality possible with respect to the class hierarchy.
For instance, all classes support the `take` method, so those benchmarks belong in `API/bench_frame_or_index.py`.

```{note}
`pytest` does not support having two benchmark files with the same name, even if they are in separate directories.
Therefore, benchmarks of internal methods of _public_ classes go in files suffixed with `_internal`.
Benchmarks of `GeoDataFrame.polygons.xy`, for instance, belong in `internal/bench_geodataframe_internal.py`.
```

Free functions have more flexibility.
Broadly speaking, they should be grouped into benchmark files containing similar functionality.
For example, I/O benchmarks can all live in `io/bench_io.py`.
For now those groupings are left to the discretion of developers.

## Running benchmarks

By default, pytest discovers test files and functions prefixed with `test_`.
For benchmarks, we configure `pytest` to instead search using the `bench_` prefix.
After installing `pytest-benchmark`, running benchmarks is as simple as just running `pytest`.

When benchmarks are run, the default behavior is to output the results in a table to the terminal.
A common requirement is to then compare the performance of benchmarks before and after a change.
We can generate these comparisons by saving the output using the `--benchmark-autosave` option to pytest.
When using this option, after the benchmarks are run the output will contain a line:
```
Saved benchmark data in: /path/to/XXXX_*.json
```

The `XXXX` is a four-digit number identifying the benchmark.
If preferred, a user may also use the `--benchmark-save=NAME` option,
which allows more control over the resulting filename.
Given two benchmark runs `XXXX` and `YYYY`, benchmarks can then be compared using
```
pytest-benchmark compare XXXX YYYY
```
Note that the comparison uses the `pytest-benchmark` command rather than the `pytest` command.
`pytest-benchmark` has a number of additional options that can be used to customize the output.
The next line contains one useful example, but developers should experiment to find a useful output
```
pytest-benchmark compare XXXX YYYY --sort="name" --columns=Mean --name=short --group-by=param
```

For more details, see the [`pytest-benchmark` documentation](https://pytest-benchmark.readthedocs.io/en/latest/comparing.html).

## Benchmark contents

### Benchmark configuration

Benchmarks must support [comparing to geopandas](#comparing-to-geopandas) and [being run as tests](#testing-benchmarks).
To satisfy these requirements, one must follow these rules when writing benchmarks:
1. Import `cuspatial` and `cupy` from the config module:
   ```python
       from ..common.config import cuspatial  # Do this
       import cuspatial  # Not this
   ```
   This enables swapping out for `geopandas`.
2. Avoid hard-coding benchmark dataset sizes, and instead use the sizes advertised by `config.py`.
   This enables running the benchmarks in "test" mode on small datasets, which will be much faster.


### Writing benchmarks 

Just as benchmarks should be written in terms of the highest level classes in the hierarchy,
they should also assume as little as possible about the nature of the data.


## Comparing to geopandas

An important aspect of benchmarking cuSpatial is comparing it to geopandas.
We often want to generate quantitative comparisons, so we need to make that as easy as possible.
Our benchmarks support this by setting the environment variable `CUSPATIAL_BENCHMARKS_USE_GEOPANDAS`.
When this variable is detected, all benchmarks will automatically be run using geopandas instead of cuspatial.
Therefore, comparisons can easily be generated by simply running the benchmarks twice,
once with the variable set and once without.
Note that this variable only affects API benchmarks, not internal benchmarks,
since the latter are not even guaranteed to be valid geopandas code.

```{note}
`CUSPATIAL_BENCHMARKS_USE_GEOPANDAS` effectively remaps `cuspatial` to `geopandas`.
It does so by aliasing these modules in `common.config.py`.
This aliasing is why it is critical for developers to import these packages from `config.py`.
```

## Testing benchmarks

Benchmarks need to be kept up to date with API changes in cuspatial.
However, we cannot simply run benchmarks in CI.
Doing so would consume too many resources, and it would significantly slow down the development cycle

To balance these issues, our benchmarks also support running in "testing" mode.
To do so, developers can set the `CUSPATIAL_BENCHMARKS_DEBUG_ONLY` environment variable.
When benchmarks are run with this variable, all data sizes are set to a minimum and the number of sizes are reduced.
Our CI testing takes advantage of this to ensure that benchmarks remain valid code.


## Profiling

Although not strictly part of our benchmarking suite, profiling is a common need so we provide some guidelines here.
Here are two easy ways (there may be others) to profile benchmarks:
1. The [`pytest-profiling`](https://github.com/man-group/pytest-plugins/tree/master/pytest-profiling) plugin.
2. The [`py-spy`](https://github.com/benfred/py-spy) package.

Using the former is as simple as adding the `--profile` (or `--profile-svg`) arguments to the `pytest` invocation.
The latter requires instead invoking pytest from py-spy, like so:
```
py-spy record -- pytest bench_foo.py
```
Each tool has different strengths and provides somewhat different information.
Developers should try both and see what works for a particular workflow.
Developers are also encouraged to share useful alternatives that they discover.
