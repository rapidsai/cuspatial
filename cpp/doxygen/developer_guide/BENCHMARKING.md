# Unit Benchmarking in libcuspatial

Unit benchmarks in libcuspatial are written using [NVBench](https://github.com/NVIDIA/nvbench).
While some existing benchmarks are written using
[Google Benchmark](https://github.com/google/benchmark), new benchmarks should use NVBench.

The NVBench library is similar to Google Benchmark, but has several quality of life improvements
when doing GPU benchmarking such as displaying the fraction of peak memory bandwidth achieved and
details about the GPU hardware.

Both NVBench and Google Benchmark provide many options for specifying ranges of parameters to
benchmark, as well as to control the time unit reported, among other options. Refer to existing
benchmarks in `cpp/benchmarks` to understand the options.

## Directory and File Naming

The naming of unit benchmark directories and source files should be consistent with the feature
being benchmarked. For example, the benchmarks for APIs in `point_in_polygon.hpp` should live in
`cpp/benchmarks/point_in_polygon.cu`. Each feature (or set of related features) should have its own
benchmark source file named `<feature>{.cu,cpp}`. 

## CUDA Asynchrony and benchmark accuracy

CUDA computations and operations like copies are typically asynchronous with respect to host code,
so it is important to carefully synchronize in order to ensure the benchmark timing is not stopped
before the feature you are benchmarking has completed. An RAII helper class `cuda_event_timer` is
provided in `cpp/benchmarks/synchronization/synchronization.hpp` to help with this. This class
can also optionally clear the GPU L2 cache in order to ensure cache hits do not artificially
inflate performance in repeated iterations.

## Data generation

For generating benchmark input data, random data generation functions are provided in
`cpp/benchmarks/utility/random.cuh`. The input data generation happens on device.

## What should we benchmark?

In general, we should benchmark all features over a range of data sizes and types, so that we can
catch regressions across libcudf changes. However, running many benchmarks is expensive, so ideally
we should sample the parameter space in such a way to get good coverage without having to test
exhaustively.

A rule of thumb is that we should benchmark with enough data to reach the point where the algorithm
reaches its saturation bottleneck, whether that bottleneck is bandwidth or computation. Using data
sets larger than this point is generally not helpful, except in specific cases where doing so
exercises different code and can therefore uncover regressions that smaller benchmarks will not
(this should be rare).


Generally we should benchmark public APIs. Benchmarking detail functions and/or internal utilities
should only be done if detecting regressions in them would be sufficiently difficult to do from
public API benchmarks.
