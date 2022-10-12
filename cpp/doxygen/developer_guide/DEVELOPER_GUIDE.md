# libcuspatial C++ Developer Guide {#DEVELOPER_GUIDE}

This document serves as a guide for contributors to libcuspatial C++ code. Developers should also
refer to these additional files for further documentation of libcuspatial best practices.

* [Documentation Guide](DOCUMENTATION.md) for guidelines on documenting libcuspatial code.
* [Testing Guide](TESTING.md) for guidelines on writing unit tests.
* [Benchmarking Guide](BENCHMARKING.md) for guidelines on writing unit benchmarks.
* [Refactoring Guide](REFACTORING_GUIDE.md) for guidelines on refactoring legacy column-based APIs into 
                                            header-only APIs.

# Overview

libcuspatial is a C++ library that provides GPU-accelerated data-parallel algorithms for processing
geospatial and spatiotemporal data. libcuspatial provides various spatial relationship algorithms
including distance computation, containment (e.g. point-in-polygon testing), bounding box
computations, and spatial indexing.

libcuspatial currently has two interfaces. The first is a C++ API based on data types from 
libcudf, (the [CUDA Dataframe library C++ API](https://github.com/rapidsai/cudf/)). In this document
we refer to it as the "column-based API". The column-based API represents spatial data as tables of
type-erased columns. 

The second API is the cuSpatial header-only C++ API, which is independent of libcudf and represents
data as arrays of structures (e.g. 2D points). The header-only API uses iterators for input and
output, and is similar in style to the C++ Standard Template Library (STL) and
[Thrust](https://nvidia.github.io/thrust/).

## Lexicon

This section defines terminology used within libcuspatial. For terms specific to libcudf, such
as Column, Table, etc., see the 
[libcudf developer guide](https://github.com/rapidsai/cudf/blob/main/cpp/docs/DEVELOPER_GUIDE.md#lexicon).

TODO: add terms

# Directory Structure and File Naming

External/public libcuspatial APIs are grouped based on functionality into an appropriately titled
header file in `cuspatial/cpp/include/cuspatial/`. For example,
`cuspatial/cpp/include/cuspatial/coordinate_transform.hpp` contains the declarations of public API
functions related to transforming coordinates. Note the `.hpp` file extension used to indicate a
C++ header file that can be included from a `.cpp` source file.

Header files should use the `#pragma once` include guard.

The naming of public column-based cuSpatial API headers should be consistent with the name of the
folder that contains the source files that implement the API. For example, the implementation of the
APIs found in `cuspatial/cpp/include/cuspatial/trajectory.hpp` are located in
`cuspatial/src/trajectory`. This rule obviously does not apply to the header-only API, since the 
headers are the source files.

Likewise, unit tests reside in folders corresponding to the names of the API headers, e.g. 
trajectory.hpp tests are in `cuspatial/tests/trajectory/`. 

Internal API headers containing `detail` namespace definitions that are used across translation
units inside libcuspatial should be placed in `include/cuspatial/detail`.

Note that (currently) header-only API files are in `include/cuspatial/experimental`, and their tests
are in `tests/experimental`. When the header-only refactoring is complete these should be renamed or
split into a separate library.

## File extensions

- `.hpp` : C++ header files
- `.cpp` : C++ source files
- `.cu`  : CUDA C++ source files
- `.cuh` : Headers containing CUDA device code

Only use `.cu` and `.cuh` if necessary. A good indicator is the inclusion of `__device__` and other
symbols that are only recognized by `nvcc`. Another indicator is Thrust algorithm APIs with a device
execution policy (always `rmm::exec_policy` in libcuspatial).

## Code and Documentation Style and Formatting

libcuspatial code uses [snake_case](https://en.wikipedia.org/wiki/Snake_case) for all names except 
in a few cases: template parameters, unit tests and test case names may use Pascal case, aka
[UpperCamelCase](https://en.wikipedia.org/wiki/Camel_case). We do not use
[Hungarian notation](https://en.wikipedia.org/wiki/Hungarian_notation), except sometimes when naming
device data variables and their corresponding host copies (e.g. `d_data` and `h_data`). Private
member variables are typically prefixed with an underscore. 

Examples:

```c++
template <typename IteratorType>
void algorithm_function(int x, rmm::cuda_stream_view s, rmm::device_memory_resource* mr)
{
  ...
}

class utility_class
{
  ...
private:
  int _rating{};
  std::unique_ptr<rmm::device_uvector> _data{};
}

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_SUITE(RepeatTypedTestFixture, TestTypes);

TYPED_TEST(RepeatTypedTestFixture, RepeatScalarCount)
{
  ...
}
```

C++ formatting is enforced using `clang-format`. You should configure `clang-format` on your
machine to use the `cuspatial/cpp/.clang-format` configuration file, and run `clang-format` on all
changed code before committing it. The easiest way to do this is to configure your editor to
"format on save", or to use `pre-commit`.

Aspects of code style not discussed in this document and not automatically enforceable are typically
caught during code review, or not enforced.

### C++ Guidelines

In general, we recommend following
[C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines). We also
recommend watching Sean Parent's [C++ Seasoning talk](https://www.youtube.com/watch?v=W2tWOdzgXHA),
and we try to follow his rules: "No raw loops. No raw pointers. No raw synchronization primitives."

 * Prefer algorithms from STL and Thrust to raw loops.
 * For device storage, prefer libcudf and RMM
   [owning data structures and views](#libcuspatial-data-structures) to raw pointers and raw memory
   allocation. When pointers are used, prefer smart pointers (e.g. `std::shared_ptr` and
   `std::unique_ptr`) to raw pointers.
 * Prefer dispatching kernels to streams instead of explicit synchronization.

Documentation is discussed in the [Documentation Guide](DOCUMENTATION.md).

### Includes

The following guidelines apply to organizing `#include` lines.

 * Group includes by library (e.g. cuSpatial, RMM, Thrust, STL). `clang-format` will respect the
   groupings and sort the individual includes within a group lexicographically.
 * Separate groups by a blank line.
 * Order the groups from "nearest" to "farthest". In other words, local includes, then includes
   from other RAPIDS libraries, then includes from related libraries, like `<thrust/...>`, then
   includes from dependencies installed with cuSpatial, and then standard library headers (for 
   example `<string>`, `<iostream>`).
 * Use `<>` instead of `""` unless the header is in the same directory as the source file.
 * Tools like `clangd` often auto-insert includes when they can, but they usually get the grouping
   and brackets wrong.
 * Always check that includes are only necessary for the file in which they are included.
   Try to avoid excessive including especially in header files. Double check this when you remove
   code.
 * Use quotes `"` to include local headers from the same relative source directory. This should only
   occur in source files and non-public header files. Otherwise use angle brackets `<>` around
   included header filenames.
 * Avoid relative paths with `..` when possible. Paths with `..` are necessary when including
   (internal) headers from source paths not in the same directory as the including file,
   because source paths are not passed with `-I`.
 * Avoid including library internal headers from non-internal files. For example, try not to include
   headers from libcuspatial `src` directories in tests or in libcuspatial public headers. If you
   find yourself doing this, start a discussion about moving (parts of) the included internal header
   to a public header.

# libcuspatial Data Structures

The header-only libcuspatial API is agnostic to the type of containers used by the application to
hold its data, because the header-only API is based on iterators (see [Iterator Requirements](#iterator-requirements)). The cuDF-based cuSpatial API, on the other hand, uses cuDF Columns and
Tables to store and access application data. 

See the [libcudf Developer guide](https://github.com/rapidsai/cudf/blob/main/cpp/docs/DEVELOPER_GUIDE.md#libcudf-data-structures) for more information on cuDF data structures, including views.

## Views and Ownership

Resource ownership is an essential concept in libcudf, and therefore in the cuDF-based libcuspatial 
API. In short, an "owning" object owns a resource (such as device memory). It acquires that resource
during construction and releases the resource in destruction 
([RAII](https://en.cppreference.com/w/cpp/language/raii)). A "non-owning" object does not own
resources. Any class in libcudf with the `*_view` suffix is non-owning. For more detail see the
[`libcudf++` presentation.](https://docs.google.com/presentation/d/1zKzAtc1AWFKfMhiUlV5yRZxSiPLwsObxMlWRWz_f5hA/edit?usp=sharing)

cuDF-based libcuspatial functions typically take views as input (`column_view` or `table_view`)
and produce `unique_ptr`s to owning objects as output. For example,

```c++
std::unique_ptr<cudf::table> points_in_spatial_window(
  ...,
  cudf::column_view const& x,
  cudf::column_view const& y);
```

## RMM Memory Resources (`rmm::device_memory_resource`)

libcuspatial allocates all device memory via RMM memory resources (MR). See the
[RMM documentation](https://github.com/rapidsai/rmm/blob/main/README.md) for details.

### Current Device Memory Resource

RMM provides a "default" memory resource for each device that can be accessed and updated via the
`rmm::mr::get_current_device_resource()` and `rmm::mr::set_current_device_resource(...)` functions,
respectively. All memory resource parameters should be defaulted to use the return value of
`rmm::mr::get_current_device_resource()`.

# libcuspatial API and Implementation

This section provides specifics about the structure and implementation of cuSpatial API functions.

## Column-based cuSpatial API

libcuspatial's column-based API is designed to integrate seamlessly with other RAPIDS libraries,
notably cuDF. To that end, this API uses `cudf::column` and `cudf::table` data structures as input
and output. This enables cuSpatial to provide Python and other language APIs (e.g. Java) that
integrate seamlessly with the APIs of other RAPIDS libraries like cuDF and cuML. This allows users
to integrate spatial data queries and transformations into end-to-end GPU-accelerated data analytics
and machine learning workflows.

### Input/Output Style

The preferred style for passing input to and returning output from column-based API functions is the
following:

- Input parameters
  - Columns:
    - `column_view const&`
  - Tables:
    - `table_view const&`
  - Scalar:
    - `scalar const&`
  - Everything else:
    - Trivial or inexpensively copied types
      - Pass by value
    - Non-trivial or expensive to copy types
      - Pass by `const&`
- Input/Output Parameters
  - Columns:
    - `mutable_column_view&`
  - Tables:
    - `mutable_table_view&`
  - Everything else:
    - Pass via raw pointer
- Output
  - Outputs should be *returned*, i.e., no output parameters
  - Columns:
    - `std::unique_ptr<column>`
  - Tables:
    - `std::unique_ptr<table>`
  - Scalars:
    - `std::unique_ptr<scalar>`

Here is an example column-based API function.

```C++
std::unique_ptr<cudf::column> haversine_distance(
  cudf::column_view const& a_lon,
  cudf::column_view const& a_lat,
  cudf::column_view const& b_lon,
  cudf::column_view const& b_lat,
  double const radius                 = EARTH_RADIUS_KM,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
```

key points:
  1. All input data are `cudf::column_view`. This is a type-erased container so determining the 
     type of data must be done at run time.
  2. All inputs are arrays of scalars. Longitude and latitude are separate. 
  3. The output is a returned `unique_ptr<cudf::column>`.
  4. The output is allocated inside the function using the passed memory resource.
  5. The public API does not take a stream. There is a `detail` version of the API that takes a
     stream. This follows libcudf, and may change in the future.

### Multiple Return Values

Sometimes it is necessary for functions to have multiple outputs. There are a few ways this can be
done in C++ (including creating a `struct` for the output). One convenient way to do this is
using `std::tie`  and `std::pair`. Note that objects passed to `std::pair` will invoke
either the copy constructor or the move constructor of the object, and it may be preferable to move
non-trivially copyable objects (and required for types with deleted copy constructors, like
`std::unique_ptr`).

Multiple column outputs that are functionally related (e.g. x- and y-coordinates), should be 
combined into a `table`.

```c++
std::pair<cudf::table, cudf::table> return_two_tables(void){
  cudf::table out0;
  cudf::table out1;
  ...
  // Do stuff with out0, out1

  // Return a std::pair of the two outputs
  return std::pair(std::move(out0), std::move(out1));
}

cudf::table out0;
cudf::table out1;
std::tie(out0, out1) = return_two_outputs();
```

Note: `std::tuple` _could_ be used if not for the fact that Cython does not support `std::tuple`
Therefore, libcuspatial public column-based APIs must use `std::pair`, and are therefore limited to
return only two objects of different types. Multiple objects of the same type may be returned via a
`std::vector<T>`.

Alternatively, C++17
[structured binding](https://en.cppreference.com/w/cpp/language/structured_binding) may be used to
disaggregate multiple return values:

```c++
auto [out0, out1] = return_two_outputs();
```

Note that the compiler might not support capturing aliases defined in a structured binding in a
lambda. One may work around this by using a capture with an initializer instead:

```c++
auto [out0, out1] = return_two_outputs();

// Direct capture of alias from structured binding might fail with:
// "error: structured binding cannot be captured"
// auto foo = [out0]() {...};

// Use an initializing capture:
auto foo = [&out0 = out0] {
  // Use out0 to compute something.
  // ...
};
```

## Header-only cuSpatial API

For C++ users and developers who do not also use libcudf or other RAPIDS APIS, depending on libcudf
could be a barrier to adoption of libcuspatial. libcudf is a very large library and building it
takes a lot of time.

Therefore, libcuspatial provides a header-only C++ API that does not depend on libcudf. The
header-only API is an iterator-based interface. This has a number of advantages.

  1. With a header-only API, users can include and build exactly what they use.
  2. A template API can flexibly support a variety of basic data types, such as float and double for
     positional data, and different integer sizes for indices.
  3. As with STL, iterators enable generic algorithms to be applied to arbitrary containers.
  4. Iterators enable cuSpatial algorithms to be fused with transformations of the
     input data, by using "fancy" iterators. Examples include transform iterators and counting
     iterators.
  5. Iterators enable the header-only cuSpatial API to use structured coordinate data (e.g. x,y 
     coordinate pairs) while maintaining compatibility with the separate arrays of x and y 
     coordinates required by the column-based API. This is accomplished with zip iterators. 
     Internally, structured data (with arithmetic operators) enables clearer, more arithmetic code.
  6. Memory resources only need to be part of APIs that allocate temporary intermediate storage.
     Output storage is allocated outside the API and an output iterator is passed as an argument. 

The main disadvantages of this type of API are

  1. Header-only APIs can increase compilation time for code that depends on them.
  2. Some users (especially the cuSpatial Python API) may prefer a cuDF-based API.

The column-based C++ API is a simple layer above the header-only API. This approach protects
column-based API users from the disadvantages while maintaining the advantages for users of the
header-only API.

### Input/Output Style

All array inputs and outputs are iterator type templates to enable generic application of the APIs.
An example function is helpful.

```C++
template <class LonLatItA,
          class LonLatItB,
          class OutputIt,
          class T        = typename cuspatial::iterator_vec_base_type<LonLatItA>>
OutputIt haversine_distance(LonLatItA a_lonlat_first,
                            LonLatItA a_lonlat_last,
                            LonLatItB b_lonlat_first,
                            OutputIt distance_first,
                            T const radius               = EARTH_RADIUS_KM,
                            rmm::cuda_stream_view stream = rmm::cuda_stream_default);
```

There are a few key points to notice.

  1. The API is very similar to STL algorithms such as `std::transform`.
  2. All array inputs and outputs are iterator type templates. 
  3. Longitude/Latitude data is passed as array of structures, using the `cuspatial::vec_2d<T>`
     type (include/cuspatial/vec_2d.hpp). This is enforced using a `static_assert` in the function
     body.
  5. The floating point type is a template (`T`) that is by default equal to the base `value_type`
     of the type iterated over by `LonLatItA`. libcuspatial provides the `iterator_vec_base_type`
     trait helper for this.
  6. The iterator types for the two input ranges (A and B) are distinct templates. This is crucial
     to enable composition of fancy iterators that may be different types for A and B.
  7. The size of the input and output ranges in the example API are equal, so the start and end of
     only the A range is provided (`a_lonlat_first` and `a_lonlat_last`). This mirrors STL APIs.
  8. This API returns an iterator to the element past the last element written to the output. This
     is inspired by `std::transform`, even though as with `transform`, many uses of 
     cuSpatial APIs will not need to use this returned iterator.
  9. All APIs that run CUDA device code (including Thrust algorithms) or allocate memory take a CUDA
     stream on which to execute the device code and allocate memory.
  10. Any API that allocate and return device data (not shown here) should also take an 
      `rmm::device_memory_resource` to use for output memory allocation.

### (Multiple) Return Values

Whenever possible in the header-only API, output data should be written to output iterators
that reference data allocated by the caller of the API. In this case, multiple "return values"
are simply written to multiple output iterators. Typically such APIs return an iterator one 
past the end of the primary output iterator (in the style of `std::transform()`.

In functions where the output size is data dependent, the API may allocate the output data and
return it as a `rmm::device_uvector` or other data structure containing `device_uvector`s.

### Iterator requirements

All input and output iterators must be device-accessible with random access. They must satisfy the
requirements of C++ [LegacyRandomAccessIterator][LinkLRAI]. Output iterators must be mutable.

## Streams

CUDA streams are not yet exposed in public column-based libcuspatial APIs. header-only libcuspatial
APIs that execute GPU work or allocate GPU memory should take a stream parameter.

In order to ease the transition to future use of streams in the public column-based API, all
libcuspatial APIs that allocate device memory or execute GPU work (including kernels, 
Thrust algorithms, or anything that can take a stream) should be implemented using asynchronous APIs
on the default stream (e.g., stream 0).

The recommended pattern for doing this is to make the definition of the external API invoke an
internal API in the `detail` namespace. The internal `detail` API has the same parameters as the
public API, plus a `rmm::cuda_stream_view` parameter at the end with no default value. If the
detail API also accepts a memory resource parameter, the stream parameter should be ideally placed
just *before* the memory resource. The public API will call the detail API and provide
`rmm::cuda_stream_default`. The implementation should be wholly contained in the `detail` API
definition and use only asynchronous versions of CUDA APIs with the stream parameter.

In order to make the `detail` API callable from other libcuspatial functions, it may be exposed in a
header placed in the `cuspatial/cpp/include/detail/` directory.

For example:

```c++
// cpp/include/cuspatial/header.hpp
void external_function(...);

// cpp/include/cuspatial/detail/header.hpp
namespace detail{
void external_function(..., rmm::cuda_stream_view stream)
} // namespace detail

// cuspatial/src/implementation.cpp
namespace detail{
    // Use the stream parameter in the detail implementation.
    void external_function(..., rmm::cuda_stream_view stream){
        // Implementation uses the stream with async APIs.
        rmm::device_buffer buff(...,stream);
        CUSPATIAL_CUDA_TRY(cudaMemcpyAsync(...,stream.value()));
        kernel<<<..., stream>>>(...);
        thrust::algorithm(rmm::exec_policy(stream), ...);
    }
} // namespace detail

void external_function(...){
    CUSPATIAL_FUNC_RANGE(); // Generates an NVTX range for the lifetime of this function.
    detail::external_function(..., rmm::cuda_stream_default);
}
```

**Note:** It is important to synchronize the stream if *and only if* it is necessary. For example,
when a non-pointer value is returned from the API that is the result of an asynchronous
device-to-host copy, the stream used for the copy should be synchronized before returning. However,
when a column is returned, the stream should not be synchronized because doing so will break
asynchrony if and when we add an asynchronous API to libcuspatial.

**Note:** `cudaDeviceSynchronize()` should *never* be used.
This limits the ability to do any multi-stream/multi-threaded work with libcuspatial APIs.

 ### NVTX Ranges

In order to aid in performance optimization and debugging, all compute intensive libcuspatial 
functions should have a corresponding NVTX range. In libcuspatial, we have a convenience macro
`CUSPATIAL_FUNC_RANGE()` that automatically annotates the lifetime of the enclosing function and
uses the function's name as the name of the NVTX range. For more information about NVTX, see
[here](https://github.com/NVIDIA/NVTX/tree/dev/cpp).

### Stream Creation

(Note: cuSpatial has not yet had the need for internal stream creation.) The following guidance is
copied from libcudf's documentation. There may be times in implementing libcuspatial features where
it would be advantageous to use streams *internally*, i.e., to accomplish overlap in implementing an
algorithm. However, dynamically creating a stream can be expensive. RMM has a stream pool class to
help avoid dynamic stream creation. However, this is not yet exposed in libcuspatial, so for the
time being, libcuspatial features should avoid creating streams (even if it is slightly less
efficient). It is a good idea to leave a `// TODO:` note indicating where using a stream would be
beneficial.

## Memory Allocation

Device [memory resources](#rmmdevice_memory_resource) are used in libcuspatial to abstract and
control how device memory is allocated.

### Output Memory

Any libcuspatial API that allocates memory that is *returned* to a user must accept a pointer to a
`device_memory_resource` as the last parameter. Inside the API, this memory resource must be used
to allocate any memory for returned objects. It should therefore be passed into functions whose
outputs will be returned. Example:

```c++
// Returned `column` contains newly allocated memory,
// therefore the API must accept a memory resource pointer
std::unique_ptr<column> returns_output_memory(
  ..., rmm::device_memory_resource * mr = rmm::mr::get_current_device_resource());

// This API does not allocate any new *output* memory, therefore
// a memory resource is unnecessary
void does_not_allocate_output_memory(...);
```

### Temporary Memory

Not all memory allocated within a libcuspatial API is returned to the caller. Often algorithms must
allocate temporary, scratch memory for intermediate results. Always use the default resource
obtained from `rmm::mr::get_current_device_resource()` for temporary memory allocations. Example:

```c++
rmm::device_buffer some_function(
  ..., rmm::mr::device_memory_resource mr * = rmm::mr::get_current_device_resource()) {
    rmm::device_buffer returned_buffer(..., mr); // Returned buffer uses the passed in MR
    ...
    rmm::device_buffer temporary_buffer(...); // Temporary buffer uses default MR
    ...
    return returned_buffer;
}
```

### Memory Management

libcuspatial code eschews raw pointers and direct memory allocation. Use RMM classes built to
use [`device_memory_resource`](https://github.com/rapidsai/rmm/#device_memory_resource) for device
memory allocation with automated lifetime management.

#### rmm::device_buffer
Allocates a specified number of bytes of untyped, uninitialized device memory using a
`device_memory_resource`. If no resource is explicitly provided, uses
`rmm::mr::get_current_device_resource()`.

`rmm::device_buffer` is movable and copyable on a stream. A copy performs a deep copy of the
`device_buffer`'s device memory on the specified stream, whereas a move moves ownership of the
device memory from one `device_buffer` to another.

```c++
// Allocates at least 100 bytes of uninitialized device memory
// using the specified resource and stream
rmm::device_buffer buff(100, stream, mr);
void * raw_data = buff.data(); // Raw pointer to underlying device memory

// Deep copies `buff` into `copy` on `stream`
rmm::device_buffer copy(buff, stream);

// Moves contents of `buff` into `moved_to`
rmm::device_buffer moved_to(std::move(buff));

custom_memory_resource *mr...;
// Allocates 100 bytes from the custom_memory_resource
rmm::device_buffer custom_buff(100, mr, stream);
```

#### rmm::device_scalar<T>
Allocates a single element of the specified type initialized to the specified value. Use this for
scalar input/outputs into device kernels, e.g., reduction results, null count, etc. This is
effectively a convenience wrapper around a `rmm::device_vector<T>` of length 1.

```c++
// Allocates device memory for a single int using the specified resource and stream
// and initializes the value to 42
rmm::device_scalar<int> int_scalar{42, stream, mr};

// scalar.data() returns pointer to value in device memory
kernel<<<...>>>(int_scalar.data(),...);

// scalar.value() synchronizes the scalar's stream and copies the
// value from device to host and returns the value
int host_value = int_scalar.value();
```

#### rmm::device_vector<T>

Allocates a specified number of elements of the specified type. If no initialization value is
provided, all elements are default initialized (this incurs a kernel launch).

**Note**: (TODO: this not true yet in libcuspatial but we should strive for it. The following is 
copied from libcudf's developer guide.)
We have removed all usage of `rmm::device_vector` and `thrust::device_vector` from
libcuspatial, and you should not use it in new code in libcuspatial without careful consideration.
Instead, use `rmm::device_uvector` along with the utility factories in `device_factories.hpp`. These
utilities enable creation of `uvector`s from host-side vectors, or creating zero-initialized
`uvector`s, so that they are as convenient to use as `device_vector`. Avoiding `device_vector` has
a number of benefits, as described in the following section on `rmm::device_uvector`.

#### rmm::device_uvector<T>

Similar to a `device_vector`, allocates a contiguous set of elements in device memory but with key
differences:
- As an optimization, elements are uninitialized and no synchronization occurs at construction.
This limits the types `T` to trivially copyable types.
- All operations are stream-ordered (i.e., they accept a `cuda_stream_view` specifying the stream
on which the operation is performed). This improves safety when using non-default streams.
- `device_uvector.hpp` does not include any `__device__` code, unlike `thrust/device_vector.hpp`,
  which means `device_uvector`s can be used in `.cpp` files, rather than just in `.cu` files.

```c++
cuda_stream s;
// Allocates uninitialized storage for 100 `int32_t` elements on stream `s` using the
// default resource
rmm::device_uvector<int32_t> v(100, s);
// Initializes the elements to 0
thrust::uninitialized_fill(thrust::cuda::par.on(s.value()), v.begin(), v.end(), int32_t{0});

rmm::mr::device_memory_resource * mr = new my_custom_resource{...};
// Allocates uninitialized storage for 100 `int32_t` elements on stream `s` using the resource `mr`
rmm::device_uvector<int32_t> v2{100, s, mr};
```

## Namespaces

### External
All public libcuspatial APIs should be placed in the `cuspatial` namespace. Example:
```c++
namespace cuspatial{
   void public_function(...);
} // namespace cuspatial
```

The top-level `cuspatial` namespace is sufficient for most of the public API. However, to logically
group a broad set of functions, further namespaces may be used.

### Internal

Many functions are not meant for public use, so place them in either the `detail` or an *anonymous*
namespace, depending on the situation.

#### detail namespace

Functions or objects that will be used across *multiple* translation units (i.e., source files),
should be exposed in an internal header file and placed in the `detail` namespace. Example:

```c++
// some_utilities.hpp
namespace cuspatial{
namespace detail{
void reusable_helper_function(...);
} // namespace detail
} // namespace cuspatial
```

#### Anonymous namespace

Functions or objects that will only be used in a *single* translation unit should be defined in an
*anonymous* namespace in the source file where it is used. Example:

```c++
// some_file.cpp
namespace{
void isolated_helper_function(...);
} // anonymous namespace
```

[**Anonymous namespaces should *never* be used in a header file.**](https://wiki.sei.cmu.edu/confluence/display/cplusplus/DCL59-CPP.+Do+not+define+an+unnamed+namespace+in+a+header+file)

# Deprecating and Removing Code

libcuspatial is constantly evolving to improve performance and better meet our users' needs. As a
result, we occasionally need to break or entirely remove APIs to respond to new and improved
understanding of the functionality we provide. Remaining free to do this is essential to making
libcuspatial an agile library that can rapidly accommodate our users needs. As a result, we do not
always provide a warning or any lead time prior to releasing breaking changes. On a best effort
basis, the libcuspatial team will notify users of changes that we expect to have significant or
widespread effects.

Where possible, indicate pending API removals using the
[deprecated](https://en.cppreference.com/w/cpp/language/attributes/deprecated) attribute and
document them using Doxygen's
[deprecated](https://www.doxygen.nl/manual/commands.html#cmddeprecated) command prior to removal.
When a replacement API is available for a deprecated API, mention the replacement in both the
deprecation message and the deprecation documentation. Pull requests that introduce deprecations
should be labeled "deprecation" to facilitate discovery and removal in the subsequent release.

Advertise breaking changes by labeling any pull request that breaks or removes an existing API with
the "breaking" tag. This ensures that the "Breaking" section of the release notes includes a
description of what has broken from the past release. Label pull requests that contain deprecations
with the "non-breaking" tag.

# Error Handling

libcuspatial follows conventions (and provides utilities) enforcing compile-time and run-time
conditions and detecting and handling CUDA errors. Communication of errors is always via C++
exceptions.

## Runtime Conditions

Use the `CUSPATIAL_EXPECTS` macro to enforce runtime conditions necessary for correct execution.

Example usage:
```c++
CUSPATIAL_EXPECTS(lhs.type() == rhs.type(), "Column type mismatch");
```

The first argument is the conditional expression expected to resolve to `true` under normal
conditions. If the conditional evaluates to `false`, then an error has occurred and an instance of
`cuspatial::logic_error` is thrown. The second argument to `CUSPATIAL_EXPECTS` is a short 
description of the error that has occurred and is used for the exception's `what()` message.

There are times where a particular code path, if reached, should indicate an error no matter what.
For example, often the `default` case of a `switch` statement represents an invalid alternative.
Use the `CUSPATIAL_FAIL` macro for such errors. This is effectively the same as calling
`CUSPATIAL_EXPECTS(false, reason)`.

Example:
```c++
CUSPATIAL_FAIL("This code path should not be reached.");
```

### CUDA Error Checking

Use the `CUSPATIAL_CUDA_TRY` macro to check for the successful completion of CUDA runtime API 
functions. This macro throws a `cuspatial::cuda_error` exception if the CUDA API return value is not
`cudaSuccess`. The thrown exception includes a description of the CUDA error code in its `what()`
message.

Example:

```c++
CUSPATIAL_CUDA_TRY( cudaMemcpy(&dst, &src, num_bytes) );
```

## Compile-Time Conditions

Use `static_assert` to enforce compile-time conditions. For example,

```c++
template <typename T>
void trivial_types_only(T t){
  static_assert(std::is_trivial<T>::value, "This function requires a trivial type.");
...
}
```

# Data Types

Columns may contain data of a number of types. cuDF supports a variety of types that are not used
in cuSpatial. cuSpatial functions mostly operate on numeric and timestamp data. For more information
on libcudf data types see the
[libcudf developer guide](https://github.com/rapidsai/cudf/blob/main/cpp/docs/DEVELOPER_GUIDE.md#data-types).

# Type Dispatcher

`cudf::column` stores data (for columns and scalars) "type erased" in `void*` device memory. This
*type-erasure* enables interoperability with other languages and type systems, such as Python and
Java. In order to determine the type, functions must use the run-time information stored in the
column `type()` to reconstruct the data type `T` by casting the `void*` to the appropriate
`T*`.

This so-called *type dispatch* is pervasive throughout libcudf and the column-based libcuspatial
API. The `cudf::type_dispatcher` is a central utility that automates the process of mapping the
runtime type information in `data_type` to a concrete C++ type. See the
[libcudf developer guide](https://github.com/rapidsai/cudf/blob/main/cpp/docs/DEVELOPER_GUIDE.md#type-dispatcher)
for more information.

[LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator "LegacyRandomAccessIterator"
