# cuSpatial C++ API Refactoring Guide

The original cuSpatial C++ API (libcuspatial) was designed to depend on RAPIDS libcudf and use
its core data types, especially `cudf::column`. For users who do not also use libcudf or other 
RAPIDS APIS, depending on libcudf could be a big barrier to adoption of libcuspatial. libcudf is 
a very large library and building it takes a lot of time.

Therefore, we are developing a standalone libcuspatial C++ API that does not depend on libcudf. This 
is a header-only template API with an iterator-based interface. This has a number of advantages

  1. With a header-only API, users can include and build exactly what they use.
  2. With a templated API, the API can be flexible to support a variety of basic data types, such 
     as float and double for positional data, and different integer sizes for indices.
  3. By templating on iterator types, cuSpatial algorithms can be fused with transformations of the
     input data, by using "fancy" iterators. Examples include transform iterators and counting
     iterators.
  4. Memory resources only need to be part of APIs that allocate temporary intermediate storage.
     Output storage is allocated outside the API and an output iterator is passed as an argument. 

The main disadvantages of this type of API are

  1. Header-only APIs can increase compilation time for code that depends on them.
  2. Some users (especially our Python API) may prefer a cuDF-based API.

The good news is that by maintaining the existing libcudf-based C++ API as a layer above the header-
only libcuspatial API, we can avoid problem 1 and problem 2 for users of the legacy API.

## Example API

Following is an example iterator-based API for `cuspatial::haversine_distance`. (See below for 
discussion of API documentation.)

```c++
template <class LonLatItA,
          class LonLatItB,
          class OutputIt,
          class Location = typename std::iterator_traits<LonLatItA>::value_type,
          class T        = typename Location::value_type>
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
  3. Longitude/Latitude data is passed as array of structures, using the `cuspatial::vec_2d`
     type (include/cuspatial/vec_2d.hpp). This is enforced using a `static_assert` in the function
     body (discussed later).
  4. The `Location` type is a template that is by default equal to the `value_type` of the input
     iterators.
  5. The floating point type is a template (`T`) that is by default equal to the `value_type` of
     `Location`.
  6. The iterator types for the two input ranges (A and B) are distinct templates. This is crucial
     to enable composition of fancy iterators that may be different types for A and B.
  7. The size of the input and output ranges in the example API are equal, so the start and end of
     only the A range is provided (`a_lonlat_first` and `a_lonlat_last`). This mirrors STL APIs.
  8. This API returns an iterator to the element past the last element written to the output. This
     is inspired by `std::transform`, even though as with `transform`, many uses of 
     `haversine_distance` will not need this returned iterator.
  9. All APIs that run CUDA device code (including Thrust algorithms) or allocate memory take a CUDA
     stream on which to execute the device code and allocate memory.

## Example Documentation

Following is the (Doxygen) documentation for the above `cuspatial::haversine_distance`.

    /**
     * @brief Compute haversine distances between points in set A to the corresponding points in set B.
     *
     * Computes N haversine distances, where N is `std::distance(a_lonlat_first, a_lonlat_last)`.
     * The distance for each `a_lonlat[i]` and `b_lonlat[i]` point pair is assigned to
     * `distance_first[i]`. `distance_first` must be an iterator to output storage allocated for N
     * distances.
     *
     * Computed distances will have the same units as `radius`.
     *
     * https://en.wikipedia.org/wiki/Haversine_formula
     *
     * @param[in]  a_lonlat_first: beginning of range of (longitude, latitude) locations in set A
     * @param[in]  a_lonlat_last: end of range of (longitude, latitude) locations in set A
     * @param[in]  b_lonlat_first: beginning of range of (longitude, latitude) locations in set B
     * @param[out] distance_first: beginning of output range of haversine distances
     * @param[in]  radius: radius of the sphere on which the points reside. default: 6371.0
     *            (approximate radius of Earth in km)
     * @param[in]  stream: The CUDA stream on which to perform computations and allocate memory.
     *
     * @tparam LonLatItA Iterator to input location set A. Must meet the requirements of
     * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
     * @tparam LonLatItB Iterator to input location set B. Must meet the requirements of
     * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
     * @tparam OutputIt Output iterator. Must meet the requirements of
     * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
     * @tparam Location The `value_type` of `LonLatItA` and `LonLatItB`. Must be `cuspatial::vec_2d<T>`.
     * @tparam T The underlying coordinate type. Must be a floating-point type.
     *
     * @pre `a_lonlat_first` may equal `distance_first`, but the range `[a_lonlat_first, a_lonlat_last)`
     * shall not overlap the range `[distance_first, distance_first + (a_lonlat_last - a_lonlat_last))
     * otherwise.
     * @pre `b_lonlat_first` may equal `distance_first`, but the range `[b_lonlat_first, b_lonlat_last)`
     * shall not overlap the range `[distance_first, distance_first + (b_lonlat_last - b_lonlat_last))
     * otherwise. 
     * @pre All iterators must have the same `Location` type, with  the same underlying floating-point
     * coordinate type (e.g. `cuspatial::vec_2d<float>`).
     *
     * @return Output iterator to the element past the last distance computed.
     *
     * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
     * "LegacyRandomAccessIterator"
     */

Key points:

  1. Precisely and succinctly documents what the API computes, and provides references.
  2. All parameters and all template parameters are documented.
  3. States the C++ standard iterator concepts that must be implemented, and that iterators must be
     device-accessible.
  4. Documents requirements as preconditions using `@pre`. 
  5. Uses preconditions to explicitly document what input ranges are allowed to overlap.
  6. Documents the units of any inputs or outputs that have them.

## cuSpatial libcudf-based C++ API (legacy API)

This is the existing API, unchanged by refactoring. Here is the existing 
`cuspatial::haversine_distance`:

```c++
template <class LonLatItA,
          class LonLatItB,
          class OutputIt,
          class T = typename cuspatial::iterator_vec_base_type<LonLatItA>>
OutputIt haversine_distance(LonLatItA a_lonlat_first,
                            LonLatItA a_lonlat_last,
                            LonLatItB b_lonlat_first,
                            OutputIt distance_first,
                            T const radius               = EARTH_RADIUS_KM,
                            rmm::cuda_stream_view stream = rmm::cuda_stream_default);
```

key points:
  1. All input data are `cudf::column_view`. This is a type-erased container so determining the 
     type of data must be done at run time.
  2. All inputs are arrays of scalars. Longitude and latitude are separate. 
  3. The output is a returned `unique_ptr<cudf::column>`.
  4. The output is allocated inside the function using the passed memory resource.
  5. The public API does not take a stream. There is a `detail` version of the API that takes a
     stream. This follows libcudf, and may change in the future.

## File Structure

For now, libcuspatial APIs should be defined in a header file in the
`cpp/include/cuspatial/experimental/` directory. Later, as we adopt the new API, we will rename
the `experimental` directory. The API header should be named after the API. In the example, 
`haversine.hpp` defines the `cuspatial::haversine_distance` API.

The implementation must also be in a header, but should be in the `cuspatial/experimental/detail`
directory.  The implementation should be included from the API definition file, at the end of the 
file. Example:

```c++
... // declaration of API above this point
#include <cuspatial/experimental/detail/haversine.hpp>
```

## Namespaces

Public APIs are in the `cuspatial` namespace. Note that both the header-only API and the libcudf-
based API can live in the same namespace, because they are non-ambiguous (very different 
parameters).

Implementation of the header-only API should be in a `cuspatial::detail` namespace.

## Implementation

The main implementation should be in detail headers. 

### Header-only API Implementation

Because it is a statically typed API, the header-only implementation can be much simpler than the 
libcudf-based API, which requires run-time type dispatching. In the case of `haversine_distance`, it is
a simple matter of a few static asserts and dynamic expectation checks, followed by a call to
`thrust::transform` with a custom transform functor.

```c++
template <class LonLatItA, class LonLatItB, class OutputIt, class T>
OutputIt haversine_distance(LonLatItA a_lonlat_first,
                            LonLatItA a_lonlat_last,
                            LonLatItB b_lonlat_first,
                            OutputIt distance_first,
                            T const radius,
                            rmm::cuda_stream_view stream)
{
  static_assert(is_same<vec_2d<T>,
                                cuspatial::iterator_value_type<LonLatItA>,
                                cuspatial::iterator_value_type<LonLatItB>>(),
                "Inputs must be cuspatial::vec_2d");
  static_assert(
    is_same_floating_point<T,
                                   typename cuspatial::iterator_vec_base_type<LonLatItA>,
                                   typename cuspatial::iterator_value_type<OutputIt>>(),
    "All iterator types and radius must have the same floating-point coordinate value type.");

  CUSPATIAL_EXPECTS(radius > 0, "radius must be positive.");

  return thrust::transform(rmm::exec_policy(stream),
                           a_lonlat_first,
                           a_lonlat_last,
                           b_lonlat_first,
                           distance_first,
                           detail::haversine_distance_functor<T>(radius));
}
```

Note that we `static_assert` that the types of the iterator inputs match documented expectations.
We also do a runtime check that the radius is positive. Finally we just call `thrust::transform`, 
passing it an instance of `haversine_distance_functor`, which is a function of two `vec_2d<T>`
inputs that implements the Haversine distance formula.

### libcudf-based API Implementation

The substance of the refactoring is making the libcudf-based API a wrapper around the header-only 
API. This mostly involves replacing business logic implementation in the type-dispatched functor 
with a call to the header-only API. We also need to convert disjoint latitude and longitude inputs 
into `vec_2d<T>` structs. This is easily done using the `cuspatial::make_vec_2d_iterator` utility
provided in `type_utils.hpp`. 

So, to refactor the libcudf-based API, we remove the following code.

```c++
auto input_tuple = thrust::make_tuple(thrust::make_constant_iterator(static_cast<T>(radius)),
                                      a_lon.begin<T>(),
                                      a_lat.begin<T>(),
                                      b_lon.begin<T>(),
                                      b_lat.begin<T>());

auto input_iter = thrust::make_zip_iterator(input_tuple);

thrust::transform(rmm::exec_policy(stream),
                  input_iter,
                  input_iter + result->size(),
                  result->mutable_view().begin<T>(),
                  [] __device__(auto inputs) {
                    return calculate_haversine_distance(thrust::get<0>(inputs),
                                                        thrust::get<1>(inputs),
                                                        thrust::get<2>(inputs),
                                                        thrust::get<3>(inputs),
                                                        thrust::get<4>(inputs));
                  });
```

And replace it with the following code.

```c++
auto lonlat_a = cuspatial::make_vec_2d_iterator(a_lon.begin<T>(), a_lat.begin<T>());
auto lonlat_b = cuspatial::make_vec_2d_iterator(b_lon.begin<T>(), b_lat.begin<T>());

cuspatial::haversine_distance(lonlat_a,
                              lonlat_a + a_lon.size(),
                              lonlat_b,
                              static_cast<cudf::mutable_column_view>(*result).begin<T>(),
                              T{radius},
                              stream);
```

## Testing

Existing libcudf-based API tests can mostly be left alone. New tests should be added to exercise
the header-only API separately in case the libcudf-based API is removed.

Note that tests, like the header-only API, should not depend on libcudf or libcudf_test. The 
cuDF-based API made the mistake of depending on libcudf_test, which results in breakages
of cuSpatial sometimes when libcudf_test changes.
