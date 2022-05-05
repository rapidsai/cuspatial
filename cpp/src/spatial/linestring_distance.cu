/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuspatial/error.hpp>
#include <cuspatial/types.hpp>
#include <cuspatial/utility/vec_2d.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <limits>
#include <memory>
#include <type_traits>

namespace cuspatial {
namespace {

/** @brief Get the index that is one-past the end point of linestring at @p linestring_idx
 *
 * @note The last endpoint of the linestring is not included in the offset array, thus
 * @p num_points is returned.
 */
template <typename SizeType, typename OffsetIterator>
inline SizeType __device__
endpoint_index_of_linestring(SizeType const& linestring_idx,
                             OffsetIterator const& linestring_offsets_begin,
                             SizeType const& num_linestrings,
                             SizeType const& num_points)
{
  return (linestring_idx == (num_linestrings - 1)
            ? (num_points)
            : *(linestring_offsets_begin + linestring_idx + 1)) -
         1;
}

/**
 * @brief Computes shortest distance between @p C and segment @p A @p B
 */
template <typename T>
T __device__ point_to_segment_distance_squared(vec_2d<T> const& c,
                                               vec_2d<T> const& a,
                                               vec_2d<T> const& b)
{
  auto ab   = b - a;
  auto ac        = c - a;
  auto bc        = c - b;
  auto l_squared = dot(ab, ab);
  if (l_squared == 0) { return dot(ac, ac); }
  auto r = dot(ac, ab);
  if (r <= 0 or r >= l_squared) { return std::min(dot(ac, ac), dot(bc, bc)); }
  auto p  = a + (r / l_squared) * ab;
  auto pc = c - p;
  return dot(pc, pc);
}

/**
 * @brief Computes shortest distance between two segments that doesn't intersect.
 */
template <typename T>
T __device__ segment_distance_no_intersect_or_colinear(vec_2d<T> const& a,
                                                       vec_2d<T> const& b,
                                                       vec_2d<T> const& c,
                                                       vec_2d<T> const& d)
{
  auto dist_sqr = std::min(std::min(point_to_segment_distance_squared(a, c, d),
                                    point_to_segment_distance_squared(b, c, d)),
                           std::min(point_to_segment_distance_squared(c, a, b),
                                    point_to_segment_distance_squared(d, a, b)));
  return dist_sqr;
}

/**
 * @brief Computes shortest distance between two segments.
 *
 * If two segments intersect, the distance is 0. Otherwise compute the shortest point
 * to segment distance.
 */
template <typename T>
T __device__ squared_segment_distance(vec_2d<T> const& a,
                                      vec_2d<T> const& b,
                                      vec_2d<T> const& c,
                                      vec_2d<T> const& d)
{
  auto ab    = b - a;
  auto cd    = d - c;
  auto denom = det(ab, cd);

  if (denom == 0) {
    // Segments parallel or collinear
    return segment_distance_no_intersect_or_colinear(a, b, c, d);
  }

  auto ac    = c - a;
  auto r_numer = det(ac, cd);
  auto r       = r_numer / denom;
  auto s       = det(ac, ab) / denom;
  if (r >= 0 and r <= 1 and s >= 0 and s <= 1) { return 0.0; }
  return segment_distance_no_intersect_or_colinear(a, b, c, d);
}

/**
 * @brief The kernel to compute point to linestring distance
 *
 * Each thread of the kernel computes the distance between a segment in a linestring in pair 1
 * to a linestring in pair 2. For a segment in pair 1, the linestring index is looked up from
 * the offset array and mapped to the linestring in the pair 2. The segment is then computed
 * with all segments in the corresponding linestringin pair 2. This forms a local minima of the
 * shortest distance, which is then combined with other segment results via an atomic operation
 * to form the globally minimum distance between the linestrings.
 *
 * @tparam CoordinateIterator Iterator to coordinates. Must meet requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OffsetIterator Iterator to linestring offsets. Must meet requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OutputIterator Iterator to output distances. Must meet requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 *
 * @param[in] linestring1_offsets_begin Iterator to the begin of the range of linestring offsets
 * in pair 1.
 * @param[in] linestring1_offsets_end Iterator to the end of the range of linestring offsets
 * in pair 1.
 * @param[in] linestring1_points_xs_begin Iterator to the begin of the range of x coordinates of
 * points in pair 1.
 * @param[in] linestring1_points_xs_end Iterator to the end of the range of x coordiantes of points
 * in pair 1.
 * @param[in] linestring1_points_ys_begin Iterator to the begin of the range of y coordinates of
 * points in pair 1.
 * @param[in] linestring2_offsets_begin Iterator to the begin of the range of linestring offsets
 * in pair 2.
 * @param[in] linestring2_points_xs_begin Iterator to the begin of the range of x coordinates of
 * points in pair 2.
 * @param[in] linestring2_points_xs_end Iterator to the end of the range of x coordiantes of points
 * in pair 2.
 * @param[in] linestring2_points_ys_begin Iterator to the begin of the range of y coordinates of
 * points in pair 2.
 * @param[out] distances Iterator to the output range of shortest distances between pairs.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename CoordinateIterator, typename OffsetIterator, typename OutputIterator>
void __global__ pairwise_linestring_distance_kernel(OffsetIterator linestring1_offsets_begin,
                                                    OffsetIterator linestring1_offsets_end,
                                                    CoordinateIterator linestring1_points_xs_begin,
                                                    CoordinateIterator linestring1_points_xs_end,
                                                    CoordinateIterator linestring1_points_ys_begin,
                                                    OffsetIterator linestring2_offsets_begin,
                                                    CoordinateIterator linestring2_points_xs_begin,
                                                    CoordinateIterator linestring2_points_xs_end,
                                                    CoordinateIterator linestring2_points_ys_begin,
                                                    OutputIterator distances)
{
  using T = typename std::iterator_traits<CoordinateIterator>::value_type;

  auto const p1_idx = threadIdx.x + blockIdx.x * blockDim.x;
  cudf::size_type const num_linestrings =
    thrust::distance(linestring1_offsets_begin, linestring1_offsets_end);
  cudf::size_type const linestring1_num_points =
    thrust::distance(linestring1_points_xs_begin, linestring1_points_xs_end);
  cudf::size_type const linestring2_num_points =
    thrust::distance(linestring2_points_xs_begin, linestring2_points_xs_end);

  if (p1_idx >= linestring1_num_points) { return; }

  cudf::size_type const linestring_idx =
    thrust::distance(linestring1_offsets_begin,
                     thrust::upper_bound(
                       thrust::seq, linestring1_offsets_begin, linestring1_offsets_end, p1_idx)) -
    1;

  cudf::size_type ls1_end = endpoint_index_of_linestring(
    linestring_idx, linestring1_offsets_begin, num_linestrings, linestring1_num_points);

  if (p1_idx == ls1_end) {
    // Current point is the end point of the line string.
    return;
  }

  cudf::size_type ls2_start = *(linestring2_offsets_begin + linestring_idx);
  cudf::size_type ls2_end   = endpoint_index_of_linestring(
    linestring_idx, linestring2_offsets_begin, num_linestrings, linestring2_num_points);

  vec_2d<T> A{linestring1_points_xs_begin[p1_idx], linestring1_points_ys_begin[p1_idx]};
  vec_2d<T> B{linestring1_points_xs_begin[p1_idx + 1], linestring1_points_ys_begin[p1_idx + 1]};

  T min_squared_distance = std::numeric_limits<T>::max();
  for (cudf::size_type p2_idx = ls2_start; p2_idx < ls2_end; p2_idx++) {
    vec_2d<T> C{linestring2_points_xs_begin[p2_idx], linestring2_points_ys_begin[p2_idx]};
    vec_2d<T> D{linestring2_points_xs_begin[p2_idx + 1], linestring2_points_ys_begin[p2_idx + 1]};
    min_squared_distance = std::min(min_squared_distance, squared_segment_distance(A, B, C, D));
  }
  atomicMin(distances + linestring_idx, static_cast<T>(std::sqrt(min_squared_distance)));
}

}  // anonymous namespace

namespace detail {

/**
 * @brief Functor that launches the kernel to compute pairwise linestring distances.
 */
struct pairwise_linestring_distance_functor {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    cudf::device_span<cudf::size_type const> linestring1_offsets,
    cudf::column_view const& linestring1_points_x,
    cudf::column_view const& linestring1_points_y,
    cudf::device_span<cudf::size_type const> linestring2_offsets,
    cudf::column_view const& linestring2_points_x,
    cudf::column_view const& linestring2_points_y,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    using namespace cudf;

    auto const num_string_pairs = static_cast<size_type>(linestring1_offsets.size());

    auto distances =
      make_numeric_column(data_type{type_to_id<T>()}, num_string_pairs, mask_state::UNALLOCATED);

    thrust::fill(rmm::exec_policy(stream),
                 distances->mutable_view().begin<T>(),
                 distances->mutable_view().end<T>(),
                 std::numeric_limits<T>::max());

    std::size_t constexpr threads_per_block = 64;
    std::size_t const num_blocks =
      (linestring1_points_x.size() + threads_per_block - 1) / threads_per_block;

    pairwise_linestring_distance_kernel<<<num_blocks, threads_per_block, 0, stream.value()>>>(
      linestring1_offsets.begin(),
      linestring1_offsets.end(),
      linestring1_points_x.begin<T>(),
      linestring1_points_x.end<T>(),
      linestring1_points_y.begin<T>(),
      linestring2_offsets.begin(),
      linestring2_points_x.begin<T>(),
      linestring2_points_x.end<T>(),
      linestring2_points_y.begin<T>(),
      distances->mutable_view().begin<T>());

    CUSPATIAL_CUDA_TRY(cudaGetLastError());

    return distances;
  }

  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Linestring distances only supports floating point coordinates.");
  }
};

std::unique_ptr<cudf::column> pairwise_linestring_distance(
  cudf::device_span<cudf::size_type const> linestring1_offsets,
  cudf::column_view const& linestring1_points_x,
  cudf::column_view const& linestring1_points_y,
  cudf::device_span<cudf::size_type const> linestring2_offsets,
  cudf::column_view const& linestring2_points_x,
  cudf::column_view const& linestring2_points_y,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(linestring1_offsets.size() == linestring2_offsets.size(),
                    "Mismatch number of linestrings in the linestring pair array.");

  CUSPATIAL_EXPECTS(linestring1_points_x.size() == linestring1_points_y.size() and
                      linestring2_points_x.size() == linestring2_points_y.size(),
                    "The lengths of linestring coordinates arrays mismatch.");

  CUSPATIAL_EXPECTS(linestring1_points_x.type() == linestring1_points_y.type() and
                      linestring2_points_x.type() == linestring2_points_y.type() and
                      linestring1_points_x.type() == linestring2_points_x.type(),
                    "The types of linestring coordinates arrays mismatch.");

  if (linestring1_offsets.size() == 0) { return cudf::empty_like(linestring1_points_x); }

  return cudf::type_dispatcher(linestring1_points_x.type(),
                               pairwise_linestring_distance_functor{},
                               linestring1_offsets,
                               linestring1_points_x,
                               linestring1_points_y,
                               linestring2_offsets,
                               linestring2_points_x,
                               linestring2_points_y,
                               stream,
                               mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> pairwise_linestring_distance(
  cudf::device_span<cudf::size_type const> linestring1_offsets,
  cudf::column_view const& linestring1_points_x,
  cudf::column_view const& linestring1_points_y,
  cudf::device_span<cudf::size_type const> linestring2_offsets,
  cudf::column_view const& linestring2_points_x,
  cudf::column_view const& linestring2_points_y,
  rmm::mr::device_memory_resource* mr)
{
  return detail::pairwise_linestring_distance(linestring1_offsets,
                                              linestring1_points_x,
                                              linestring1_points_y,
                                              linestring2_offsets,
                                              linestring2_points_x,
                                              linestring2_points_y,
                                              rmm::cuda_stream_default,
                                              mr);
}

}  // namespace cuspatial
