

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

#ifndef DEBUG
// #define DEBUG 1
#endif

template <typename T>
void __device__ print(cuspatial::coord_2d<T> const& point)
{
  printf("POINT (%f, %f)\n", point.x, point.y);
}

template <typename T>
void __device__ print(cuspatial::coord_2d<T> const& A, cuspatial::coord_2d<T> const& B)
{
  printf("SEGMENT (%f, %f) -> (%f, %f)\n", A.x, A.y, B.x, B.y);
}

namespace cuspatial {
namespace {

template <typename T>
double __device__ point_to_segment_distance(coord_2d<T> const& C,
                                            coord_2d<T> const& A,
                                            coord_2d<T> const& B)
{
  // Subject 1.02 of https://www.inf.pucrs.br/~pinho/CG/faq.html
  // Project the point to the segment, if it lands on the segment,
  // the distance is the length of proejction, otherwise it's the
  // length to one of the end points.

  double L_squared = (A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y);
  if (L_squared == 0) { return hypot(C.x - A.x, C.y - A.y); }
  double r = ((C.x - A.x) * (B.x - A.x) + (C.y - A.y) * (B.y - A.y)) / L_squared;
#ifdef DEBUG
  printf("\t r=%f\n", r);
#endif
  if (r <= 0 or r >= 1) {
    return std::min(hypot(C.x - A.x, C.y - A.y), hypot(C.x - B.x, C.y - B.y));
  }
  double Px = A.x + r * (B.x - A.x);
  double Py = A.y + r * (B.y - A.y);
  return hypot(C.x - Px, C.y - Py);
}

template <typename T>
double __device__ segment_distance_no_intersect(coord_2d<T> const& A,
                                                coord_2d<T> const& B,
                                                coord_2d<T> const& C,
                                                coord_2d<T> const& D)
{
#ifdef DEBUG
  printf("From: \n");
  print(A);
  printf("To: \n");
  print(C, D);
  printf("Distance %f\n", point_to_segment_distance(A, C, D));

  printf("From: \n");
  print(B);
  printf("To: \n");
  print(C, D);
  printf("Distance %f\n", point_to_segment_distance(B, C, D));

  printf("From: \n");
  print(C);
  printf("To: \n");
  print(A, B);
  printf("Distance %f\n", point_to_segment_distance(C, A, B));

  printf("From: \n");
  print(D);
  printf("To: \n");
  print(A, B);
  printf("Distance %f\n", point_to_segment_distance(D, A, B));
#endif
  return std::min(std::min(point_to_segment_distance(A, C, D), point_to_segment_distance(B, C, D)),
                  std::min(point_to_segment_distance(C, A, B), point_to_segment_distance(D, A, B)));
}

/**
 * @brief Computes shortest distance between two segments.
 *
 * If two segment intersects, distance is 0.
 */
template <typename T>
double __device__ segment_distance(coord_2d<T> const& A,
                                   coord_2d<T> const& B,
                                   coord_2d<T> const& C,
                                   coord_2d<T> const& D)
{
  // Subject 1.03 of https://www.inf.pucrs.br/~pinho/CG/faq.html
  // Construct a parametrized ray of AB and CD, solve for the parameters.
  // If both parameters are within [0, 1], the intersection exists.

  double r_denom = (B.x - A.x) * (D.y - C.y) - (B.y - A.y) * (D.x - C.x);
  double r_numer = (A.y - C.y) * (D.x - C.x) - (A.x - C.x) * (D.y - C.y);
  if (r_denom == 0) {
    if (r_numer == 0) { return 0.0; }  // Segments coincides
    // Segments parallel
    return segment_distance_no_intersect(A, B, C, D);
  }
  double r = r_numer / r_denom;
  double s = ((A.y - C.y) * (B.x - A.x) - (A.x - C.x) * (B.y - A.x)) /
             ((B.x - A.x) * (D.y - C.y) - (B.y - A.y) * (D.x - C.x));
  if (r >= 0 and r <= 1 and s >= 0 and s <= 1) { return 0.0; }
  return segment_distance_no_intersect(A, B, C, D);
}

template <typename CoordinateIterator, typename OffsetIterator, typename OutputIterator>
void __global__ kernel(OffsetIterator linestring1_offsets_begin,
                       OffsetIterator linestring1_offsets_end,
                       CoordinateIterator linestring1_points_xs_begin,
                       CoordinateIterator linestring1_points_xs_end,
                       CoordinateIterator linestring1_points_ys_begin,
                       OffsetIterator linestring2_offsets_begin,
                       CoordinateIterator linestring2_points_xs_begin,
                       CoordinateIterator linestring2_points_xs_end,
                       CoordinateIterator linestring2_points_ys_begin,
                       OutputIterator min_distances)
{
  using T = typename std::iterator_traits<CoordinateIterator>::value_type;

  auto const p1Idx = threadIdx.x + blockIdx.x * blockDim.x;
  cudf::size_type const num_linestrings =
    thrust::distance(linestring1_offsets_begin, linestring1_offsets_end);
  cudf::size_type const linestring1_num_points =
    thrust::distance(linestring1_points_xs_begin, linestring1_points_xs_end);
  cudf::size_type const linestring2_num_points =
    thrust::distance(linestring2_points_xs_begin, linestring2_points_xs_end);

#ifdef DEBUG
  printf("p1Idx: %d\n", p1Idx);
  printf("linestring1_num_points: %d\n", linestring1_num_points);
#endif

  if (p1Idx >= linestring1_num_points) { return; }

  cudf::size_type const linestring_idx =
    thrust::distance(
      linestring1_offsets_begin,
      thrust::upper_bound(thrust::seq, linestring1_offsets_begin, linestring1_offsets_end, p1Idx)) -
    1;

  cudf::size_type ls1End =
    (linestring_idx == (num_linestrings - 1) ? (linestring1_num_points)
                                             : *(linestring1_offsets_begin + linestring_idx + 1)) -
    1;

#ifdef DEBUG
  printf("p1Idx: %d\n", p1Idx);
  printf("ls1End: %d\n", ls1End);
  printf("linestring_idx: %d\n", linestring_idx);
  printf("num_linestrings: %d\n", num_linestrings);
#endif
  if (p1Idx == ls1End) {
    // Current point is the end point of the line string.
    return;
  }

  cudf::size_type ls2Start = *(linestring2_offsets_begin + linestring_idx);
  cudf::size_type ls2End =
    (linestring_idx == (num_linestrings - 1) ? linestring2_num_points
                                             : *(linestring2_offsets_begin + linestring_idx + 1)) -
    1;

  coord_2d<T> A{linestring1_points_xs_begin[p1Idx], linestring1_points_ys_begin[p1Idx]};
  coord_2d<T> B{linestring1_points_xs_begin[p1Idx + 1], linestring1_points_ys_begin[p1Idx + 1]};

#ifdef DEBUG
  printf("linestring_idx: %d\n", linestring_idx);
  printf("num_linestrings: %d\n", num_linestrings);
  printf("linestring1_num_points: %d\n", linestring1_num_points);
  printf("linestring2_num_points: %d\n", linestring2_num_points);
  printf("p1Idx: %d\n", p1Idx);
  printf("ls2Start: %d\n", ls2Start);
  printf("ls2End: %d\n", ls2End);
  print(A, B);
#endif
  double min_distance = std::numeric_limits<double>::max();
  for (cudf::size_type p2Idx = ls2Start; p2Idx < ls2End; p2Idx++) {
    coord_2d<T> C{linestring2_points_xs_begin[p2Idx], linestring2_points_ys_begin[p2Idx]};
    coord_2d<T> D{linestring2_points_xs_begin[p2Idx + 1], linestring2_points_ys_begin[p2Idx + 1]};
#ifdef DEBUG
    print(C, D);
#endif
    min_distance = std::min(min_distance, segment_distance(A, B, C, D));
  }
  atomicMin(min_distances + linestring_idx, static_cast<T>(min_distance));
}

}  // anonymous namespace

namespace detail {
struct pariwise_linestring_distance_functor {
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

    auto min_distances =
      make_numeric_column(data_type{type_to_id<T>()}, num_string_pairs, mask_state::UNALLOCATED);

    // auto functor = linestirngs_pairs_min_distance_functor(num_string_pairs,
    //                                                       linestring1_points_x.size(),
    //                                                       linestring2_points_x.size(),
    //                                                       linestring1_offsets.begin(),
    //                                                       linestring1_points_x.begin<T>(),
    //                                                       linestring1_points_y.begin<T>(),
    //                                                       linestring2_offsets.begin(),
    //                                                       linestring2_points_x.begin<T>(),
    //                                                       linestring2_points_y.begin<T>());

    // thrust::transform(rmm::exec_policy(stream),
    //                   thrust::make_counting_iterator(0),
    //                   thrust::make_counting_iterator(num_string_pairs),
    //                   min_distances->mutable_view().begin<T>(),
    //                   functor);

    thrust::fill(rmm::exec_policy(stream),
                 min_distances->mutable_view().begin<T>(),
                 min_distances->mutable_view().end<T>(),
                 std::numeric_limits<T>::max());

    std::size_t const threads_per_block = 64;
    std::size_t const num_blocks =
      (linestring1_points_x.size() + threads_per_block - 1) / threads_per_block;

#ifdef DEBUG
    std::cout << "number of strings: " << num_string_pairs << std::endl;
    std::cout << "num blocks" << num_blocks << std::endl;
#endif
    kernel<<<num_blocks, threads_per_block, 0, stream.value()>>>(
      linestring1_offsets.begin(),
      linestring1_offsets.end(),
      linestring1_points_x.begin<T>(),
      linestring1_points_x.end<T>(),
      linestring1_points_y.begin<T>(),
      linestring2_offsets.begin(),
      linestring2_points_x.begin<T>(),
      linestring2_points_x.end<T>(),
      linestring2_points_y.begin<T>(),
      min_distances->mutable_view().begin<T>());

    CUSPATIAL_CUDA_TRY(cudaGetLastError());

    return min_distances;
  }

  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Linestring distances only supports floating point coordinates.");
  }
};

/**
 * @brief Check if every linestring in the input contains at least 2 end points.
 */
bool validate_linestring(cudf::device_span<cudf::size_type const> linestring_offsets,
                         cudf::column_view const& linestring_points_x,
                         rmm::cuda_stream_view stream)
{
  if (linestring_offsets.size() == 1) { return linestring_points_x.size() >= 2; }
  return thrust::reduce(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(cudf::size_type{0}),
    thrust::make_counting_iterator(static_cast<cudf::size_type>(linestring_offsets.size())),
    true,
    [offsets     = linestring_offsets.begin(),
     num_offsets = static_cast<cudf::size_type>(linestring_offsets.size()),
     num_points  = static_cast<cudf::size_type>(
       linestring_points_x.size())] __device__(bool prev, cudf::size_type i) {
      cudf::size_type begin = offsets[i];
      cudf::size_type end   = i == num_offsets ? num_points : offsets[i + 1];
      return prev && (end - begin);
    });
}

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

  CUSPATIAL_EXPECTS(validate_linestring(linestring1_offsets, linestring1_points_x, stream),
                    "Each item of linestring1 should contain at least 2 end points.");
  CUSPATIAL_EXPECTS(validate_linestring(linestring2_offsets, linestring2_points_x, stream),
                    "Each item of linestring2 should contain at least 2 end points.");

  return cudf::type_dispatcher(linestring1_points_x.type(),
                               pariwise_linestring_distance_functor{},
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
