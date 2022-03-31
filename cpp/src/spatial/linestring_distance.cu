

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
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <limits>
#include <memory>
#include <type_traits>

#ifndef DEBUG
#define DEBUG 1
#endif

namespace cuspatial {
namespace {

template <typename T>
double __device__ point_to_segment_distance(coord_2d<T> const& P,
                                            coord_2d<T> const& A,
                                            coord_2d<T> const& B)
{
  // Subject 1.02 of https://www.inf.pucrs.br/~pinho/CG/faq.html
  // Project the point to the segment, if it lands on the segment,
  // the distance is the length of proejction, otherwise it's the
  // length to one of the end points.

  double L_squared = (A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y);
  if (L_squared == 0) { return hypot(P.x - A.x, P.y - A.y); }
  double r = ((P.x - A.x) * (B.x - A.x) + (P.y - A.y) * (B.y - A.y)) / L_squared;
  if (r <= 0 or r >= 1) {
    return std::min(hypot(P.x - A.x, P.y - A.y), hypot(P.x - B.x, P.y - A.y));
  }
  double s = ((A.y - P.y) * (B.x - A.x) - (A.x - P.x) * (B.y - A.y)) / L_squared;
  return fabs(s) * sqrt(L_squared);
}

template <typename T>
double __device__ segment_distance_no_intersect(coord_2d<T> const& A,
                                                coord_2d<T> const& B,
                                                coord_2d<T> const& C,
                                                coord_2d<T> const& D)
{
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

template <typename CoordinateIterator, typename OffsetIterator>
struct linestirngs_pairs_min_distance_functor {
  using T = typename std::iterator_traits<CoordinateIterator>::value_type;

  cudf::size_type num_linestrings;
  cudf::size_type linestring1_num_points;
  cudf::size_type linestring2_num_points;
  OffsetIterator linestring1_offsets;
  CoordinateIterator linestring1_points_xs;
  CoordinateIterator linestring1_points_ys;
  OffsetIterator linestring2_offsets;
  CoordinateIterator linestring2_points_xs;
  CoordinateIterator linestring2_points_ys;

  linestirngs_pairs_min_distance_functor(cudf::size_type num_linestrings,
                                         cudf::size_type linestring1_num_points,
                                         cudf::size_type linestring2_num_points,
                                         OffsetIterator linestring1_offsets,
                                         CoordinateIterator linestring1_points_xs,
                                         CoordinateIterator linestring1_points_ys,
                                         OffsetIterator linestring2_offsets,
                                         CoordinateIterator linestring2_points_xs,
                                         CoordinateIterator linestring2_points_ys)
    : num_linestrings(num_linestrings),
      linestring1_num_points(linestring1_num_points),
      linestring2_num_points(linestring2_num_points),
      linestring1_offsets(linestring1_offsets),
      linestring1_points_xs(linestring1_points_xs),
      linestring1_points_ys(linestring1_points_ys),
      linestring2_offsets(linestring2_offsets),
      linestring2_points_xs(linestring2_points_xs),
      linestring2_points_ys(linestring2_points_ys)
  {
  }

  T __device__ operator()(cudf::size_type idx)
  {
    auto const l1pts_start = linestring1_offsets[idx];
    auto const l1pts_end =
      idx == (num_linestrings - 1) ? linestring1_num_points : linestring1_offsets[idx + 1];
    auto const l2pts_start = linestring2_offsets[idx];
    auto const l2pts_end =
      idx == (num_linestrings - 1) ? linestring2_num_points : linestring2_offsets[idx + 1];
#ifdef DEBUG
    printf("idx: %d\n", idx);
    printf("num_points_ls1: %d\n", linestring1_num_points);
    printf("num_points_ls2: %d\n", linestring2_num_points);
    printf("l1pts: %d, %d\n", l1pts_start, l1pts_end);
    printf("l2pts: %d, %d\n", l2pts_start, l2pts_end);
#endif
    double min_distance = std::numeric_limits<T>::max();
    for (cudf::size_type i = l1pts_start; i < l1pts_end - 1; i++) {
      for (cudf::size_type j = l2pts_start; j < l2pts_end - 1; j++) {
        coord_2d<T> A{linestring1_points_xs[i], linestring1_points_ys[i]};
        coord_2d<T> B{linestring1_points_xs[i + 1], linestring1_points_ys[i + 1]};
        coord_2d<T> C{linestring2_points_xs[j], linestring2_points_ys[j]};
        coord_2d<T> D{linestring2_points_xs[j + 1], linestring2_points_ys[j + 1]};
        min_distance = std::min(segment_distance(A, B, C, D), min_distance);
#ifdef DEBUG
        printf("%d %d, %f\n", i, j, min_distance);
#endif
      }
    }
    return min_distance;
  }
};

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

    auto const num_strings = static_cast<size_type>(linestring1_offsets.size());

    auto min_distances =
      make_numeric_column(data_type{type_to_id<T>()}, num_strings, mask_state::UNALLOCATED);

    auto functor = linestirngs_pairs_min_distance_functor(num_strings,
                                                          linestring1_points_x.size(),
                                                          linestring2_points_x.size(),
                                                          linestring1_offsets.begin(),
                                                          linestring1_points_x.begin<T>(),
                                                          linestring1_points_y.begin<T>(),
                                                          linestring2_offsets.begin(),
                                                          linestring2_points_x.begin<T>(),
                                                          linestring2_points_y.begin<T>());

    std::cout << "number of strings: " << num_strings << std::endl;
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(num_strings),
                      min_distances->mutable_view().begin<T>(),
                      functor);
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
 * @brief Check if every linestring in the input contains at least 1 segment.
 */
bool validate_linestring(cudf::device_span<cudf::size_type const> linestring_offsets,
                         cudf::column_view const& linestring_points_x,
                         rmm::cuda_stream_view stream)
{
  if (linestring_offsets.size() == 1) { return linestring_points_x.size() >= 2; }
  auto linestring_validate_iter = thrust::make_transform_iterator(
    thrust::make_zip_iterator(linestring_offsets.begin(), linestring_offsets.begin() + 1),
    [] __device__(thrust::tuple<cudf::size_type, cudf::size_type> indices) {
      return (indices.get<1>() - indices.get<0>()) >= 2;
    });
  return thrust::reduce(rmm::exec_policy(stream),
                        linestring_validate_iter,
                        linestring_validate_iter + linestring_offsets.size() - 1,
                        true,
                        thrust::logical_and<bool>());
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
