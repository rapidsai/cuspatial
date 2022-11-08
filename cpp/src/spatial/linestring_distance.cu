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
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/linestring_distance.cuh>
#include <cuspatial/experimental/ranges/multilinestring_range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <limits>
#include <memory>
#include <type_traits>

namespace cuspatial {
namespace detail {
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
    auto const num_string_pairs = static_cast<cudf::size_type>(linestring1_offsets.size()) - 1;

    auto distances = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<T>()},
                                               num_string_pairs,
                                               cudf::mask_state::UNALLOCATED,
                                               stream,
                                               mr);

    auto linestring1_coords_it =
      make_vec_2d_iterator(linestring1_points_x.begin<T>(), linestring1_points_y.begin<T>());
    auto linestring2_coords_it =
      make_vec_2d_iterator(linestring2_points_x.begin<T>(), linestring2_points_y.begin<T>());

    auto multilinestrings1 = make_multilinestring_range(num_string_pairs,
                                                        thrust::make_counting_iterator(0),
                                                        num_string_pairs,
                                                        linestring1_offsets.begin(),
                                                        linestring1_points_x.size(),
                                                        linestring1_coords_it);

    auto multilinestrings2 = make_multilinestring_range(num_string_pairs,
                                                        thrust::make_counting_iterator(0),
                                                        num_string_pairs,
                                                        linestring2_offsets.begin(),
                                                        linestring2_points_x.size(),
                                                        linestring2_coords_it);

    pairwise_linestring_distance(
      multilinestrings1, multilinestrings2, distances->mutable_view().begin<T>());

    return distances;
  }

  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Linestring distance API only supports floating point coordinates.");
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

  if (linestring1_offsets.size() - 1 == 0) { return cudf::empty_like(linestring1_points_x); }

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
