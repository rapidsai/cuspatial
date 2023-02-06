/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "../utility/iterator.hpp"
#include "../utility/multi_geometry_dispatch.hpp"

#include <cuspatial_test/test_util.cuh>

#include <cuspatial/column/geometry_column_view.hpp>
#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/linestring_intersection.cuh>
#include <cuspatial/experimental/ranges/multilinestring_range.cuh>
#include <cuspatial/linestring_intersection.hpp>
#include <cuspatial/types.hpp>
#include <cuspatial/vec_2d.hpp>

#include <cuspatial_test/test_util.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <memory>
#include <type_traits>

namespace cuspatial {
namespace detail {

/**
 * @brief Move the ownership to a buffer from a `device_uvector` to a `cudf::column`
 */
template <typename T>
std::unique_ptr<cudf::column> move_uvector(std::unique_ptr<rmm::device_uvector<T>> ptr)
{
  return std::unique_ptr<cudf::column>(new cudf::column(std::move(*ptr)));
}

template <collection_type_id lhs_type, collection_type_id rhs_type>
struct pairwise_linestring_intersection_launch {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, linestring_intersection_column_result>
  operator()(geometry_column_view const& multilinestrings1,
             geometry_column_view const& multilinestrings2,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr)
  {
    using index_t = cudf::size_type;

    auto multilinestrings_range1 =
      make_multilinestring_range<lhs_type, T, index_t>(multilinestrings1);

    auto multilinestrings_range2 =
      make_multilinestring_range<rhs_type, T, index_t>(multilinestrings2);

    auto intersection_results = pairwise_linestring_intersection<T, index_t>(
      multilinestrings_range1, multilinestrings_range2, mr, stream);

    auto num_point_coords = 2 * intersection_results.points_coords->size();
    auto points_xy        = std::make_unique<cudf::column>(cudf::data_type(cudf::type_to_id<T>()),
                                                    num_point_coords,
                                                    intersection_results.points_coords->release());

    auto segment_offsets =
      cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<index_t>()},
                                intersection_results.segments_coords->size() + 1,
                                cudf::mask_state::UNALLOCATED,
                                stream,
                                mr);

    thrust::sequence(rmm::exec_policy(stream),
                     segment_offsets->mutable_view().template begin<index_t>(),
                     segment_offsets->mutable_view().template end<index_t>(),
                     0,
                     2);

    auto num_segment_coords = 4 * intersection_results.segments_coords->size();
    auto segments_xy =
      std::make_unique<cudf::column>(cudf::data_type(cudf::type_to_id<T>()),
                                     num_segment_coords,
                                     intersection_results.segments_coords->release());

    auto num_segments = segment_offsets->size()-1;
    auto segments = cudf::make_lists_column(
        num_segments,
        std::move(segment_offsets),
        std::move(segments_xy),
        0,
        {},
        stream,
        mr
    );

    return linestring_intersection_column_result{
      move_uvector(std::move(intersection_results.geometry_collection_offset)),
      move_uvector(std::move(intersection_results.types_buffer)),
      move_uvector(std::move(intersection_results.offset_buffer)),
      std::move(points_xy),
      std::move(segments),
      move_uvector(std::move(intersection_results.lhs_linestring_id)),
      move_uvector(std::move(intersection_results.lhs_segment_id)),
      move_uvector(std::move(intersection_results.rhs_linestring_id)),
      move_uvector(std::move(intersection_results.rhs_segment_id))};
  }

  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, linestring_intersection_column_result>
  operator()(Args&&...)
  {
    CUSPATIAL_FAIL("Linestring intersections only supports floating point coordinates.");
  }
};

template <collection_type_id lhs_type, collection_type_id rhs_type>
struct pairwise_linestring_intersection {
  linestring_intersection_column_result operator()(geometry_column_view const& linestrings1,
                                                   geometry_column_view const& linestrings2,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource* mr)
  {
    CUSPATIAL_EXPECTS(linestrings1.coordinate_type() == linestrings2.coordinate_type(),
                      "Input linestring coordinates must be the same type.");
    CUSPATIAL_EXPECTS(linestrings1.size() == linestrings2.size(),
                      "Input geometry array size mismatches.");

    return cudf::type_dispatcher(linestrings1.coordinate_type(),
                                 pairwise_linestring_intersection_launch<lhs_type, rhs_type>{},
                                 linestrings1,
                                 linestrings2,
                                 stream,
                                 mr);
  }
};

}  // namespace detail

linestring_intersection_column_result pairwise_linestring_intersection(
  geometry_column_view const& lhs,
  geometry_column_view const& rhs,
  rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(lhs.geometry_type() == geometry_type_id::LINESTRING &&
                      rhs.geometry_type() == geometry_type_id::LINESTRING,
                    "Input must be linestring columns.");

  return multi_geometry_double_dispatch<detail::pairwise_linestring_intersection>(
    lhs.collection_type(), rhs.collection_type(), lhs, rhs, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
