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

#include <cuspatial/types.hpp>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <optional>
#include <rmm/cuda_stream_view.hpp>

namespace cuspatial {
namespace test {

using namespace cudf;
using namespace cudf::test;

std::unique_ptr<cudf::column> coords_offset(size_type num_points, rmm::cuda_stream_view stream)
{
  auto zero = make_fixed_width_scalar<size_type>(0, stream);
  auto two  = make_fixed_width_scalar<size_type>(2, stream);

  return cudf::sequence(num_points, *zero, *two);
}

// helper function to make a point column
template <typename T>
std::pair<collection_type_id, std::unique_ptr<cudf::column>> make_point_column(
  std::initializer_list<T>&& point_coords, rmm::cuda_stream_view stream)
{
  auto num_points = point_coords.size() / 2;
  auto size       = point_offsets.size() - 1;

  auto zero           = make_fixed_width_scalar<size_type>(0, stream);
  auto two            = make_fixed_width_scalar<size_type>(2, stream);
  auto offsets_column = wrapper<cudf::size_type>(point_offsets).release();
  auto coords_offset  = cudf::sequence(num_points + 1, *zero, *two);
  auto coords_column  = wrapper<T>(point_coords).release();

  return {collection_type_id::SINGLE,
          cudf::make_lists_column(
            size,
            std::move(offsets_column),
            cudf::make_lists_column(
              num_points, std::move(coords_offset), std::move(coords_column), 0, {}),
            0,
            {})};
}

// helper function to make a multipoint column
template <typename T>
std::pair<collection_type_id, std::unique_ptr<cudf::column>> make_point_column(
  std::initializer_list<cudf::size_type>&& multipoint_offsets,
  std::initializer_list<cudf::size_type>&& point_offsets,
  std::initializer_list<T> point_coords,
  rmm::cuda_stream_view stream)
{
  auto geometry_size = multipoint_offsets.size() - 1;
  auto part_size     = point_offsets.size() - 1;
  auto num_points    = point_coords.size() / 2;

  auto zero            = make_fixed_width_scalar<size_type>(0, stream);
  auto two             = make_fixed_width_scalar<size_type>(2, stream);
  auto geometry_column = wrapper<cudf::size_type>(multipoint_offsets).release();
  auto part_column     = wrapper<cudf::size_type>(point_offsets).release();
  auto coords_offset   = cudf::sequence(num_points + 1, *zero, *two);
  auto coord_column    = wrapper<T>(point_coords).release();

  return {collection_type_id::MULTI,
          cudf::make_lists_column(
            geometry_size,
            std::move(geometry_column),
            cudf::make_lists_column(
              part_size,
              std::move(part_column),
              cudf::make_lists_column(
                num_points, std::move(coords_offset), std::move(coord_column), 0, {}),
              0,
              {}),
            0,
            {})};
}

}  // namespace test
}  // namespace cuspatial
