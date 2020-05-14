/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>

#include <cuspatial/soa_readers.hpp>
#include <cuspatial/types.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/iterator/discard_iterator.h>

#include <fstream>

namespace {

template <typename Container, typename TransformFunctor>
std::unique_ptr<cudf::experimental::table> read_points(std::string const& filename,
                                                       cudaStream_t stream,
                                                       rmm::mr::device_memory_resource* mr,
                                                       TransformFunctor transform)
{
  using Element = typename Container::element_type;

  // Read the size of the soa file
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  file.unsetf(std::ios::skipws);
  file.seekg(0, std::ios::end);
  size_t nbytes = file.tellg();
  file.seekg(0, std::ios::beg);

  // Allocate cudf::columns for the point xs/ys
  auto x_col =
    cudf::make_fixed_width_column(cudf::data_type{cudf::experimental::type_to_id<Element>()},
                                  nbytes / sizeof(Container),
                                  cudf::mask_state::UNALLOCATED,
                                  stream,
                                  mr);
  auto y_col =
    cudf::make_fixed_width_column(cudf::data_type{cudf::experimental::type_to_id<Element>()},
                                  nbytes / sizeof(Container),
                                  cudf::mask_state::UNALLOCATED,
                                  stream,
                                  mr);

  std::vector<Container> structs(x_col->size());
  file.read(reinterpret_cast<char*>(structs.data()), nbytes);

  auto elements_iter = thrust::make_transform_iterator(structs.begin(), transform);

  // Copy the file stream into the point x/y device buffers
  thrust::copy(elements_iter,
               elements_iter + x_col->size(),
               thrust::make_zip_iterator(thrust::make_tuple(
                 x_col->mutable_view().begin<Element>(), y_col->mutable_view().begin<Element>())));

  std::vector<std::unique_ptr<cudf::column>> cols{};
  cols.reserve(2);
  cols.push_back(std::move(x_col));
  cols.push_back(std::move(y_col));
  return std::make_unique<cudf::experimental::table>(std::move(cols));
}

};  // namespace

namespace cuspatial {
namespace experimental {
namespace detail {

std::unique_ptr<cudf::experimental::table> read_lonlat_points_soa(
  std::string const& filename, cudaStream_t stream, rmm::mr::device_memory_resource* mr)
{
  return read_points<location_3d<double>>(
    filename, stream, mr, [](location_3d<double> const& coord) {
      return thrust::make_tuple(coord.longitude, coord.latitude);
    });
}

std::unique_ptr<cudf::experimental::table> read_xy_points_soa(std::string const& filename,
                                                              cudaStream_t stream,
                                                              rmm::mr::device_memory_resource* mr)
{
  return read_points<coord_2d<double>>(filename, stream, mr, [](coord_2d<double> const& coord) {
    return thrust::make_tuple(coord.x, coord.y);
  });
}

}  // namespace detail

/**
 * @brief read lon/lat from 3D point file into a table of two FLOAT64 columns
 *
 * @see cuspatial/soa_readers.hpp
 */
std::unique_ptr<cudf::experimental::table> read_lonlat_points_soa(
  std::string const& filename, rmm::mr::device_memory_resource* mr)
{
  return detail::read_lonlat_points_soa(filename, 0, rmm::mr::get_default_resource());
}

/**
 * @brief read x/y from 2D point file into a table of two FLOAT64 columns
 *
 * @see soa_readers.hpp
 */
std::unique_ptr<cudf::experimental::table> read_xy_points_soa(std::string const& filename,
                                                              rmm::mr::device_memory_resource* mr)
{
  return detail::read_xy_points_soa(filename, 0, rmm::mr::get_default_resource());
}

}  // namespace experimental
}  // namespace cuspatial
