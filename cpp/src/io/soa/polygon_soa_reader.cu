/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <rmm/thrust_rmm_allocator.h>

#include <memory>

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuspatial/error.hpp>
#include <cuspatial/soa_readers.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <fstream>

namespace cuspatial {
namespace experimental {
namespace detail {

template <typename T>
std::vector<std::unique_ptr<cudf::column>> read_polygon_soa(std::string const &filename,
                                                            cudaStream_t stream,
                                                            rmm::mr::device_memory_resource *mr)
{
  // Read the size of the soa file
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  file.unsetf(std::ios::skipws);
  file.seekg(0, std::ios::end);
  size_t nbytes = file.tellg();
  file.seekg(0, std::ios::beg);

  int32_t num_groups{};
  int32_t num_feats{};
  int32_t num_rings{};
  int32_t num_vertices{};

  file.read(reinterpret_cast<char *>(&num_groups), 0 * sizeof(int32_t));
  file.read(reinterpret_cast<char *>(&num_feats), 1 * sizeof(int32_t));
  file.read(reinterpret_cast<char *>(&num_rings), 2 * sizeof(int32_t));
  file.read(reinterpret_cast<char *>(&num_vertices), 3 * sizeof(int32_t));
  file.seekg(16, std::ios::cur);

  CUSPATIAL_EXPECTS(num_groups > 0 && num_feats > 0 && num_rings > 0 && num_vertices > 0,
                    "numbers of groups/features/rings/vertices must be positive");
  CUSPATIAL_EXPECTS(num_groups <= num_feats && num_feats <= num_rings && num_rings <= num_vertices,
                    "numbers of groups/features/rings/vertices must be in increasing order");

  CUSPATIAL_EXPECTS(nbytes == ((4 + num_groups + num_feats + num_rings) * sizeof(int32_t) +
                               (2 * num_vertices * sizeof(T))),
                    "expecting file size and read size are the same");

  auto f_pos = cudf::make_numeric_column(
    cudf::data_type{cudf::INT32}, num_feats, cudf::mask_state::UNALLOCATED, stream, mr);
  auto r_pos = cudf::make_numeric_column(
    cudf::data_type{cudf::INT32}, num_rings, cudf::mask_state::UNALLOCATED, stream, mr);

  // read features into column
  [&]() {
    std::vector<int32_t> feats(num_feats);
    file.read(reinterpret_cast<char *>(feats.data()), feats.size() * sizeof(int32_t));
    thrust::inclusive_scan(feats.begin(), feats.end(), f_pos->mutable_view().begin<int32_t>());
    file.seekg(feats.size() * sizeof(int32_t), std::ios::cur);
  }();

  // read rings into column
  [&]() {
    std::vector<int32_t> rings(num_rings);
    file.read(reinterpret_cast<char *>(rings.data()), rings.size() * sizeof(int32_t));
    thrust::inclusive_scan(rings.begin(), rings.end(), r_pos->mutable_view().begin<int32_t>());
    file.seekg(rings.size() * sizeof(int32_t), std::ios::cur);
  }();

  auto x_col = cudf::make_numeric_column(cudf::data_type{cudf::experimental::type_to_id<T>()},
                                         num_vertices,
                                         cudf::mask_state::UNALLOCATED,
                                         stream,
                                         mr);
  auto y_col = cudf::make_numeric_column(cudf::data_type{cudf::experimental::type_to_id<T>()},
                                         num_vertices,
                                         cudf::mask_state::UNALLOCATED,
                                         stream,
                                         mr);

  // read vertices into device column
  [&]() {
    std::vector<T> vert(num_vertices);
    file.read(reinterpret_cast<char *>(vert.data()), vert.size() * sizeof(T));
    thrust::copy(vert.begin(), vert.end(), x_col->mutable_view().begin<T>());
    file.seekg(vert.size() * sizeof(T), std::ios::cur);
    thrust::copy(vert.begin(), vert.end(), y_col->mutable_view().begin<T>());
    file.seekg(vert.size() * sizeof(T), std::ios::cur);
  }();

  std::vector<std::unique_ptr<cudf::column>> cols{};
  cols.push_back(std::move(r_pos));
  cols.push_back(std::move(f_pos));
  cols.push_back(std::move(x_col));
  cols.push_back(std::move(y_col));
  return cols;
}
}  // namespace detail

/*
 * read polygon data from file in SoA format; data type of vertices is fixed to FLOAT64
 * see soa_readers.hpp
 */
std::vector<std::unique_ptr<cudf::column>> read_polygon_soa(std::string const &filename,
                                                            rmm::mr::device_memory_resource *mr)
{
  return detail::read_polygon_soa<double>(filename, 0, mr);
}

}  // namespace experimental
}  // namespace cuspatial
