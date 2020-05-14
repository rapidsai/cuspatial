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

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuspatial/soa_readers.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>

// #include <thrust/iterator/transform_output_iterator.h>

#include <fstream>

namespace {

template <typename Element>
std::unique_ptr<cudf::column> read_values(std::string const& filename,
                                          cudaStream_t stream,
                                          rmm::mr::device_memory_resource* mr)
{
  // Read the size of the soa file
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  file.unsetf(std::ios::skipws);
  file.seekg(0, std::ios::end);
  size_t nbytes = file.tellg();
  file.seekg(0, std::ios::beg);

  // Read the file into memory
  std::vector<Element> vec(nbytes / sizeof(Element));
  file.read(reinterpret_cast<char*>(vec.data()), nbytes);

  // Copy the data to device buffer and return the column
  return std::make_unique<cudf::column>(cudf::data_type{cudf::experimental::type_to_id<Element>()},
                                        nbytes / sizeof(Element),
                                        rmm::device_buffer{vec.data(), nbytes, stream, mr});
}

};  // namespace

namespace cuspatial {
namespace experimental {
/**
 * @brief read int32_t (unsigned integer with 32 bit fixed length) data from file as column

 * see soa_readers.hpp
*/

std::unique_ptr<cudf::column> read_int32_soa(std::string const& filename,
                                             rmm::mr::device_memory_resource* mr)
{
  return read_values<int32_t>(filename, 0, mr);
}

}  // namespace experimental
}  // namespace cuspatial
