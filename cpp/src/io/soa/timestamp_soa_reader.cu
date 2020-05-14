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

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuspatial/soa_readers.hpp>

#include <utility/legacy/utility.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace cuspatial {
namespace experimental {

// TODO: define timestamp to cuspatial timestamp kernel here
// Reason: No more its_timestamp - its_timestamp is always converted to libcudf++
// timestamp.

/**
 * @brief read timestamp (ts: Time type) data from file as column
 * see soa_readers.hpp
 */
std::unique_ptr<cudf::column> read_timestamp_soa(std::string const& filename,
                                                 rmm::mr::device_memory_resource* mr)
{
  std::vector<its_timestamp> timestamp =
    cuspatial::detail::read_field_to_vec<its_timestamp>(filename.c_str());

  auto tid  = cudf::experimental::type_to_id<int64_t>();
  auto type = cudf::data_type{tid};
  rmm::device_buffer dbuff(timestamp.data(), timestamp.size() * sizeof(its_timestamp));
  auto ts = std::make_unique<cudf::column>(type, timestamp.size(), dbuff);
  return ts;
}

}  // namespace experimental
}  // namespace cuspatial
