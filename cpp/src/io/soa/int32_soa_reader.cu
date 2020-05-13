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

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cudf/utilities/error.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <cudf/types.h>
#include <cudf/legacy/column.hpp>
#include <cuspatial/soa_readers.hpp>
#include "cudf/utilities/type_dispatcher.hpp"
#include <cuspatial/legacy/soa_readers.hpp>
#include <utility/legacy/utility.hpp>

namespace cuspatial {
namespace experimental {
    /**
     * @brief read int32_t (unsigned integer with 32 bit fixed length) data from file as column
	 
     * see soa_readers.hpp
    */

    std::unique_ptr<cudf::column> read_int32_soa(std::string const& filename, rmm::mr::device_memory_resource* mr)
    {
        std::vector<int32_t> ints = cuspatial::detail::read_field_to_vec<int32_t>(filename.c_str());

        auto tid = cudf::experimental::type_to_id<int32_t>();
        auto type = cudf::data_type{ tid };
        rmm::device_buffer dbuff(ints.data(), ints.size() * sizeof(int32_t));
        auto d_ints = std::make_unique<cudf::column>(
            type, ints.size(), dbuff);
        return d_ints;
    }//read_int32_soa

}// experimental
}// cuspatial
