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
#include <thrust/device_vector.h>
#include <rmm/rmm.h>
#include <cudf/types.h>
#include <cudf/utilities/error.hpp>
#include <cuspatial/soa_readers.hpp>
#include <cuspatial/error.hpp>
#include <utility/utility.hpp>
#include "cudf/utilities/type_dispatcher.hpp"
#include "rmm/thrust_rmm_allocator.h"

namespace cuspatial
{
    /*
    * read polygon data from file in SoA format; data type of vertices is fixed to FLOAT64
    * see soa_readers.hpp
    */

    std::vector<std::unique_ptr<cudf::column>> read_polygon_soa(const char *filename)
    {
        struct polygons<double> pm;
        read_polygon_soa<double>(filename, &pm);

        cudaStream_t stream{0};
        auto exec_policy = rmm::exec_policy(stream);    

        auto tid_int32 = cudf::experimental::type_to_id<int32_t>();
        auto type_int32 = cudf::data_type{ tid_int32 };
        auto tid_float4 = cudf::experimental::type_to_id<float4>();
        auto type_float4 = cudf::data_type{ tid_float4 };

        rmm::device_buffer fpos_buffer(pm.feature_length, pm.num_feature * sizeof(int32_t));
        auto fpos = std::make_unique<cudf::column>(cudf::column(type_int32, pm.num_feature, fpos_buffer));
        auto fpos_begin = fpos->view().begin<int32_t>();
        auto mutable_fpos = fpos->mutable_view().begin<int32_t>();
        thrust::inclusive_scan(fpos_begin, fpos_begin + pm.num_feature, mutable_fpos);

        rmm::device_buffer rpos_buffer(pm.ring_length, pm.num_ring * sizeof(int32_t));
        auto rpos = std::make_unique<cudf::column>(cudf::column(type_int32, pm.num_ring, rpos_buffer));
        auto rpos_begin = rpos->view().begin<int32_t>();
        auto mutable_rpos = rpos->mutable_view().begin<int32_t>();
        thrust::inclusive_scan(rpos_begin, rpos_begin + pm.num_ring, mutable_rpos);

        rmm::device_buffer x_buffer(pm.x, pm.num_vertex * sizeof(float4));
        auto x = std::make_unique<cudf::column>(cudf::column(type_float4, pm.num_vertex, x_buffer));
        rmm::device_buffer y_buffer(pm.y, pm.num_vertex * sizeof(float4));
        auto y = std::make_unique<cudf::column>(cudf::column(type_float4, pm.num_vertex, y_buffer));

        delete[] pm.feature_length;
        delete[] pm.ring_length;
        delete[] pm.x;
        delete[] pm.y;
        delete[] pm.group_length;

        std::vector<std::unique_ptr<cudf::column>> result;
        result.push_back(std::move(rpos));
        result.push_back(std::move(fpos));
        result.push_back(std::move(x));
        result.push_back(std::move(y));
        return result;
    }

}// namespace cuspatial
