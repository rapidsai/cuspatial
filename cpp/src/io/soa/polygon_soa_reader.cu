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
#include <thrust/device_ptr.h>
#include <rmm/thrust_rmm_allocator.h>
#include <cudf/types.h>
#include <cudf/legacy/column.hpp>
#include <utilities/error_utils.hpp>
#include <cuspatial/soa_readers.hpp>
#include <utility/utility.hpp>

namespace cuspatial
{
/*
    * read polygon data from file in SoA format; data type of vertices is fixed to double (GDF_FLOAT64)
    * see soa_readers.hpp
*/
void read_polygon_soa(const char *filename,
                      gdf_column* ply_fpos, gdf_column* ply_rpos,
                      gdf_column* ply_x, gdf_column* ply_y)
{
    CUDF_EXPECTS(ply_fpos != nullptr && ply_rpos != nullptr &&
                 ply_x != nullptr && ply_y != nullptr,
                 "Null column pointer");

    memset(ply_fpos,0,sizeof(gdf_column));
    memset(ply_rpos,0,sizeof(gdf_column));
    memset(ply_x,0,sizeof(gdf_column));
    memset(ply_y,0,sizeof(gdf_column));

    struct polygons<double> pm;
    read_polygon_soa<double>(filename, &pm);
    if (pm.num_feature <=0) return;

    cudaStream_t stream{0};
    auto exec_policy = rmm::exec_policy(stream)->on(stream);

    int32_t* temp{nullptr};
    RMM_TRY( RMM_ALLOC(&temp, pm.num_feature * sizeof(int32_t), stream) );
    CUDA_TRY( cudaMemcpyAsync(temp, pm.feature_length,
                              pm.num_feature * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream) );
    //prefix-sum: len to pos
    thrust::inclusive_scan(exec_policy, temp, temp + pm.num_feature, temp);
    gdf_column_view_augmented(ply_fpos, temp, nullptr, pm.num_feature,
                              GDF_INT32, 0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE}, "f_pos");

    RMM_TRY( RMM_ALLOC(&temp, pm.num_ring * sizeof(int32_t), stream) );
    CUDA_TRY( cudaMemcpyAsync(temp, pm.ring_length,
                              pm.num_ring * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream) );
    thrust::inclusive_scan(exec_policy, temp, temp + pm.num_feature, temp);
    gdf_column_view_augmented(ply_rpos, temp, nullptr, pm.num_ring,
                              GDF_INT32, 0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE}, "r_pos");

    RMM_TRY( RMM_ALLOC(&temp, pm.num_vertex * sizeof(double), stream) );
    CUDA_TRY( cudaMemcpyAsync(temp, pm.x,
                              pm.num_vertex * sizeof(double),
                              cudaMemcpyHostToDevice, stream) );
    gdf_column_view_augmented(ply_x, temp, nullptr, pm.num_vertex,
                              GDF_FLOAT64, 0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE}, "x");

    RMM_TRY( RMM_ALLOC(&temp, pm.num_vertex * sizeof(double), stream) );
    CUDA_TRY( cudaMemcpyAsync(temp, pm.y,
                              pm.num_vertex * sizeof(double),
                              cudaMemcpyHostToDevice, stream) );
    gdf_column_view_augmented(ply_y, temp, nullptr, pm.num_vertex,
                              GDF_FLOAT64, 0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE}, "y");

    delete[] pm.feature_length;
    delete[] pm.ring_length;
    delete[] pm.x;
    delete[] pm.y;
    delete[] pm.group_length;
}

}// namespace cuspatial
