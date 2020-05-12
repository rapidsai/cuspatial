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
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <rmm/thrust_rmm_allocator.h>
#include <cudf/types.h>
#include <cudf/legacy/column.hpp>
#include <cudf/utilities/error.hpp>
#include <cuspatial/legacy/shapefile_readers.hpp>
#include <cuspatial/error.hpp>
#include <utility/legacy/utility.hpp>

namespace cuspatial
{
namespace detail
{ 
    /*
    * Read a polygon shapefile and fill in a polygons structure
    * ToDo: read associated relational data into a CUDF Table 
    *
    * filename: ESRI shapefile name (wtih .shp extension
    * pm: structure polygons (fixed to double type) to hold polygon data
    
    * Note: only the first layer is read - shapefiles have only one layer in GDALDataset model    
    */
    void polygon_from_shapefile(const char *filename, polygons<double>& pm);
} // namespace detail

/*
* read polygon data from file in ESRI Shapefile format; data type of vertices is fixed to double (GDF_FLOAT64)
* see shp_readers.hpp
*/

void read_polygon_shapefile(const char *filename,
                    gdf_column* ply_fpos, gdf_column* ply_rpos,
                    gdf_column* ply_x, gdf_column* ply_y)
{
    memset(ply_fpos,0,sizeof(gdf_column));
    memset(ply_rpos,0,sizeof(gdf_column));
    memset(ply_x,0,sizeof(gdf_column));
    memset(ply_y,0,sizeof(gdf_column));

    polygons<double> pm{};
    detail::polygon_from_shapefile(filename,pm);
    if (pm.num_feature <=0) return;

    cudaStream_t stream{0};
    auto exec_policy = rmm::exec_policy(stream);    

    int32_t* temp{nullptr};
    RMM_TRY( RMM_ALLOC(&temp, pm.num_feature * sizeof(int32_t), stream) );
    CUDA_TRY( cudaMemcpyAsync(temp, pm.feature_length,
                            pm.num_feature * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream) );
    //prefix-sum: len to pos
    thrust::inclusive_scan(exec_policy->on(stream), temp, temp + pm.num_feature, temp);
    gdf_column_view_augmented(ply_fpos, temp, nullptr, pm.num_feature,
                            GDF_INT32, 0,
                            gdf_dtype_extra_info{TIME_UNIT_NONE}, "f_pos");

    RMM_TRY( RMM_ALLOC(&temp, pm.num_ring * sizeof(int32_t), stream) );
    CUDA_TRY( cudaMemcpyAsync(temp, pm.ring_length,
                            pm.num_ring * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream) );
    thrust::inclusive_scan(exec_policy->on(stream), temp, temp + pm.num_ring, temp);
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
