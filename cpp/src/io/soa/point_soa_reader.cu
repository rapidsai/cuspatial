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
#include <cudf/types.h>
#include <cudf/legacy/column.hpp>
#include <utilities/error_utils.hpp>
#include <rmm/rmm.h>
#include <cuspatial/soa_readers.hpp>
#include <utility/utility.hpp>

namespace cuspatial
{
    /**
    * @brief read lon/lat from file into two columns; data type is fixed to double (GDF_FLOAT64)
    *
    * see soa_readers.hpp
    */
    std::pair<gdf_column, gdf_column>  read_lonlat_points_soa(const char *filename)                                  
    {

        cudaStream_t stream{0};

        double * p_lon=nullptr, *p_lat=nullptr;
        int num_p=read_point_lonlat<double>(filename,p_lon,p_lat);
       
        gdf_column pnt_lon,pnt_lat;
        memset(&pnt_lon,0,sizeof(gdf_column));
        memset(&pnt_lat,0,sizeof(gdf_column));
	    
        double* temp_lon{nullptr};
        RMM_TRY( RMM_ALLOC(&temp_lon, num_p * sizeof(double), 0) );
        CUDA_TRY( cudaMemcpyAsync(temp_lon, p_lon,
                                  num_p * sizeof(double) , 
                                  cudaMemcpyHostToDevice,stream) );		
        gdf_column_view_augmented(&pnt_lon, temp_lon, nullptr, num_p,
                              GDF_FLOAT64, 0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE}, "lon");          
        delete[] p_lon;

        double* temp_lat{nullptr};
        RMM_TRY( RMM_ALLOC(&temp_lat, num_p * sizeof(double), 0) );
        CUDA_TRY( cudaMemcpyAsync(temp_lat, p_lat,
                                  num_p * sizeof(double) , 
                                  cudaMemcpyHostToDevice,stream) );		
        gdf_column_view_augmented(&pnt_lat, temp_lat, nullptr, num_p,
                              GDF_FLOAT64, 0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE}, "lat");          
        delete[] p_lat;
	    
        return std::make_pair(pnt_lon,pnt_lat);
    }
	
    /**
    * @brief read x/y from file into two columns; data type is fixed to double (GDF_FLOAT64)
    *
    * see soa_readers.hpp
    */
    std::pair<gdf_column, gdf_column>  read_xy_points_soa(const char *filename)                              
    {
        double * p_x=nullptr, *p_y=nullptr;
        int num_p=read_point_xy<double>(filename,p_x,p_y);
       
        gdf_column pnt_x,pnt_y;
        memset(&pnt_x,0,sizeof(gdf_column));
        memset(&pnt_y,0,sizeof(gdf_column));
	    
        double* temp_x{nullptr};
        RMM_TRY( RMM_ALLOC(&temp_x, num_p * sizeof(double), 0) );
        cudaStream_t stream{0};
        CUDA_TRY( cudaMemcpyAsync(temp_x, p_x,
                                  num_p * sizeof(double) , 
                                  cudaMemcpyHostToDevice,stream) );		
        gdf_column_view_augmented(&pnt_x, temp_x, nullptr, num_p,
                              GDF_FLOAT64, 0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE}, "x");          
        delete[] p_x;

        double* temp_y{nullptr};
        RMM_TRY( RMM_ALLOC(&temp_y, num_p * sizeof(double), 0) );
        CUDA_TRY( cudaMemcpyAsync(temp_y, p_y,
                                  num_p * sizeof(double) , 
                                  cudaMemcpyHostToDevice,stream) );		
        gdf_column_view_augmented(&pnt_y, temp_y, nullptr, num_p,
                              GDF_FLOAT64, 0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE}, "y");          
        delete[] p_y;
	    
        return std::make_pair(pnt_x,pnt_y);
    }	
}
