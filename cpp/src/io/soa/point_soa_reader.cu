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
#include <cudf/types.h>
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

	    double * p_lon=nullptr, *p_lat=nullptr;
	    int num_p=read_point_lonlat<double>(filename,p_lon,p_lat);
	    gdf_column pnt_lon,pnt_lat;
	    
 	    pnt_lon.dtype= GDF_FLOAT64;
 	    pnt_lon.col_name=(char *)malloc(strlen("lon")+ 1);
	    strcpy(pnt_lon.col_name,"lon");
	    RMM_TRY( RMM_ALLOC(&pnt_lon.data, num_p * sizeof(double), 0) );
	    cudaMemcpy(pnt_lon.data, p_lon,num_p * sizeof(double) , cudaMemcpyHostToDevice);		
	    pnt_lon.size=num_p;
	    pnt_lon.valid=nullptr;
	    pnt_lon.null_count=0;		
	    delete[] p_lon;

 	    pnt_lat.dtype= GDF_FLOAT64;
 	    pnt_lat.col_name=(char *)malloc(strlen("lat")+ 1);
	    strcpy(pnt_lat.col_name,"lat");
	    pnt_lat.data=nullptr;
	    RMM_TRY( RMM_ALLOC(&pnt_lat.data, num_p * sizeof(double), 0) );
	    cudaMemcpy(pnt_lat.data, p_lat,num_p * sizeof(double) , cudaMemcpyHostToDevice);		
	    pnt_lat.size=num_p;
	    pnt_lat.valid=nullptr;
            pnt_lat.null_count=0;
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

	    double * x=nullptr, *y=nullptr;
	    int num_p=read_point_xy<double>(filename,x,y);
	    gdf_column pnt_x,pnt_y;
 	    
 	    pnt_x.dtype= GDF_FLOAT64;
 	    pnt_x.col_name=(char *)malloc(strlen("x")+ 1);
	    strcpy(pnt_x.col_name,"x");
	    RMM_TRY( RMM_ALLOC(&pnt_x.data, num_p * sizeof(double), 0) );
	    cudaMemcpy(pnt_x.data, x,num_p * sizeof(double) , cudaMemcpyHostToDevice);		
	    pnt_x.size=num_p;
	    pnt_x.valid=nullptr;
	    pnt_x.null_count=0;		
	    delete[] x;

 	    pnt_y.dtype= GDF_FLOAT64;
 	    pnt_y.col_name=(char *)malloc(strlen("y")+ 1);
	    strcpy(pnt_y.col_name,"y");
	    pnt_y.data=nullptr;
	    RMM_TRY( RMM_ALLOC(&pnt_y.data, num_p * sizeof(double), 0) );
	    cudaMemcpy(pnt_y.data, y,num_p * sizeof(double) , cudaMemcpyHostToDevice);		
	    pnt_y.size=num_p;
	    pnt_y.valid=nullptr;
	    pnt_y.null_count=0;
	    delete[] y;
	    
	    return std::make_pair(pnt_x,pnt_y);
	}	
}
