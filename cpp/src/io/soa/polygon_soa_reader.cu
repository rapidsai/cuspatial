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
#include <rmm/rmm.h>
#include <cudf/types.h>
#include <utilities/error_utils.hpp>
#include <cuspatial/soa_readers.hpp>
#include <utility/utility.hpp>

namespace cuspatial
{
	/**
	 * @brief read poygon data from file in SoA format; data type of vertices is fixed to double (GDF_FLOAT64)
	 * see soa_readers.hpp
	*/	
	
	void read_polygon_soa(const char *filename,gdf_column* ply_fpos, gdf_column* ply_rpos,
		gdf_column* ply_x,gdf_column* ply_y)
	{
	        CUDF_EXPECTS(ply_fpos!=nullptr && ply_rpos!=nullptr && ply_x!=nullptr && ply_y!=nullptr,
	        	"none of the polygon offset/ring offset/x coorinate/y coordinate column can be null ");
	        
	        memset(ply_fpos,0,sizeof(gdf_column));
	        memset(ply_rpos,0,sizeof(gdf_column));
	        memset(ply_x,0,sizeof(gdf_column));
	        memset(ply_y,0,sizeof(gdf_column));
	
	        struct polygons<double> pm;
	        read_polygon_soa<double>(filename,&pm);	        
	        if(pm.num_feature<=0) return;
	        
  		ply_fpos->dtype=GDF_INT32;
  		ply_fpos->col_name=(char *)malloc(strlen("f_pos")+ 1);
		strcpy(ply_fpos->col_name,"f_pos");
		ply_fpos->data=nullptr;
		RMM_TRY( RMM_ALLOC(&(ply_fpos->data), pm.num_feature * sizeof(uint32_t), 0) );
		cudaMemcpy(ply_fpos->data, pm.feature_length,pm.num_feature * sizeof(uint32_t) , cudaMemcpyHostToDevice);
		thrust::device_ptr<uint32_t> d_pfp_ptr=thrust::device_pointer_cast((uint32_t *)(ply_fpos->data));
		//prefix-sum: len to pos
		thrust::inclusive_scan(d_pfp_ptr,d_pfp_ptr+pm.num_feature,d_pfp_ptr);
		ply_fpos->size=pm.num_feature;
		ply_fpos->valid=nullptr;
		ply_fpos->null_count=0;
		delete[] pm.feature_length;

 		ply_rpos->dtype=GDF_INT32;
 		ply_rpos->col_name=(char *)malloc(strlen("r_pos")+ 1);
		strcpy(ply_rpos->col_name,"r_pos");
		ply_rpos->data=nullptr;
		RMM_TRY( RMM_ALLOC(&(ply_rpos->data), pm.num_ring * sizeof(uint32_t), 0) );
		cudaMemcpy(ply_rpos->data, pm.ring_length,pm.num_ring * sizeof(uint32_t) , cudaMemcpyHostToDevice);
		thrust::device_ptr<uint32_t> d_prp_ptr=thrust::device_pointer_cast((uint32_t *)(ply_rpos->data));
		//prefix-sum: len to pos
		thrust::inclusive_scan(d_prp_ptr,d_prp_ptr+pm.num_ring,d_prp_ptr);
		ply_rpos->size=pm.num_ring;
		ply_rpos->valid=nullptr;
		ply_rpos->null_count=0;
		delete[] pm.ring_length;

 		ply_x->dtype= GDF_FLOAT64;
 		ply_x->col_name=(char *)malloc(strlen("x")+ 1);
		strcpy(ply_x->col_name,"x");
		RMM_TRY( RMM_ALLOC(&(ply_x->data), pm.num_vertex * sizeof(double), 0) );
		cudaMemcpy(ply_x->data, pm.x,pm.num_vertex * sizeof(double) , cudaMemcpyHostToDevice);		
		ply_x->size=pm.num_vertex;
		ply_x->valid=nullptr;
		ply_x->null_count=0;		
		delete[] pm.x;

 		ply_y->dtype= GDF_FLOAT64;
 		ply_y->col_name=(char *)malloc(strlen("y")+ 1);
		strcpy(ply_y->col_name,"y");
		ply_y->data=nullptr;
		RMM_TRY( RMM_ALLOC(&(ply_y->data), pm.num_vertex * sizeof(double), 0) );
		cudaMemcpy(ply_y->data, pm.y,pm.num_vertex * sizeof(double) , cudaMemcpyHostToDevice);		
		ply_y->size=pm.num_vertex;
		ply_y->valid=nullptr;
		ply_y->null_count=0;
		delete[] pm.y;
		
		delete[] pm.group_length;
	}//read_polygon_soa
}// namespace cuspatial
