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
#include <utilities/error_utils.hpp>
#include <rmm/rmm.h>
#include <cuspatial/soa_readers.hpp>
#include <utility/utility.hpp>

namespace cuspatial
{
	/**
	 * @brief read uint32_t (unsigned integer with 32 bit fixed length) data from file as column
	 
	 * see soa_readers.hpp
	*/

	gdf_column read_uint32_soa(const char *filename)                                
	{
  		gdf_column values;
   		memset(&values,0,sizeof(gdf_column));
    		
    		uint32_t *data=nullptr;
    		size_t num_l=read_field<uint32_t>(filename,data);
    		if(data==nullptr) 
    			return values;
		
 		values.dtype= GDF_INT32;
 		values.col_name=(char *)malloc(strlen("id")+ 1);
		strcpy(values.col_name,"id");
		RMM_TRY( RMM_ALLOC(&values.data, num_l * sizeof(uint32_t), 0) );
		cudaMemcpy(values.data,data ,num_l * sizeof(uint32_t) , cudaMemcpyHostToDevice);		
		values.size=num_l;
		values.valid=nullptr;
		values.null_count=0;		
		delete[] data;
		
		return values;
	}//read_uint32_soa
}//cuspatial
