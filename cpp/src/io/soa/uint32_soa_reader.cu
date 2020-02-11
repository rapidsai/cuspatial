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
#include <rmm/rmm.h>
#include <cudf/types.h>
#include <cudf/legacy/column.hpp>
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
        
        uint32_t* temp_val{nullptr};
        RMM_TRY( RMM_ALLOC(&temp_val, num_l * sizeof(uint32_t), 0) );
        cudaStream_t stream{0};
        CUDA_TRY( cudaMemcpyAsync(temp_val, data,
                                  num_l * sizeof(uint32_t) , 
                                  cudaMemcpyHostToDevice,stream) );		
        gdf_column_view_augmented(&values, temp_val, nullptr, num_l,
                               GDF_INT32, 0,
                               gdf_dtype_extra_info{TIME_UNIT_NONE}, "id");  		
        return values;
    }//read_uint32_soa
    
}//cuspatial
