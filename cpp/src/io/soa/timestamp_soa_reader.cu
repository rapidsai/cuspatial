#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cudf/utilities/error.hpp>
#include <rmm/rmm.h>
#include <cudf/types.h>
#include <cudf/legacy/column.hpp>
#include <cuspatial/soa_readers.hpp>
#include <utility/legacy/utility.hpp>

namespace cuspatial
{
    /**
	* @brief read timestamp (ts: Time type) data from file as column
	 
    * see soa_readers.hpp
    */

    gdf_column read_timestamp_soa(const char *filename)                      
    {
        gdf_column ts;
        memset(&ts,0,sizeof(gdf_column));
    		
        struct its_timestamp * timestamp=nullptr;
        size_t num_t=read_field<its_timestamp>(filename,timestamp);
        if(timestamp==nullptr) 
            return ts;
 
        its_timestamp* temp_ts{nullptr};
        RMM_TRY( RMM_ALLOC(&temp_ts, num_t * sizeof(its_timestamp), 0) );
        cudaStream_t stream{0};
        CUDA_TRY( cudaMemcpyAsync(temp_ts, timestamp,
                                  num_t * sizeof(its_timestamp) , 
                                  cudaMemcpyHostToDevice,stream) );		
        gdf_column_view_augmented(&ts, temp_ts, nullptr, num_t,
                               GDF_INT64, 0,
                               gdf_dtype_extra_info{TIME_UNIT_NONE}, "timestamp");          
  	delete[] timestamp;
  
        return ts;
    }//read_timestamp_soa
    
}//cuspatial
