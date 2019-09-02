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
	* @brief read timestamp (ts: Time type) data from file as column
	 
	* see soa_readers.hpp
	*/
	gdf_column read_timestamp_soa(const char *filename)
                             
	{
    		gdf_column ts;
    		memset(&ts,0,sizeof(gdf_column));
    		
    		struct its_timestamp * time=nullptr;
    		size_t num_t=read_field<its_timestamp>(filename,time);
    		if(time==nullptr) 
    			return ts;
    		 		
 		ts.dtype= GDF_INT64;
 		ts.col_name=(char *)malloc(strlen("ts")+ 1);
		strcpy(ts.col_name,"ts");
		//make sure sizeof(TIME)==sizeof(GDF_INT64)
		RMM_TRY( RMM_ALLOC(&ts.data, num_t * sizeof(its_timestamp), 0) );
		cudaMemcpy(ts.data,time ,num_t * sizeof(its_timestamp) , cudaMemcpyHostToDevice);		
		ts.size=num_t;
		ts.valid=nullptr;
		ts.null_count=0;		
		delete[] time;
		
		return ts;
	}//read_timestamp_soa
}//cuspatial