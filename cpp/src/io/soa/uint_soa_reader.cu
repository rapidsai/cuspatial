#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <utilities/error_utils.hpp>
#include <rmm/rmm.h>
#include <cuspatial/soa_readers.hpp>
#include <cuspatial/shared_util.h>

namespace cuSpatial
{
	/**
	 * @Brief read uint32_t (unsigned integer with 32 bit fixed length) data from file as column
	 * see soa_readers.hpp
	*/

	void read_uint_soa(const char *id_uint, gdf_column& values)                                 
	{
    		uint *data=NULL;
    		size_t num_l=read_field<uint32_t>(id_uint,data);
    		if(data==NULL) return;
		
 		values.dtype= GDF_INT32;
 		values.col_name=(char *)malloc(strlen("id")+ 1);
		strcpy(values.col_name,"id");
		RMM_TRY( RMM_ALLOC(&values.data, num_l * sizeof(uint32_t), 0) );
		cudaMemcpy(values.data,data ,num_l * sizeof(uint32_t) , cudaMemcpyHostToDevice);		
		values.size=num_l;
		values.valid=nullptr;
		values.null_count=0;		
		delete[] data;
	}
}