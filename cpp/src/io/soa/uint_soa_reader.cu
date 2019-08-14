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
	 * @Brief read uint (unsigned integer) data from file as column
	*
	*/

	void read_uint_soa(const char *id_uint, gdf_column& values)                                 
	{
    		uint *data=NULL;
    		size_t num_l=read_field<uint>(id_uint,data);
    		if(data==NULL) return;
		
 		values.dtype= GDF_INT32;
 		values.col_name=(char *)malloc(strlen("id")+ 1);
		strcpy(values.col_name,"id");
		RMM_TRY( RMM_ALLOC(&values.data, num_l * sizeof(uint), 0) );
		cudaMemcpy(values.data,data ,num_l * sizeof(uint) , cudaMemcpyHostToDevice);		
		values.size=num_l;
		values.valid=nullptr;
		values.null_count=0;		
		delete[] data;
	}
}