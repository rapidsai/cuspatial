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
	 * @Brief read ID (ids: uint type) data from file as column
	*
	*/

	void read_id_soa(const char *id_fn, gdf_column& ids)                                 
	{
    		uint *objid=NULL;
    		size_t num_l=read_field<uint>(id_fn,objid);
    		if(objid==NULL) return;
		
 		ids.dtype= GDF_INT32;
 		ids.col_name=(char *)malloc(strlen("id")+ 1);
		strcpy(ids.col_name,"id");
		RMM_TRY( RMM_ALLOC(&ids.data, num_l * sizeof(uint), 0) );
		cudaMemcpy(ids.data,objid ,num_l * sizeof(uint) , cudaMemcpyHostToDevice);		
		ids.size=num_l;
		ids.valid=nullptr;
		ids.null_count=0;		
		delete[] objid;
	}
}