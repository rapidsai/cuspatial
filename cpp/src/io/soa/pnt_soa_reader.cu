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
	 * @Brief read x/y from file into two columns; data type is fixed to double (GDF_FLOAT64)
	*
	*/
	void read_pnt_soa(const char *pnt_fn,gdf_column& pnt_x,gdf_column& pnt_y)                                   
	{

		double * p_x=NULL, *p_y=NULL;
		int num_p=read_point_xy<double>(pnt_fn,p_x,p_y);
		
 		pnt_x.dtype= GDF_FLOAT64;
 		pnt_x.col_name=(char *)malloc(strlen("x")+ 1);
		strcpy(pnt_x.col_name,"x");
		RMM_TRY( RMM_ALLOC(&pnt_x.data, num_p * sizeof(double), 0) );
		cudaMemcpy(pnt_x.data, p_x,num_p * sizeof(double) , cudaMemcpyHostToDevice);		
		pnt_x.size=num_p;
		pnt_x.valid=nullptr;
		pnt_x.null_count=0;		
		delete[] p_x;

 		pnt_y.dtype= GDF_FLOAT64;
 		pnt_y.col_name=(char *)malloc(strlen("y")+ 1);
		strcpy(pnt_y.col_name,"y");
		pnt_y.data=NULL;
		RMM_TRY( RMM_ALLOC(&pnt_y.data, num_p * sizeof(double), 0) );
		cudaMemcpy(pnt_y.data, p_y,num_p * sizeof(double) , cudaMemcpyHostToDevice);		
		pnt_y.size=num_p;
		pnt_y.valid=nullptr;
		pnt_y.null_count=0;
		delete[] p_y;
	}
}