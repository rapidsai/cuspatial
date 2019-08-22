#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <rmm/rmm.h>
#include <utilities/error_utils.hpp>
#include <cuspatial/soa_readers.hpp>
#include <cuspatial/shared_util.h>

namespace cuSpatial
{
	/**
	 * @Brief read poygon data from file in SoA format; data type of vertices is fixed to double (GDF_FLOAT64)
	 * see soa_readers.hpp
	*/	
	
	void read_ply_soa(const char *poly_fn,gdf_column& ply_fpos, gdf_column& ply_rpos,
                                   gdf_column& ply_x,gdf_column& ply_y)
	{
	        struct PolyMeta<double> pm;
	        int num_p=read_polygon_soa<double>(poly_fn,pm);
	        if(num_p<=0) return;
                
  		ply_fpos.dtype=GDF_INT32;
  		ply_fpos.col_name=(char *)malloc(strlen("f_pos")+ 1);
		strcpy(ply_fpos.col_name,"f_pos");
		ply_fpos.data=NULL;
		RMM_TRY( RMM_ALLOC(&ply_fpos.data, pm.num_f * sizeof(uint32_t), 0) );
		cudaMemcpy(ply_fpos.data, pm.p_f_len,pm.num_f * sizeof(uint32_t) , cudaMemcpyHostToDevice);
		thrust::device_ptr<uint32_t> d_pfp_ptr=thrust::device_pointer_cast((uint32_t *)ply_fpos.data);
		//prefix-sum: len to pos
		thrust::inclusive_scan(d_pfp_ptr,d_pfp_ptr+pm.num_f,d_pfp_ptr);
		ply_fpos.size=pm.num_f;
		ply_fpos.valid=nullptr;
		ply_fpos.null_count=0;
		delete[] pm.p_f_len;

 		ply_rpos.dtype=GDF_INT32;
 		ply_rpos.col_name=(char *)malloc(strlen("r_pos")+ 1);
		strcpy(ply_rpos.col_name,"r_pos");
		ply_rpos.data=NULL;
		RMM_TRY( RMM_ALLOC(&ply_rpos.data, pm.num_r * sizeof(uint32_t), 0) );
		cudaMemcpy(ply_rpos.data, pm.p_r_len,pm.num_r * sizeof(uint32_t) , cudaMemcpyHostToDevice);
		thrust::device_ptr<uint32_t> d_prp_ptr=thrust::device_pointer_cast((uint32_t *)ply_rpos.data);
		//prefix-sum: len to pos
		thrust::inclusive_scan(d_prp_ptr,d_prp_ptr+pm.num_r,d_prp_ptr);
		ply_rpos.size=pm.num_r;
		ply_rpos.valid=nullptr;
		ply_rpos.null_count=0;
		delete[] pm.p_r_len;

 		ply_x.dtype= GDF_FLOAT64;
 		ply_x.col_name=(char *)malloc(strlen("x")+ 1);
		strcpy(ply_x.col_name,"x");
		RMM_TRY( RMM_ALLOC(&ply_x.data, pm.num_v * sizeof(double), 0) );
		cudaMemcpy(ply_x.data, pm.p_x,pm.num_v * sizeof(double) , cudaMemcpyHostToDevice);		
		ply_x.size=pm.num_v;
		ply_x.valid=nullptr;
		ply_x.null_count=0;		
		delete[] pm.p_x;

 		ply_y.dtype= GDF_FLOAT64;
 		ply_y.col_name=(char *)malloc(strlen("y")+ 1);
		strcpy(ply_y.col_name,"y");
		ply_y.data=NULL;
		RMM_TRY( RMM_ALLOC(&ply_y.data, pm.num_v * sizeof(double), 0) );
		cudaMemcpy(ply_y.data, pm.p_y,pm.num_v * sizeof(double) , cudaMemcpyHostToDevice);		
		ply_y.size=pm.num_v;
		ply_y.valid=nullptr;
		ply_y.null_count=0;
		delete[] pm.p_y;
		
		delete[] pm.p_g_len;
	}//read_ply_soa
}// namespace cuSpatial