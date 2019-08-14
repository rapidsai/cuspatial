#include <sys/time.h>
#include <time.h>

#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <utilities/cuda_utils.hpp>
#include <type_traits>
#include <thrust/device_vector.h>
#include <cuspatial/shared_util.h>
#include <cuspatial/pip2.hpp>

using namespace std; 
using namespace cudf;

/** 
 */


 template <typename T>
 __global__ void pip2_kernel(gdf_size_type pnt_size,const T* const __restrict__ pnt_x,const T* const __restrict__ pnt_y,
	gdf_size_type ply_size,const uint* const __restrict__ ply_fpos,const uint* const __restrict__ ply_rpos,	
        const T* const __restrict__ ply_x,const T* const __restrict__ ply_y,
        uint* const __restrict__ res_bm)
{
    uint mask=0;
    //assuming 1D grid/block config
    uint idx =blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=pnt_size) return;
    
    T x = pnt_x[idx];
    T y = pnt_y[idx];
    for (uint j = 0; j < ply_size; j++) //for each polygon
    {
       uint r_f = (0 == j) ? 0 : ply_fpos[j-1];
       uint r_t=ply_fpos[j];
       bool in_polygon = false;
       for (uint k = r_f; k < r_t; k++) //for each ring
       {
           uint m = (k==0)?0:ply_rpos[k-1];
           
           /*if(idx==0)
           	printf("%d %d %d %d %d %d\n",j,k,r_f,r_t,m,ply_rpos[k]-1);
           __syncthreads();*/
           	
           for (;m < ply_rpos[k]-1; m++) //for each line segment
           {
              T x0, x1, y0, y1;
              x0 = ply_x[m];
              y0 = ply_y[m];
              x1 = ply_x[m+1];
              y1 = ply_y[m+1];
              
              /*if(idx==0)
              	printf("idx=%3d: %3d %3d %3d %15.10f %15.10f %15.10f %15.10f %15.10f %15.10f \n",idx,j,k,m,x,y,x0,y0,x1,y1);
               __syncthreads();*/
               
              if ((((y0 <= y) && (y < y1)) ||
                   ((y1 <= y) && (y < y0))) &&
                       (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0))
                 in_polygon = !in_polygon;
            }
      }
      if(in_polygon)
      	mask|=(0x01<<j);
   }
   res_bm[idx]=mask;
   //printf("idx=%3d: %08x\n",idx,mask);

}

struct pip2_functor {
    template <typename col_type>
    static constexpr bool is_supported()
    {
        return std::is_floating_point<col_type>::value;
    }

    template <typename col_type, std::enable_if_t< is_supported<col_type>() >* = nullptr>
    gdf_column operator()(gdf_column const & pnt_x,gdf_column const & pnt_y,
 			  gdf_column const & ply_fpos,gdf_column const & ply_rpos,
			  gdf_column const & ply_x,gdf_column const & ply_y /* ,cudaStream_t stream = 0   */)
    {
        gdf_column res_bm;
        uint* data;
        
        /*cout<<"output coordinate in pip2.cu<<std::endl;
        thrust::device_ptr<col_type> d_pntx_ptr=thrust::device_pointer_cast(static_cast<col_type*>(pnt_x.data));
        thrust::copy(d_pntx_ptr,d_pntx_ptr+pnt_x.size,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;  
        thrust::device_ptr<col_type> d_pnty_ptr=thrust::device_pointer_cast(static_cast<col_type*>(pnt_y.data));
        thrust::copy(d_pnty_ptr,d_pnty_ptr+pnt_y.size,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;  

        thrust::device_ptr<uint> d_fpos_ptr=thrust::device_pointer_cast(static_cast<uint*>(ply_fpos.data));
        thrust::copy(d_fpos_ptr,d_fpos_ptr+ply_fpos.size,std::ostream_iterator<uint>(std::cout, " "));std::cout<<std::endl;       
        thrust::device_ptr<uint> d_rpos_ptr=thrust::device_pointer_cast(static_cast<uint*>(ply_rpos.data));
        thrust::copy(d_rpos_ptr,d_rpos_ptr+ply_rpos.size,std::ostream_iterator<uint>(std::cout, " "));std::cout<<std::endl;       
    	
    	thrust::device_ptr<col_type> d_plyx_ptr=thrust::device_pointer_cast(static_cast<col_type*>(ply_x.data));
        thrust::copy(d_plyx_ptr,d_plyx_ptr+ply_x.size,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;  
        thrust::device_ptr<col_type> d_plyy_ptr=thrust::device_pointer_cast(static_cast<col_type*>(ply_y.data));
        thrust::copy(d_plyy_ptr,d_plyy_ptr+ply_y.size,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;*/       
        
        RMM_TRY( RMM_ALLOC(&data, pnt_y.size * sizeof(uint), 0) );
        gdf_column_view(&res_bm, data, nullptr, pnt_y.size, GDF_INT32);

        struct timeval t0,t1;
        gettimeofday(&t0, NULL);
        
        gdf_size_type min_grid_size = 0, block_size = 0;
        CUDA_TRY( cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, pip2_kernel<col_type>) );
        cudf::util::cuda::grid_config_1d grid{pnt_y.size, block_size, 1};
        
        std::cout<<"pnt_x.size="<<pnt_x.size<<" poly_size="<<ply_fpos.size<<" block_size="<<block_size<<std::endl;
        
        pip2_kernel<col_type> <<< grid.num_blocks, block_size >>> (pnt_x.size,
               	static_cast<col_type*>(pnt_x.data), static_cast<col_type*>(pnt_y.data),
        	ply_fpos.size,static_cast<uint*>(ply_fpos.data),static_cast<uint*>(ply_rpos.data),
        	static_cast<col_type*>(ply_x.data), static_cast<col_type*>(ply_y.data),
                static_cast<uint*>(res_bm.data) );
        CUDA_TRY( cudaDeviceSynchronize() );
	
	gettimeofday(&t1, NULL);	
 	float pip2_kernel_time=calc_time("pip2_kernel_time in ms=",t0,t1);
        
        //CHECK_STREAM(stream);
        /*thrust::device_ptr<uint> d_resbm_ptr=thrust::device_pointer_cast(static_cast<uint*>(res_bm.data));
        thrust::copy(d_resbm_ptr,d_resbm_ptr+pnt_x.size,std::ostream_iterator<uint>(std::cout, " "));std::cout<<std::endl;*/       
 
        return res_bm;
    }

    template <typename col_type, std::enable_if_t< !is_supported<col_type>() >* = nullptr>
    gdf_column operator()(gdf_column const & pnt_x,gdf_column const & pnt_y,
 			  gdf_column const & ply_fpos,gdf_column const & ply_rpos,
			  gdf_column const & ply_x,gdf_column const & ply_y
			  /*,cudaStream_t stream = 0 */)
    {
        CUDF_FAIL("Non-arithmetic operation is not supported");
    }
};


/**
 */

namespace cuSpatial {

gdf_column pip2_bm(const gdf_column& pnt_x,const gdf_column& pnt_y,
                                   const gdf_column& ply_fpos, const gdf_column& ply_rpos,
                                   const gdf_column& ply_x,const gdf_column& ply_y
                          /* ,cudaStream_t stream */)
{       
    struct timeval t0,t1;
    gettimeofday(&t0, NULL);
    
    CUDF_EXPECTS(pnt_y.data != nullptr && pnt_x.data != nullptr, "query point data cannot be empty");
    CUDF_EXPECTS(pnt_y.dtype == pnt_x.dtype, "polygon vertex and point data type mismatch for x array ");
    
    //future versions might allow pnt_(x/y) have null_count>0, which might be useful for taking query results as inputs 
    CUDF_EXPECTS(pnt_x.null_count == 0 && pnt_y.null_count == 0, "this version does not support pnt_x/pnt_y contains nulls");
  
    CUDF_EXPECTS(ply_fpos.data != nullptr &&ply_rpos.data!=nullptr, "polygon index cannot be empty");
    CUDF_EXPECTS(ply_fpos.size >0 && (size_t)ply_fpos.size<=sizeof(uint)*8, "#polygon of polygons can not exceed bitmap capacity (32 for unsigned int)");
    CUDF_EXPECTS(ply_y.data != nullptr && ply_x.data != nullptr, "polygon data cannot be empty");
    CUDF_EXPECTS(ply_fpos.size <=ply_rpos.size,"#of polygons must be equal or less than # of rings (one polygon has at least one ring");
    CUDF_EXPECTS(ply_y.size == ply_x.size, "polygon vertice sizes mismatch between x/y arrays");
    CUDF_EXPECTS(pnt_y.size == pnt_x.size, "query points size mismatch from between x/y arrays");
    CUDF_EXPECTS(ply_y.dtype == ply_x.dtype, "polygon vertex data type mismatch between x/y arrays");
    CUDF_EXPECTS(ply_y.dtype == pnt_y.dtype, "polygon vertex and point data type mismatch for y array");
    CUDF_EXPECTS(ply_x.null_count == 0 && ply_y.null_count == 0, "polygon should not contain nulls");
    
    gdf_column res_bm = cudf::type_dispatcher( pnt_x.dtype, pip2_functor(), 
    		pnt_x,pnt_y,ply_fpos,ply_rpos,ply_x,ply_y /*,stream */);
    		
    gettimeofday(&t1, NULL);
    float pip2_end2end_time=calc_time("C++ pip2_bm end-to-end time in ms=",t0,t1);
    
    return res_bm;
  }//pip2 
  
}// namespace cuSpatial
