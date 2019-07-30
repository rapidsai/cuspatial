#include <utilities/type_dispatcher.hpp>
#include <utilities/cuda_utils.hpp>
#include <type_traits>
#include <thrust/device_vector.h>
#include <sys/time.h>
#include <time.h>

#include <cuspatial/shared_util.h>
#include <cuspatial/traj_util.h>
#include <cuspatial/traj_thrust.h>
#include <cuspatial/traj2.hpp>

using namespace std; 
using namespace cudf;

/**
 * @Brief CUDA kernel for computing spatial bounding boxes of trjectories

 * @param[in] num_traj: number of trajectories to process

 * @param[in] coor_x/coor_y: x and y coordiantes reative to a camera origin ordered by (id,timestamp)

 * @param[in] len: numbers of points of trajectories column ordered by (id,timestamp)

 * @param[in] pos: position offsets of trajectories used to index coor_x/coor_y/ ordered by (id,timestamp)

 * @param[out] bbox_x1/bbox_y1/bbox_x2/bbox_y2: computed spaital bounding boxes in four columns

 */

template <typename T>
__global__ void sbbox_kernel(gdf_size_type num_traj,const T* const __restrict__ coor_x,const T* const __restrict__ coor_y,
	 const uint * const __restrict__ len,const uint * const __restrict__ pos,
	 T* const __restrict__ bbox_x1, T* const __restrict__ bbox_y1,T* const __restrict__ bbox_x2, T* const __restrict__ bbox_y2)
	 
{
   	 int pid=blockIdx.x*blockDim.x+threadIdx.x;  
   	 if(pid>=num_traj) return;
   	 int bp=(pid==0)?0:pos[pid-1];
   	 int ep=pos[pid];

   	 bbox_x2[pid]=bbox_x1[pid]=coor_x[bp];
   	 bbox_y2[pid]=bbox_y1[pid]=coor_y[bp];
   
   	 for(int i=bp+1;i<ep;i++)
   	 {
   	 	if(bbox_x1[pid]>coor_x[i]) bbox_x1[pid]=coor_x[i];
   	 	if(bbox_x2[pid]<coor_x[i]) bbox_x2[pid]=coor_x[i];
   	 	if(bbox_y1[pid]>coor_y[i]) bbox_y1[pid]=coor_y[i];
   	 	if(bbox_y2[pid]<coor_y[i]) bbox_y2[pid]=coor_y[i];
    	 }
}

struct sbbox_functor {
    template <typename col_type>
    static constexpr bool is_supported()
    {
        return std::is_arithmetic<col_type>::value;
    }

    template <typename col_type, std::enable_if_t< is_supported<col_type>() >* = nullptr>
    void operator()(const gdf_column& coor_x,const gdf_column& coor_y,
 		const gdf_column& len,const gdf_column& pos,
		gdf_column& bbox_x1,gdf_column& bbox_y1,gdf_column& bbox_x2,gdf_column& bbox_y2)
    	
    { 
 	bbox_x1.dtype= coor_x.dtype;
  	bbox_x1.col_name=(char *)malloc(strlen("bbox_x1")+ 1);
	strcpy(bbox_x1.col_name,"bbox_x1");    
        RMM_TRY( RMM_ALLOC(&bbox_x1.data, len.size * sizeof(col_type), 0) );
     	bbox_x1.size=len.size;
     	bbox_x1.valid=nullptr;
     	bbox_x1.null_count=0;		
	
	bbox_x2.dtype= coor_x.dtype;
  	bbox_x2.col_name=(char *)malloc(strlen("bbox_x2")+ 1);
	strcpy(bbox_x2.col_name,"bbox_x2");    
        RMM_TRY( RMM_ALLOC(&bbox_x2.data, len.size * sizeof(col_type), 0) );
     	bbox_x2.size=len.size;
     	bbox_x2.valid=nullptr;
     	bbox_x2.null_count=0;		
 	
	bbox_y1.dtype= coor_x.dtype;
  	bbox_y1.col_name=(char *)malloc(strlen("bbox_y1")+ 1);
	strcpy(bbox_y1.col_name,"bbox_y1");    
        RMM_TRY( RMM_ALLOC(&bbox_y1.data, len.size * sizeof(col_type), 0) );
     	bbox_y1.size=len.size;
     	bbox_y1.valid=nullptr;
     	bbox_y1.null_count=0;		
	
	bbox_y2.dtype= coor_x.dtype;
  	bbox_y2.col_name=(char *)malloc(strlen("bbox_y2")+ 1);
	strcpy(bbox_y2.col_name,"bbox_y2");    
        RMM_TRY( RMM_ALLOC(&bbox_y2.data, len.size * sizeof(col_type), 0) );
     	bbox_y2.size=len.size;
     	bbox_y2.valid=nullptr;
     	bbox_y2.null_count=0;	
     	
        struct timeval t0,t1;
        gettimeofday(&t0, NULL);
        
        gdf_size_type min_grid_size = 0, block_size = 0;
        CUDA_TRY( cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, sbbox_kernel<col_type>) );
        cudf::util::cuda::grid_config_1d grid{coor_x.size, block_size, 1};
        std::cout<<"coor_x.size="<<coor_x.size<<" block_size="<<block_size<<std::endl;
       
        sbbox_kernel<col_type> <<< grid.num_blocks, block_size >>> (len.size,
        	static_cast<col_type*>(coor_x.data),static_cast<col_type*>(coor_y.data),static_cast<uint*>(len.data), static_cast<uint*>(pos.data),
   	    	static_cast<col_type*>(bbox_x1.data), static_cast<col_type*>(bbox_y1.data),static_cast<col_type*>(bbox_x2.data), static_cast<col_type*>(bbox_y2.data) );           
        CUDA_TRY( cudaDeviceSynchronize() );

	gettimeofday(&t1, NULL);
	float sbbox_kernel_time=calc_time("spatial bbox kernel time in ms=",t0,t1);
        //CHECK_STREAM(stream);
        
        int num_print=(len.size<10)?len.size:10;
        std::cout<<"showing the first "<< num_print<<" output records"<<std::endl;
        thrust::device_ptr<col_type> x1_ptr=thrust::device_pointer_cast(static_cast<col_type*>(bbox_x1.data));
        thrust::device_ptr<col_type> y1_ptr=thrust::device_pointer_cast(static_cast<col_type*>(bbox_x2.data));
        thrust::device_ptr<col_type> x2_ptr=thrust::device_pointer_cast(static_cast<col_type*>(bbox_y1.data));
        thrust::device_ptr<col_type> y2_ptr=thrust::device_pointer_cast(static_cast<col_type*>(bbox_y2.data));
        std::cout<<"x1:"<<std::endl;
        thrust::copy(x1_ptr,x1_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;     
        std::cout<<"y1:"<<std::endl;
 	thrust::copy(y1_ptr,y1_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;     
 	std::cout<<"x2:"<<std::endl;
        thrust::copy(x2_ptr,x2_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;    
        std::cout<<"y2:"<<std::endl;
 	thrust::copy(y2_ptr,y2_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;     
    }

    template <typename col_type, std::enable_if_t< !is_supported<col_type>() >* = nullptr>
    void operator()(const gdf_column& coor_x,const gdf_column& coor_y,
 		const gdf_column& len,const gdf_column& pos,
		gdf_column& bbox_x1,gdf_column& bbox_y1,gdf_column& bbox_x2,gdf_column& bbox_y2)
    {
        CUDF_FAIL("Non-arithmetic operation is not supported");
    }
};
    

namespace cuSpatial {

/**
 * @Brief computing spatial bounding boxes of trjectories

 * @param[in] coor_x/coor_y: x and y coordiantes reative to a camera origin ordered by (id,timestamp)

 * @param[in] len: numbers of points of trajectories column ordered by (id,timestamp)

 * @param[in] pos: position offsets of trajectories used to index coor_x/coor_y/ ordered by (id,timestamp)

 * @param[out] bbox_x1/bbox_y1/bbox_x2/bbox_y2: computed spaital bounding boxes in four columns

 * Note: temporal 1D bounding box can be computed similary but it seems that there is no such a need;
 * Similar to the dicussion in coor2traj, the temporal 1D bounding box can be retrieved directly
 */
 
void traj_sbbox(const gdf_column& coor_x,const gdf_column& coor_y,
 			const gdf_column& len,const gdf_column& pos,
			gdf_column& bbox_x1,gdf_column& bbox_y1,gdf_column& bbox_x2,gdf_column& bbox_y2)
{       
    struct timeval t0,t1;
    gettimeofday(&t0, NULL);
   
    CUDF_EXPECTS(coor_x.data != nullptr &&coor_y.data!=nullptr && len.data!=NULL && pos.data!=NULL,
    	"coor_x/coor_y/len/pos data can not be null");
    CUDF_EXPECTS(coor_x.size == coor_y.size ,"coor_x/coor_y/ must have the same size");
    CUDF_EXPECTS(len.size == pos.size ,"len/pos must have the same size");
     
    //future versions might allow coor_x/coor_y/pos/len have null_count>0, which might be useful for taking query results as inputs 
    CUDF_EXPECTS(coor_x.null_count == 0 && coor_y.null_count == 0 && len.null_count==0 &&  pos.null_count==0,
    	"this version does not support coor_x/coor_y/len/pos contains nulls");
    
    CUDF_EXPECTS(coor_x.size >= pos.size ,"one trajectory must have at least one point");  
    
    cudf::type_dispatcher(coor_x.dtype, sbbox_functor(), coor_x,coor_y,len,pos,bbox_x1,bbox_y1,bbox_x2,bbox_y2/*,stream */);
    
    // handle null_count if needed 
     
    gettimeofday(&t1, NULL);
    float sbbox_end2end_time=calc_time("spatial bbox end2end time in ms=",t0,t1);
    
    }//traj_distspeed     
    	
}// namespace cuSpatial
