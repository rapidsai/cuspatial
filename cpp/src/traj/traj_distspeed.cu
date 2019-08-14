#include <cudf/utilities/legacy/type_dispatcher.hpp>
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
 * @Brief computing distance(length) and speed of trajectories after their formation (e.g., from coor2traj)

 * @param[in] num_traj: number of trajectories to process
 
 * @param[in] coor_x/coor_y: x and y coordiantes reative to a camera origin ordered by (id,timestamp)

 * @param[in] time: timestamp array ordered by (id,timestamp)

 * @param[in] len: number of points array ordered by (id,timestamp)

 * @param[in] pos: position offsets of trajectories used to index coor_x/coor_y/oid/ts ordered by (id,timestamp)

 * @param[out] dis: computed distances/lengths of trajectories in meters (m)

 * @param[out] sp: computed speed of trajectories in meters per second (m/s)

 * Note: we might output duration (in addtiion to distance/speed), should there is a need;
 * duration can be easiy computed on CPUs by fetching begining/ending timestamps of a trajectory in the time array 
 */
 
template <typename T>
__global__ void distspeed_kernel(gdf_size_type num_traj,const T* const __restrict__ coor_x,const T* const __restrict__ coor_y,
	 const Time *const __restrict__ time,const uint * const __restrict__ len,const uint * const __restrict__ pos,
	 T* const __restrict__ dis, T* const __restrict__ sp)
	 
{
   	 int pid=blockIdx.x*blockDim.x+threadIdx.x;  
   	 if(pid>=num_traj) return;
   	 int bp=(pid==0)?0:pos[pid-1];
   	 int ep=pos[pid]-1;

  	 //assuming the same year --restriction to be removed 	 
  	 float td=(time[ep].yd-time[bp].yd)*86400;
  	 td+=(time[ep].hh-time[bp].hh)*3600;
  	 td+=(time[ep].mm-time[bp].mm)*60;
  	 td+=(time[ep].ss-time[bp].ss);
  	 td+=(time[ep].ms-time[bp].ms)/(float)1000; 	 
 
   	 if((len[pid]<2)||(td==0)||(time[ep].y!=time[bp].y)) 
   	 {
   	 	dis[pid]=-1;
   	 	sp[pid]=-1;
   	 }
   	 else
   	 {
   	 	float ds=0;
   	 	for(int i=0;i<len[pid]-1;i++)
   	 	{
   	 		float dt=(coor_x[bp+i+1]-coor_x[bp+i])*(coor_x[bp+i+1]-coor_x[bp+i]);
   	 		dt+=(coor_y[bp+i+1]-coor_y[bp+i])*(coor_y[bp+i+1]-coor_y[bp+i]);
   	 		ds+=sqrt(dt);
   	 	}
   	 	dis[pid]=ds*1000; //km to m
   	 	sp[pid]=ds*1000/td; // m/s
   	 }
}

struct distspeed_functor {
    template <typename col_type>
    static constexpr bool is_supported()
    {
         return std::is_floating_point<col_type>::value;
    }

    template <typename col_type, std::enable_if_t< is_supported<col_type>() >* = nullptr>
    void operator()(const gdf_column& coor_x,const gdf_column& coor_y,const gdf_column& ts,
 			    const gdf_column& len,const gdf_column& pos,
 			    gdf_column& dist,gdf_column& speed/* ,cudaStream_t stream = 0   */)
    	
    { 
 	dist.dtype= coor_x.dtype;
  	dist.col_name=(char *)malloc(strlen("dist")+ 1);
	strcpy(dist.col_name,"dist");    
        RMM_TRY( RMM_ALLOC(&dist.data, len.size * sizeof(col_type), 0) );
     	dist.size=len.size;
     	dist.valid=nullptr;
     	dist.null_count=0;		

 	speed.dtype= coor_x.dtype;
  	speed.col_name=(char *)malloc(strlen("speed")+ 1);
	strcpy(dist.col_name,"speed");    
        RMM_TRY( RMM_ALLOC(&speed.data, len.size * sizeof(col_type), 0) );
     	speed.size=len.size;
     	speed.valid=nullptr;
     	speed.null_count=0;	
        
        struct timeval t0,t1;
        gettimeofday(&t0, NULL);
        
        gdf_size_type min_grid_size = 0, block_size = 0;
        CUDA_TRY( cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, distspeed_kernel<col_type>) );
        cudf::util::cuda::grid_config_1d grid{coor_x.size, block_size, 1};
        std::cout<<"coor_x.size="<<coor_x.size<<" block_size="<<block_size<<std::endl;
       
        distspeed_kernel<col_type> <<< grid.num_blocks, block_size >>> (len.size,
        	static_cast<col_type*>(coor_x.data),static_cast<col_type*>(coor_y.data),
        	static_cast<Time*>(ts.data),static_cast<uint*>(len.data), static_cast<uint*>(pos.data),
   	    	static_cast<col_type*>(dist.data), static_cast<col_type*>(speed.data) );           
        CUDA_TRY( cudaDeviceSynchronize() );

	gettimeofday(&t1, NULL);
	float distspeed_kernel_time=calc_time("distspeed_kernel_time in ms=",t0,t1);
        //CHECK_STREAM(stream);
        
        int num_print=(len.size<10)?len.size:10;
        std::cout<<"showing the first "<< num_print<<" output records"<<std::endl;
        thrust::device_ptr<col_type> dist_ptr=thrust::device_pointer_cast(static_cast<col_type*>(dist.data));
        thrust::device_ptr<col_type> speed_ptr=thrust::device_pointer_cast(static_cast<col_type*>(speed.data));
        std::cout<<"distance:"<<std::endl;
        thrust::copy(dist_ptr,dist_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl; 
        std::cout<<"speed:"<<std::endl;
 	thrust::copy(speed_ptr,speed_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;     
    }

    template <typename col_type, std::enable_if_t< !is_supported<col_type>() >* = nullptr>
    void operator()(const gdf_column& coor_x,const gdf_column& coor_y,const gdf_column& ts,
 			    const gdf_column& len,const gdf_column& pos,
 			    gdf_column& dist,gdf_column& speed/* ,cudaStream_t stream = 0   */)
    {
        CUDF_FAIL("Non-arithmetic operation is not supported");
    }
};
    

/**
 * @Brief computing distance(length) and speed of trajectories after their formation (e.g., from coor2traj)

 * @param[in] coor_x/coor_y: x and y coordiantes reative to a camera origin ordered by (id,timestamp)

 * @param[in] ts: timestamp column ordered by (id,timestamp)

 * @param[in] len: number of points column ordered by (id,timestamp)

 * @param[in] pos: position offsets of trajectories used to index coor_x/coor_y/oid/ts ordered by (id,timestamp)

 * @param[out] dist: computed distances/lengths of trajectories in meters (m)

 * @param[out] speed: computed speed of trajectories in meters per second (m/s)

 * Note: we might output duration (in addtiion to distance/speed), should there is a need;
 * duration can be easiy computed on CPUs by fetching begining/ending timestamps of a trajectory in the timestamp array 
 */
namespace cuSpatial {

void traj_distspeed(const gdf_column& coor_x,const gdf_column& coor_y,const gdf_column& ts,
 			    const gdf_column& len,const gdf_column& pos,gdf_column& dist,gdf_column& speed
 			    /* ,cudaStream_t stream = 0   */)
{       
    struct timeval t0,t1;
    gettimeofday(&t0, NULL);
    
    CUDF_EXPECTS(coor_x.data != nullptr &&coor_y.data!=nullptr && ts.data!=NULL && len.data!=NULL && pos.data!=NULL,
    	"coor_x/coor_y/ts/len/pos data can not be null");
    CUDF_EXPECTS(coor_x.size == coor_y.size && coor_x.size==ts.size ,"coor_x/coor_y/ts must have the same size");
    CUDF_EXPECTS(len.size == pos.size ,"len/pos must have the same size");
     
    //future versions might allow coor_x/coor_y/ts/pos/len have null_count>0, which might be useful for taking query results as inputs 
    CUDF_EXPECTS(coor_x.null_count == 0 && coor_y.null_count == 0 && ts.null_count==0 && len.null_count==0 &&  pos.null_count==0,
    	"this version does not support coor_x/coor_y/ts/len/pos contains nulls");
    
    CUDF_EXPECTS(coor_x.size >= pos.size ,"one trajectory must have at least one point");
 
  
    cudf::type_dispatcher(coor_x.dtype, distspeed_functor(), coor_x,coor_y,ts,len,pos,dist,speed/*,stream */);
    
    gettimeofday(&t1, NULL);
    float distspeed_end2end_time=calc_time("C++ traj_distspeed end-to-end time in ms=",t0,t1);
    
    }//traj_distspeed     
    	
}// namespace cuSpatial
