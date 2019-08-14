#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <utilities/cuda_utils.hpp>
#include <type_traits>
#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <sys/time.h>
#include <time.h>

#include <cuspatial/shared_util.h>
#include <cuspatial/traj2.hpp>
#include <cuspatial/traj_thrust.h>

using namespace std; 
using namespace cudf;

struct coor2traj_functor {
    template <typename col_type>
    static constexpr bool is_supported()
    {
         return std::is_floating_point<col_type>::value;
    }

    template <typename col_type, std::enable_if_t< is_supported<col_type>() >* = nullptr>
    int operator()(gdf_column& coor_x,gdf_column& coor_y,gdf_column& oid, gdf_column& ts, 
 			    gdf_column& tid, gdf_column& len,gdf_column& pos/* ,cudaStream_t stream = 0   */)
    {        
        int num_print=(oid.size<10)?oid.size:10;
        std::cout<<"showing the first "<< num_print<<" input records before sort"<<std::endl;

        std::cout<<"x"<<std::endl;
        thrust::device_ptr<col_type> coorx_ptr=thrust::device_pointer_cast(static_cast<col_type*>(coor_x.data));
        thrust::copy(coorx_ptr,coorx_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;  
        std::cout<<"y"<<std::endl;
        thrust::device_ptr<col_type> coory_ptr=thrust::device_pointer_cast(static_cast<col_type*>(coor_y.data));
        thrust::copy(coory_ptr,coory_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;  
	
	std::cout<<"oid"<<std::endl;
        thrust::device_ptr<uint> id_ptr=thrust::device_pointer_cast(static_cast<uint*>(oid.data));
        thrust::copy(id_ptr,id_ptr+num_print,std::ostream_iterator<uint>(std::cout, " "));std::cout<<std::endl;  
        thrust::device_ptr<Time> time_ptr=thrust::device_pointer_cast(static_cast<Time *>(ts.data));
        std::cout<<"timestamp"<<std::endl;
        thrust::copy(time_ptr,time_ptr+num_print,std::ostream_iterator<Time>(std::cout, " "));std::cout<<std::endl;    
    	         
        struct timeval t0,t1;
        gettimeofday(&t0, NULL);
        
        uint num_rec=oid.size;
        auto od_it=thrust::make_zip_iterator(thrust::make_tuple(id_ptr,coorx_ptr,coory_ptr));
        thrust::stable_sort_by_key(time_ptr,time_ptr+num_rec,od_it);
        auto tl_it=thrust::make_zip_iterator(thrust::make_tuple(time_ptr,coorx_ptr,coory_ptr));
        thrust::stable_sort_by_key(id_ptr,id_ptr+num_rec,tl_it);
        
        //allocate sufficient memory to hold id,cnt and pos before reduce_by_key        
        uint *objcnt=NULL,*objpos=NULL,*objid=NULL;
        RMM_TRY( RMM_ALLOC((void**)&objcnt,num_rec* sizeof(uint),0) ) ; 
        RMM_TRY( RMM_ALLOC((void**)&objpos,num_rec* sizeof(uint),0) ) ; 
        RMM_TRY( RMM_ALLOC((void**)&objid,num_rec* sizeof(uint),0) ) ; 
        
        thrust::device_ptr<uint> objid_ptr=thrust::device_pointer_cast(objid);
        thrust::device_ptr<uint> objcnt_ptr=thrust::device_pointer_cast(objcnt);
        thrust::device_ptr<uint> objpos_ptr=thrust::device_pointer_cast(objpos);
        
	int num_traj=thrust::reduce_by_key(thrust::device,id_ptr,id_ptr+num_rec,
   		thrust::constant_iterator<int>(1),objid_ptr,objcnt_ptr).second-objcnt_ptr;
        std::cout<<"#traj="<<num_traj<<std::endl;

	//allocate just enough memory (num_traj), copy over and then free large (num_rec) arrays         
        uint *trajid=NULL,*trajcnt=NULL,*trajpos=NULL;
        RMM_TRY( RMM_ALLOC((void**)&trajid,num_traj* sizeof(uint),0) ) ; 
        RMM_TRY( RMM_ALLOC((void**)&trajcnt,num_traj* sizeof(uint),0) ) ; 
        RMM_TRY( RMM_ALLOC((void**)&trajpos,num_traj* sizeof(uint),0) ) ; 
        
        thrust::device_ptr<uint> trajid_ptr=thrust::device_pointer_cast(trajid);
        thrust::device_ptr<uint> trajcnt_ptr=thrust::device_pointer_cast(trajcnt);
        thrust::device_ptr<uint> trajpos_ptr=thrust::device_pointer_cast(trajpos);        
        
        thrust::copy(objid_ptr,objid_ptr+num_traj,trajid);
        thrust::copy(objcnt_ptr,objcnt_ptr+num_traj,trajcnt);
        thrust::copy(objpos_ptr,objpos_ptr+num_traj,trajpos);
        
        RMM_TRY( RMM_FREE(objid, 0) );
        RMM_TRY( RMM_FREE(objcnt, 0) );
        RMM_TRY( RMM_FREE(objpos, 0) );
        
        //to avoid lost memory problem when tid/cnt/pos gdf columns are associated with dvice memory
        gdf_column_view(&tid, trajid, nullptr, num_traj, GDF_INT32);
        gdf_column_view(&len, trajcnt, nullptr, num_traj, GDF_INT32);
        thrust::inclusive_scan(thrust::device,trajcnt_ptr,trajcnt_ptr+num_traj,trajpos_ptr);
        gdf_column_view(&pos, trajpos, nullptr, num_traj, GDF_INT32);  
      
	gettimeofday(&t1, NULL);
        float coor2traj_kernel_time=calc_time("coor2traj kernel time in ms=",t0,t1);
        //CHECK_STREAM(stream);
    
   	std::cout<<"showing the first "<< num_print<<" records aftr sort"<<std::endl;
        std::cout<<"x"<<std::endl;
        thrust::copy(coorx_ptr,coorx_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;  
        std::cout<<"y"<<std::endl;
        thrust::copy(coory_ptr,coory_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;  
    	
    	std::cout<<"oid"<<std::endl;
        thrust::copy(id_ptr,id_ptr+num_print,std::ostream_iterator<uint>(std::cout, " "));std::cout<<std::endl;  
        std::cout<<"timestamp"<<std::endl;
        thrust::copy(time_ptr,time_ptr+num_print,std::ostream_iterator<Time>(std::cout, " "));std::cout<<std::endl;  
        
        num_print=(num_traj<10)?num_traj:10;
        std::cout<<"showing the first "<< num_print<<" trajectory records"<<std::endl;
        std::cout<<"trajectory id"<<std::endl;
        thrust::copy(trajid_ptr,trajid_ptr+num_print,std::ostream_iterator<uint>(std::cout, " "));std::cout<<std::endl;
        std::cout<<"trajectory #of points"<<std::endl;
        thrust::copy(trajcnt_ptr,trajcnt_ptr+num_print,std::ostream_iterator<uint>(std::cout, " "));std::cout<<std::endl;
        std::cout<<"trajectory poisition index on sorted point x/y array"<<std::endl;
        thrust::copy(trajpos_ptr,trajpos_ptr+num_print,std::ostream_iterator<uint>(std::cout, " "));std::cout<<std::endl;      
 
        return num_traj;
    }

    template <typename col_type, std::enable_if_t< !is_supported<col_type>() >* = nullptr>
    int operator()(gdf_column& coor_x,gdf_column& coor_y,gdf_column& oid, gdf_column& ts, 
 			    gdf_column& tid, gdf_column& len,gdf_column& pos/* ,cudaStream_t stream = 0   */)
    {
        CUDF_FAIL("Non-arithmetic operation is not supported");
    }
};
    

namespace cuSpatial {

/**
 * @Brief deriving trajectories from points (x/y relative to an origin), timestamps and objectids
 * by first sorting based on id and timestamp and then group by id.

 * @param[in/out] coor_x/coor_x: x and y coordiantes reative to a camera origin (before/after sorting)

 * @param[in/out] oid: object (e.g., vehicle) id column (before/after sorting); upon completion, unqiue ids become trajectory ids

 * @param[in/out] ts: timestamp column (before/after sorting)

 * @param[out] tid: trajectory id column (see comments on oid)

 * @param[out] len: #of points in the derived trajectories

 * @param[out] pos: position offsets of trajectories used to index coor_x/coor_x/oid/ts after completion

 * @returns the number of derived trajectory

 */

int coor2traj(gdf_column& coor_x,gdf_column& coor_y,gdf_column& oid, gdf_column& ts, 
 			    gdf_column& tid,gdf_column& len,gdf_column& pos/* ,cudaStream_t stream = 0   */)
{       
    struct timeval t0,t1;
    gettimeofday(&t0, NULL);
    
    CUDF_EXPECTS(coor_x.data != nullptr &&coor_y.data!=nullptr&&oid.data!=NULL&&ts.data!=NULL, "coor_x/coor_y/oid/ts data can not be null");
    CUDF_EXPECTS(coor_x.size == coor_y.size && coor_x.size==oid.size && coor_x.size==ts.size ,"coor_x/coor_y/oid/ts must have the same size");
    
    //future versions might allow coor_x/coor_y/oid/ts have null_count>0, which might be useful for taking query results as inputs 
    CUDF_EXPECTS(coor_x.null_count == 0 && coor_y.null_count == 0 && oid.null_count==0 && ts.null_count==0, 
    	"this version does not support coor_x/coor_y/oid/ts contains nulls");
    
    int num_traj = cudf::type_dispatcher( coor_x.dtype, coor2traj_functor(), 
    		coor_x,coor_y,oid,ts,tid,len,pos /*,stream */);
    		
    gettimeofday(&t1, NULL);
    float coor2traj_end2end_time=calc_time("coor2traj end-to-end time in ms=",t0,t1);
    
    return num_traj;
  }//coor2traj 
  
}// namespace cuSpatial
