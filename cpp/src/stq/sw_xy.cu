
#include <utilities/type_dispatcher.hpp>
#include <utilities/cuda_utils.hpp>
#include <type_traits>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <sys/time.h>
#include <time.h>

#include <cuspatial/shared_util.h>
#include <cuspatial/stq_thrust.h>
#include <cuspatial/stq2.hpp>

using namespace std; 
using namespace cudf;

struct sw_xy_functor {
    template <typename col_type>
    static constexpr bool is_supported()
    {
        return std::is_arithmetic<col_type>::value;
    }

    template <typename col_type, std::enable_if_t< is_supported<col_type>() >* = nullptr>
    int operator()(const gdf_scalar x1,const gdf_scalar y1,const gdf_scalar x2,const gdf_scalar y2,
	const gdf_column& in_x,const gdf_column& in_y, gdf_column& out_x,gdf_column& out_y
		/* , cudaStream_t stream = 0   */)
    {        
        col_type q_x1=*((col_type*)(&(x1.data)));
	col_type q_y1=*((col_type*)(&(y1.data)));
	col_type q_x2=*((col_type*)(&(x2.data)));
        col_type q_y2=*((col_type*)(&(y2.data)));
        std::cout<<"query window: (x1="<<q_x1<<" x2="<<q_x2<<" y1="<<q_y1<<" y2="<<q_y2<<std::endl;
        
        CUDF_EXPECTS(q_x1<q_x2,"x1 must be less than x2 in a spatial window query");
        CUDF_EXPECTS(q_y1<q_y2,"y1 must be less than y2 in a spatial window query");
        
        int num_print=(in_x.size<100)?in_x.size:10;
        std::cout<<"showing the first "<< num_print<<" input records"<<std::endl;

        std::cout<<"x:"<<std::endl;
        thrust::device_ptr<col_type> inx_ptr=thrust::device_pointer_cast(static_cast<col_type*>(in_x.data));
        thrust::copy(inx_ptr,inx_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;  
        std::cout<<"x:"<<std::endl;
        thrust::device_ptr<col_type> iny_ptr=thrust::device_pointer_cast(static_cast<col_type*>(in_y.data));
        thrust::copy(iny_ptr,iny_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;
    	         
        struct timeval t0,t1;
        gettimeofday(&t0, NULL);
       
        auto in_it=thrust::make_zip_iterator(thrust::make_tuple(inx_ptr,iny_ptr));
  	int num_hits= thrust::count_if(thrust::device, in_it, in_it+in_x.size, sw_functor_xy<col_type>(q_x1,q_x2,q_y1,q_y2));
        std::cout<<"#hits="<<num_hits<<std::endl;
   
        out_x.dtype= in_x.dtype;
        out_x.col_name=(char *)malloc(strlen("coord_x")+ 1);
       	strcpy(out_x.col_name,"coord_x");    
        RMM_TRY( RMM_ALLOC(&out_x.data, num_hits* sizeof(col_type), 0) );
        out_x.size=in_x.size;
        out_x.valid=nullptr;
        out_x.null_count=0;		
       
        out_y.dtype= in_y.dtype;
        out_y.col_name=(char *)malloc(strlen("coord_y")+ 1);
       	strcpy(out_x.col_name,"coord_x");    
        RMM_TRY( RMM_ALLOC(&out_y.data, num_hits*sizeof(col_type), 0) );
        out_y.size=in_y.size;
        out_y.valid=nullptr;
     	out_y.null_count=0;	
        
        num_print=(num_hits<10)?num_hits:10;
        thrust::device_ptr<col_type> outx_ptr=thrust::device_pointer_cast(static_cast<col_type*>(out_x.data));
        thrust::device_ptr<col_type> outy_ptr=thrust::device_pointer_cast(static_cast<col_type*>(out_y.data));
        auto out_it=thrust::make_zip_iterator(thrust::make_tuple(outx_ptr,outy_ptr));
        thrust::copy_if(thrust::device, in_it, in_it+in_x.size,out_it, sw_functor_xy<col_type>(q_x1,q_x2,q_y1,q_y2));
        
	gettimeofday(&t1, NULL);
	float swxy_kernel_time=calc_time("swxy kernel time in ms=",t0,t1);
        //CHECK_STREAM(stream);
    
        std::cout<<"showing the first "<< num_print<<" output records"<<std::endl;
        std::cout<<"x:"<<std::endl;
        thrust::copy(outx_ptr,outx_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;  
        std::cout<<"y:"<<std::endl;
        thrust::copy(outy_ptr,outy_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;  
    
        return num_hits;
    }

    template <typename col_type, std::enable_if_t< !is_supported<col_type>() >* = nullptr>
    int operator()(const gdf_scalar x1,const gdf_scalar y1,const gdf_scalar x2,const gdf_scalar y2,
	const gdf_column& in_x,const gdf_column& in_y, gdf_column& out_x,gdf_column& out_y
	/* , cudaStream_t stream = 0   */)
    {
        CUDF_FAIL("Non-arithmetic operation is not supported");
    }
};
    

/**
 * @Brief retrive all points (x,y) that fall within a query window (x1,y1,x2,y2) and output the filtered points
 
 * @param[in] x1/x2/y1/y2: defines the query window

 * @param[in] in_x/in_y: pointer/array of x/y coordiantes of points to be queried

 * @param[out] out_x/out_y: pointer/array of x/y coordiantes of points that fall within the query window

 * @returns the number of points that satisfy the spatial window query criteria
 */
 
namespace cuSpatial {

int sw_xy(const gdf_scalar x1,const gdf_scalar y1,const gdf_scalar x2,const gdf_scalar y2,
	const gdf_column& in_x,const gdf_column& in_y, gdf_column& out_x,gdf_column& out_y
	/* , cudaStream_t stream = 0   */)
{       
    struct timeval t0,t1;
    gettimeofday(&t0, NULL);
    
    CUDF_EXPECTS(in_x.dtype == in_y.dtype, "point type mismatch between x/y arrays");
    CUDF_EXPECTS(in_x.size == in_y.size, "#of points mismatch between x/y arrays");
    
    
    CUDF_EXPECTS(in_x.null_count == 0 && in_y.null_count == 0, "this version does not support point data that contains nulls");
    
    int num_traj = cudf::type_dispatcher( in_x.dtype, sw_xy_functor(), 
    		x1,y1,x2,y2,in_x,in_y, out_x,out_y/*,stream */);
    		
    gettimeofday(&t1, NULL);
    float swxy_end2end_time=calc_time("C++ sw_xy end-to-end time in ms= ",t0,t1);
    
    return num_traj;
  }//sw_xy 
  
}// namespace cuSpatail
