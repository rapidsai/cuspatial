/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <utilities/cuda_utils.hpp>
#include <type_traits>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/discard_iterator.h>
#include <sys/time.h>
#include <time.h>

#include <utility/utility.hpp>
#include <utility/trajectory_thrust.cuh>
#include <cuspatial/trajectory.hpp>

using namespace std; 
using namespace cudf;
using namespace cuspatial;

struct is_true
{
	__device__
	bool operator()(const bool t)
	{
		return(t);
	}
};

struct subset_functor {
    template <typename T>
    static constexpr bool is_supported()
    {
        return std::is_floating_point<T>::value;
    }

    template <typename T, std::enable_if_t< is_supported<T>() >* = nullptr>
    uint32_t operator()(const gdf_column& ids,
    	const gdf_column& in_x,const gdf_column& in_y, const gdf_column& in_id,const gdf_column& in_ts,
 	gdf_column& out_x, gdf_column& out_y, gdf_column& out_id,gdf_column& out_ts)                
    {

        //struct timeval t0,t1;
        //gettimeofday(&t0, nullptr);

	thrust::device_ptr<T> in_x_ptr = thrust::device_pointer_cast(static_cast<T*>(in_x.data));
        thrust::device_ptr<T> in_y_ptr = thrust::device_pointer_cast(static_cast<T*>(in_y.data));
        thrust::device_ptr<uint32_t> in_id_ptr =thrust::device_pointer_cast(static_cast<uint32_t*>(in_id.data));
        thrust::device_ptr<its_timestamp> in_ts_ptr = thrust::device_pointer_cast(static_cast<its_timestamp*>(in_ts.data));

#ifdef DEBUG
        int num_print = (in_id.size < 10) ? in_id.size : 10;
        std::cout<<"showing the first "<< num_print<<" input records before sort"<<std::endl;

        std::cout<<"input x"<<std::endl;
        thrust::copy(in_x_ptr,in_x_ptr+num_print,std::ostream_iterator<T>(std::cout, " "));std::cout<<std::endl;  
        std::cout<<"input y"<<std::endl;
        thrust::copy(in_y_ptr,in_y_ptr+num_print,std::ostream_iterator<T>(std::cout, " "));std::cout<<std::endl;  

        std::cout<<"input id"<<std::endl;
        thrust::copy(in_id_ptr,in_id_ptr+num_print,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;  
        std::cout<<"input timestamp"<<std::endl;
        thrust::copy(in_ts_ptr,in_ts_ptr+num_print,std::ostream_iterator<its_timestamp>(std::cout, " "));std::cout<<std::endl;    
#endif

	uint32_t num_id=ids.size;
	uint32_t num_rec=in_id.size;
	
	uint32_t *temp_id{nullptr};
        RMM_TRY( RMM_ALLOC((void**)&temp_id,num_id* sizeof(uint32_t),0) ) ; 
        thrust::device_ptr<uint32_t> temp_id_ptr=thrust::device_pointer_cast(temp_id);
        
 	thrust::device_ptr<uint32_t> ids_ptr=thrust::device_pointer_cast(static_cast<uint32_t*>(ids.data));
 	thrust::copy(ids_ptr, ids_ptr+num_id,temp_id_ptr);
        thrust::sort(temp_id_ptr,temp_id_ptr+num_id);
        thrust::copy(temp_id_ptr,temp_id_ptr+num_id,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
        
	thrust::device_vector<bool> hit_vec(num_rec);
	std::cout<<"beginning binary_search .............."<<std::endl;
	thrust::binary_search(temp_id_ptr, temp_id_ptr+num_id,in_id_ptr,in_id_ptr+num_rec,hit_vec.begin());
	std::cout<<"binary_search results.............."<<std::endl;
	thrust::copy(hit_vec.begin(),hit_vec.end(),std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
	thrust::device_vector<bool> temp_vec=hit_vec;
	thrust::copy(temp_vec.begin(),temp_vec.end(),std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
	
	uint32_t num_hit=thrust::count_if(temp_vec.begin(),temp_vec.end(),is_true());
	std::cout<<"num_hit="<<num_hit<<std::endl;
	
	RMM_TRY( RMM_ALLOC((void**)(&(out_x.data)),num_hit* sizeof(double),0) ) ; 
	RMM_TRY( RMM_ALLOC((void**)(&(out_y.data)),num_hit* sizeof(double),0) ) ;
	RMM_TRY( RMM_ALLOC((void**)(&(out_id.data)),num_hit* sizeof(uint32_t),0) ) ;
	RMM_TRY( RMM_ALLOC((void**)(&(out_ts.data)),num_hit* sizeof(its_timestamp),0) ) ;
	
	thrust::device_ptr<T> out_x_ptr = thrust::device_pointer_cast(static_cast<T*>(out_x.data));
        thrust::device_ptr<T> out_y_ptr = thrust::device_pointer_cast(static_cast<T*>(out_y.data));
        thrust::device_ptr<uint32_t> out_id_ptr =thrust::device_pointer_cast(static_cast<uint32_t*>(out_id.data));
        thrust::device_ptr<its_timestamp> out_ts_ptr = thrust::device_pointer_cast(static_cast<its_timestamp*>(out_ts.data));
	
	auto in_itr=thrust::make_zip_iterator(thrust::make_tuple(in_x_ptr, in_y_ptr, in_id_ptr,in_ts_ptr));
	auto out_itr=thrust::make_zip_iterator(thrust::make_tuple(out_x_ptr, out_y_ptr, out_id_ptr,out_ts_ptr));
	uint32_t num_keep=thrust::copy_if(in_itr,in_itr+num_rec,hit_vec.begin(),out_itr,is_true())-out_itr;
	CUDF_EXPECTS(num_hit==num_keep,"expecting count_if and copy_if return the same size");
	
	return(num_hit);
   }

    template <typename col_type, std::enable_if_t< !is_supported<col_type>() >* = nullptr>
    uint32_t operator()(const gdf_column& ids,
    	const gdf_column& in_x,const gdf_column& in_y, const gdf_column& in_id,const gdf_column& in_ts,
 	gdf_column& out_x,gdf_column& out_y, gdf_column& out_id,gdf_column& out_ts)                
    {
        CUDF_FAIL("Non-floating point operation is not supported");
    }
};
    

namespace cuspatial {

/*

*/
uint32_t subset_trajectory_id(const gdf_column& ids,
		const gdf_column& in_x, const gdf_column& in_y, const gdf_column& in_id, const gdf_column& in_ts,
		gdf_column& out_x, gdf_column& out_y,gdf_column& out_id, gdf_column& out_ts)
{       
    struct timeval t0,t1;
    gettimeofday(&t0, nullptr);
    
    CUDF_EXPECTS(in_x.data != nullptr && in_x.data != nullptr &&
                 in_id.data != nullptr && in_ts.data != nullptr,
                 "x/y/in_id/in_ts data cannot be null");
    CUDF_EXPECTS(in_x.size == in_y.size && in_x.size == in_id.size &&
                 in_x.size == in_ts.size ,
                 "x/y/in_id/timestamp must have equal size");
    
    // future versions might allow x/y/in_id/timestamp to have null_count > 0,
    // which might be useful for taking query results as inputs 
    CUDF_EXPECTS(in_x.null_count == 0 && in_y.null_count == 0 &&
                 in_id.null_count==0 && in_ts.null_count==0, 
                 "NULL support unimplemented");
    
    uint32_t num_hit=cudf::type_dispatcher(in_x.dtype, subset_functor(),ids,
                                                 in_x, in_y, in_id, in_ts,
                                                 out_x, out_y,out_id,out_ts);
    std::cout<<"number of resulting points:"<<num_hit<<std::endl;
    gettimeofday(&t1, nullptr);
    float subset_end2end_time=calc_time("trajectory subset end-to-end time in ms=",t0,t1);
    return num_hit;
}
  
}// namespace cuspatial
