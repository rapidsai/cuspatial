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
#include <thrust/iterator/discard_iterator.h>
#include <sys/time.h>
#include <time.h>

#include <cuspatial/shared_util.h>
#include <cuspatial/trajectory.hpp>
#include <include/trajectory_thrust.cuh>

using namespace std; 
using namespace cudf;
using namespace cuspatial;

struct derive_trajectories_functor {
    template <typename T>
    static constexpr bool is_supported()
    {
        return std::is_floating_point<T>::value;
    }

    template <typename T, std::enable_if_t< is_supported<T>() >* = nullptr>
    int operator()(gdf_column& x,gdf_column& y, gdf_column& object_id,
                   gdf_column& timestamp, gdf_column& trajectory_id,
                   gdf_column& length, gdf_column& offset)
    {
        T* x_ptr = static_cast<T*>(x.data);
        T* y_ptr = static_cast<T*>(y.data);
        uint32_t* id_ptr = static_cast<uint32_t*>(object_id.data);
        its_timestamp * time_ptr = static_cast<its_timestamp*>(timestamp.data);

#ifdef DEBUG
        int num_print = (object_id.size < 10) ? object_id.size : 10;
        std::cout<<"showing the first "<< num_print<<" input records before sort"<<std::endl;

        std::cout<<"x"<<std::endl;
        thrust::copy(x_ptr,x_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;  
        std::cout<<"y"<<std::endl;
        thrust::copy(y_ptr,y_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;  

        std::cout<<"oid"<<std::endl;
        thrust::copy(id_ptr,id_ptr+num_print,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;  
        std::cout<<"timestamp"<<std::endl;
        thrust::copy(time_ptr,time_ptr+num_print,std::ostream_iterator<its_timestamp>(std::cout, " "));std::cout<<std::endl;    
#endif

        struct timeval t0,t1;
        gettimeofday(&t0, nullptr);

        uint32_t num_rec = object_id.size;
        thrust::stable_sort_by_key(time_ptr, time_ptr + num_rec,
            thrust::make_zip_iterator(thrust::make_tuple(id_ptr, x_ptr, y_ptr)));
        thrust::stable_sort_by_key(id_ptr, id_ptr+num_rec,
            thrust::make_zip_iterator(thrust::make_tuple(time_ptr, x_ptr, y_ptr)));

        //allocate sufficient memory to hold id,cnt and pos before reduce_by_key
        uint32_t *objcnt = nullptr, *objpos = nullptr, *objid = nullptr;
        RMM_TRY( RMM_ALLOC((void**)&objcnt,num_rec* sizeof(uint32_t),0) ) ; 
        RMM_TRY( RMM_ALLOC((void**)&objpos,num_rec* sizeof(uint32_t),0) ) ; 
        RMM_TRY( RMM_ALLOC((void**)&objid,num_rec* sizeof(uint32_t),0) ) ; 
        
        int num_traj =
            thrust::reduce_by_key(thrust::device, id_ptr, id_ptr+num_rec,
                                  thrust::constant_iterator<int>(1),
                                  objid, objcnt).second - objcnt;

        //allocate just enough memory (num_traj), copy over and then free large (num_rec) arrays         
        uint32_t *trajid=nullptr,*trajcnt=nullptr,*trajpos=nullptr;
        RMM_TRY( RMM_ALLOC((void**)&trajid,  num_traj * sizeof(uint32_t),0) ) ; 
        RMM_TRY( RMM_ALLOC((void**)&trajcnt, num_traj * sizeof(uint32_t),0) ) ; 
        RMM_TRY( RMM_ALLOC((void**)&trajpos, num_traj * sizeof(uint32_t),0) ) ; 

        thrust::device_ptr<uint32_t> trajcnt_ptr=thrust::device_pointer_cast(trajcnt);
        thrust::device_ptr<uint32_t> trajpos_ptr=thrust::device_pointer_cast(trajpos);        

        thrust::copy(objid, objid + num_traj, trajid);
        thrust::copy(objcnt, objcnt + num_traj, trajcnt);
        thrust::copy(objpos, objpos + num_traj, trajpos);
        
        RMM_TRY( RMM_FREE(objid, 0) );
        RMM_TRY( RMM_FREE(objcnt, 0) );
        RMM_TRY( RMM_FREE(objpos, 0) );
        
        //to avoid lost memory problem when tid/cnt/pos gdf columns are associated with dvice memory
        gdf_column_view(&trajectory_id, trajid, nullptr, num_traj, GDF_INT32);
        gdf_column_view(&length, trajcnt, nullptr, num_traj, GDF_INT32);
        thrust::inclusive_scan(thrust::device, trajcnt_ptr, trajcnt_ptr+num_traj, trajpos_ptr);
        gdf_column_view(&offset, trajpos, nullptr, num_traj, GDF_INT32);

        gettimeofday(&t1, nullptr);
        float derive_trajectories_kernel_time=calc_time("coord_to_traj kernel time in ms=",t0,t1);

#ifdef DEBUG
        std::cout<<"#traj="<<num_traj<<std::endl;
        std::cout<<"showing the first "<< num_print<<" records aftr sort"<<std::endl;
        std::cout<<"x"<<std::endl;
        thrust::copy(x_ptr,x_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;  
        std::cout<<"y"<<std::endl;
        thrust::copy(y_ptr,y_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;  
    
        std::cout<<"oid"<<std::endl;
        thrust::copy(id_ptr,id_ptr+num_print,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;  
        std::cout<<"timestamp"<<std::endl;
        thrust::copy(time_ptr,time_ptr+num_print,std::ostream_iterator<its_timestamp>(std::cout, " "));std::cout<<std::endl;  
        
        num_print=(num_traj<10)?num_traj:10;
        std::cout<<"showing the first "<< num_print<<" trajectory records"<<std::endl;
        std::cout<<"trajectory id"<<std::endl;
        thrust::device_ptr<uint32_t> trajid_ptr=thrust::device_pointer_cast(trajid);
        thrust::copy(trajid_ptr,trajid_ptr+num_print,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
        std::cout<<"trajectory #of points"<<std::endl;
        thrust::copy(trajcnt_ptr,trajcnt_ptr+num_print,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
        std::cout<<"trajectory poisition index on sorted point x/y array"<<std::endl;
        thrust::copy(trajpos_ptr,trajpos_ptr+num_print,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;      
#endif
        return num_traj;
    }

    template <typename col_type, std::enable_if_t< !is_supported<col_type>() >* = nullptr>
    int operator()(gdf_column& x,gdf_column& y,gdf_column& oid, gdf_column& ts, 
 			    gdf_column& tid, gdf_column& len,gdf_column& pos)
    {
        CUDF_FAIL("Non-floating point operation is not supported");
    }
};
    

namespace cuspatial {

/*
 * Derive trajectories from points (x/y relative to an origin), timestamps and
 * object IDs by first sorting based on id and timestamp and then group by id.
 * see trajectory.hpp
*/
int derive_trajectories(gdf_column& x, gdf_column& y, gdf_column& object_id,
                        gdf_column& timestamp, gdf_column& trajectory_id,
                        gdf_column& length, gdf_column& offset)
{       
    struct timeval t0,t1;
    gettimeofday(&t0, nullptr);
    
    CUDF_EXPECTS(x.data != nullptr && y.data != nullptr &&
                 object_id.data != nullptr && timestamp.data != nullptr,
                 "x/y/object_id/timetamp data cannot be null");
    CUDF_EXPECTS(x.size == y.size && x.size == object_id.size &&
                 x.size == timestamp.size ,
                 "x/y/object_id/timestamp must have equal size");
    
    // future versions might allow x/y/object_id/timestamp to have null_count > 0,
    // which might be useful for taking query results as inputs 
    CUDF_EXPECTS(x.null_count == 0 && y.null_count == 0 &&
                 object_id.null_count==0 && timestamp.null_count==0, 
                 "NULL support unimplemented");
    
    int num_trajectories = cudf::type_dispatcher(x.dtype, derive_trajectories_functor(),
                                                 x, y, object_id, timestamp,
                                                 trajectory_id, length, offset);

    gettimeofday(&t1, nullptr);
    float coor2traj_end2end_time=calc_time("coord_to_traj end-to-end time in ms=",t0,t1);

    return num_trajectories;
}
  
}// namespace cuspatial
