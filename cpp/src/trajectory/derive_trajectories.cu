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

#include <type_traits>

#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <utilities/cuda_utils.hpp>
#include <rmm/thrust_rmm_allocator.h>

#include <utility/utility.hpp>
#include <utility/trajectory_thrust.cuh>
#include <cuspatial/trajectory.hpp>

namespace {

struct derive_trajectories_functor {
    template <typename T>
    static constexpr bool is_supported()
    {
        return std::is_floating_point<T>::value;
    }

    template <typename T, std::enable_if_t< is_supported<T>() >* = nullptr>
    int operator()(gdf_column& x, gdf_column& y, gdf_column& object_id,
                   gdf_column& timestamp, gdf_column& trajectory_id,
                   gdf_column& length, gdf_column& offset)
    {
        T* x_ptr = static_cast<T*>(x.data);
        T* y_ptr = static_cast<T*>(y.data);
        uint32_t* id_ptr = static_cast<uint32_t*>(object_id.data);
        cuspatial::its_timestamp * time_ptr =
            static_cast<cuspatial::its_timestamp*>(timestamp.data);

        cudaStream_t stream{0};
        auto exec_policy = rmm::exec_policy(stream)->on(stream);

        uint32_t num_rec = object_id.size;
        thrust::stable_sort_by_key(exec_policy, time_ptr, time_ptr + num_rec,
            thrust::make_zip_iterator(thrust::make_tuple(id_ptr, x_ptr, y_ptr)));
        thrust::stable_sort_by_key(exec_policy, id_ptr, id_ptr+num_rec,
            thrust::make_zip_iterator(thrust::make_tuple(time_ptr, x_ptr, y_ptr)));

        //allocate sufficient memory to hold id, cnt and pos before reduce_by_key
        uint32_t *objcnt{nullptr};
        uint32_t *objid{nullptr};
        RMM_TRY( RMM_ALLOC(&objcnt, num_rec * sizeof(uint32_t), 0) );
        RMM_TRY( RMM_ALLOC(&objid, num_rec * sizeof(uint32_t), 0) );
        
        int num_traj =
            thrust::reduce_by_key(exec_policy, id_ptr, id_ptr + num_rec,
                                  thrust::constant_iterator<int>(1),
                                  objid, objcnt).second - objcnt;

        //allocate just enough memory (num_traj), copy over and then free large (num_rec) arrays         
        uint32_t *trajid{nullptr};
        uint32_t *trajcnt{nullptr};
        uint32_t *trajpos{nullptr};
        RMM_TRY( RMM_ALLOC(&trajid,  num_traj * sizeof(uint32_t), 0) );
        RMM_TRY( RMM_ALLOC(&trajcnt, num_traj * sizeof(uint32_t), 0) );
        RMM_TRY( RMM_ALLOC(&trajpos, num_traj * sizeof(uint32_t), 0) );

        thrust::copy(exec_policy, objid, objid + num_traj, trajid);
        thrust::copy(exec_policy, objcnt, objcnt + num_traj, trajcnt);
        thrust::inclusive_scan(exec_policy, trajcnt, trajcnt + num_traj, trajpos);

        RMM_TRY( RMM_FREE(objid, 0) );
        RMM_TRY( RMM_FREE(objcnt, 0) );

        gdf_column_view(&trajectory_id, trajid, nullptr, num_traj, GDF_INT32);
        gdf_column_view(&length, trajcnt, nullptr, num_traj, GDF_INT32);
        gdf_column_view(&offset, trajpos, nullptr, num_traj, GDF_INT32);

        return num_traj;
    }

    template <typename T, std::enable_if_t< !is_supported<T>() >* = nullptr>
    int operator()(gdf_column& x, gdf_column& y, gdf_column& object_id,
                   gdf_column& timestamp, gdf_column& trajectory_id,
                   gdf_column& length, gdf_column& offset)
    {
        CUDF_FAIL("Non-floating point operation is not supported");
    }
};

} // namespace anonymous


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
    
    int num_trajectories = cudf::type_dispatcher(x.dtype, 
                                                 derive_trajectories_functor(),
                                                 x, y, object_id, timestamp,
                                                 trajectory_id, length, offset);

    return num_trajectories;
}
  
}// namespace cuspatial
