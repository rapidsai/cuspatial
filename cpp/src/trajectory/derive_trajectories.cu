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
#include <utilities/legacy/cuda_utils.hpp>
#include <rmm/thrust_rmm_allocator.h>

#include <utility/utility.hpp>
#include <utility/trajectory_thrust.cuh>
#include <cuspatial/trajectory.hpp>

#include <cudf/legacy/column.hpp>

namespace {

struct derive_trajectories_functor {
    template <typename T>
    static constexpr bool is_supported()
    {
        return std::is_floating_point<T>::value;
    }

    template <typename T, std::enable_if_t< is_supported<T>() >* = nullptr>
    gdf_size_type operator()(const gdf_column& x, const gdf_column& y,
                             const gdf_column& object_id,
                             const gdf_column& timestamp,
                             gdf_column& trajectory_id,
                             gdf_column& length,
                             gdf_column& offset)
    {
        cudaStream_t stream{0};
        auto exec_policy = rmm::exec_policy(stream);    

        T* x_ptr = static_cast<T*>(x.data);
        T* y_ptr = static_cast<T*>(y.data);
        int32_t* id_ptr = static_cast<int32_t*>(object_id.data);
        cudf::timestamp * time_ptr =
            static_cast<cudf::timestamp*>(timestamp.data);

        gdf_size_type num_rec = object_id.size;
        thrust::stable_sort_by_key(exec_policy->on(stream), time_ptr, time_ptr + num_rec,
            thrust::make_zip_iterator(thrust::make_tuple(id_ptr, x_ptr, y_ptr)));
        thrust::stable_sort_by_key(exec_policy->on(stream), id_ptr, id_ptr+num_rec,
            thrust::make_zip_iterator(thrust::make_tuple(time_ptr, x_ptr, y_ptr)));

        //allocate sufficient memory to hold id, cnt and pos before reduce_by_key
        rmm::device_vector<gdf_size_type> obj_count(num_rec);
        rmm::device_vector<gdf_size_type> obj_id(num_rec);

        auto end = thrust::reduce_by_key(exec_policy->on(stream), id_ptr, id_ptr + num_rec,
                                         thrust::constant_iterator<int>(1),
                                         obj_id.begin(),
                                         obj_count.begin());
        gdf_size_type num_traj = end.second - obj_count.begin();

        gdf_size_type* traj_id{nullptr};
        gdf_size_type* traj_count{nullptr};
        gdf_size_type* traj_pos{nullptr};
        RMM_TRY( RMM_ALLOC(&traj_id,  num_traj * sizeof(gdf_size_type), 0) );
        RMM_TRY( RMM_ALLOC(&traj_count, num_traj * sizeof(gdf_size_type), 0) );
        RMM_TRY( RMM_ALLOC(&traj_pos, num_traj * sizeof(gdf_size_type), 0) );

        thrust::copy_n(exec_policy->on(stream), obj_id.begin(), num_traj, traj_id);
        thrust::copy_n(exec_policy->on(stream), obj_count.begin(), num_traj, traj_count);
        thrust::inclusive_scan(exec_policy->on(stream), traj_count, traj_count +num_traj,
                               traj_pos);

        gdf_column_view(&trajectory_id, traj_id, nullptr, num_traj, GDF_INT32);
        gdf_column_view(&length, traj_count, nullptr, num_traj, GDF_INT32);
        gdf_column_view(&offset, traj_pos, nullptr, num_traj, GDF_INT32);

        return num_traj;
    }

    template <typename T, std::enable_if_t< !is_supported<T>() >* = nullptr>
    gdf_size_type operator()(const gdf_column& x, const gdf_column& y,
                             const gdf_column& object_id,
                             const gdf_column& timestamp,
                             gdf_column& trajectory_id,
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
gdf_size_type derive_trajectories(const gdf_column& x, const gdf_column& y,
                                  const gdf_column& object_id,
                                  const gdf_column& timestamp,
                                  gdf_column& trajectory_id,
                                  gdf_column& length,
                                  gdf_column& offset)
{
    CUDF_EXPECTS(x.data != nullptr && y.data != nullptr &&
                 object_id.data != nullptr && timestamp.data != nullptr,
                 "Null input data");
    CUDF_EXPECTS(x.size == y.size && x.size == object_id.size &&
                 x.size == timestamp.size ,
                 "Data size mismatch");
    CUDF_EXPECTS(object_id.dtype == GDF_INT32,
                 "Invalid trajectory ID datatype");
    CUDF_EXPECTS(timestamp.dtype == GDF_TIMESTAMP,
                 "Invalid timestamp datatype");
    CUDF_EXPECTS(x.null_count == 0 && y.null_count == 0 &&
                 object_id.null_count==0 && timestamp.null_count==0,
                 "NULL support unimplemented");

    gdf_size_type num_trajectories =
        cudf::type_dispatcher(x.dtype, derive_trajectories_functor(),
                              x, y, object_id, timestamp,
                              trajectory_id, length, offset);

    return num_trajectories;
}

}// namespace cuspatial
