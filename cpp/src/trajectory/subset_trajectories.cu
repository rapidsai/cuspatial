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
#include <cudf/legacy/copying.hpp>
#include <utilities/legacy/cuda_utils.hpp>
#include <type_traits>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <sys/time.h>
#include <time.h>

#include <rmm/thrust_rmm_allocator.h>

#include <utility/utility.hpp>
#include <utility/trajectory_thrust.cuh>
#include <cuspatial/trajectory.hpp>

namespace{

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
    gdf_size_type operator()(const gdf_column& id,
                             const gdf_column& in_x, const gdf_column& in_y,
                             const gdf_column& in_id,
                             const gdf_column& in_timestamp,
                             gdf_column& out_x, gdf_column& out_y,
                             gdf_column& out_id, gdf_column& out_timestamp)
    {

        cudaStream_t stream{0};
        auto exec_policy = rmm::exec_policy(stream);    

        gdf_size_type num_hit{0};
        gdf_size_type num_id{id.size};
        gdf_size_type num_rec{in_id.size};

        if (num_id > 0 && id.data != nullptr && num_rec > 0) {
            int32_t* in_id_ptr = static_cast<int32_t*>(in_id.data);
            int32_t* id_ptr = static_cast<int32_t*>(id.data);

            rmm::device_vector<int32_t> temp_id(id_ptr, id_ptr + num_id);
            thrust::sort(exec_policy->on(stream), temp_id.begin(), temp_id.end());
            thrust::device_vector<bool> hit_vec(num_rec);
            thrust::binary_search(exec_policy->on(stream), temp_id.cbegin(), temp_id.cend(),
                                in_id_ptr, in_id_ptr + num_rec, hit_vec.begin());

            num_hit = thrust::count(exec_policy->on(stream), hit_vec.begin(),
                                    hit_vec.end(), true);

            if (num_hit > 0) {
                out_x = cudf::allocate_like(in_x, num_hit);
                out_y = cudf::allocate_like(in_y, num_hit);
                out_id = cudf::allocate_like(in_id, num_hit);
                out_timestamp = cudf::allocate_like(in_timestamp, num_hit);

                auto in_itr = thrust::make_zip_iterator(thrust::make_tuple(
                    static_cast<T*>(in_x.data), static_cast<T*>(in_y.data),
                    static_cast<int32_t*>(in_id.data),
                    static_cast<cudf::timestamp*>(in_timestamp.data)));
                auto out_itr = thrust::make_zip_iterator(thrust::make_tuple(
                    static_cast<T*>(out_x.data), static_cast<T*>(out_y.data),
                    static_cast<int32_t*>(out_id.data),
                    static_cast<cudf::timestamp*>(out_timestamp.data)));

                auto end = thrust::copy_if(exec_policy->on(stream), in_itr, in_itr + num_rec,
                                           hit_vec.begin(), out_itr, is_true());
                gdf_size_type num_keep = end - out_itr;

                CUDF_EXPECTS(num_hit == num_keep,
                            "count_if and copy_if result mismatch");
            }
        }

        return num_hit;
    }

    template <typename T, std::enable_if_t< !is_supported<T>() >* = nullptr>
    gdf_size_type operator()(const gdf_column& ids,
                             const gdf_column& in_x, const gdf_column& in_y,
                             const gdf_column& in_id, const gdf_column& in_ts,
                             gdf_column& out_x, gdf_column& out_y,
                             gdf_column& out_id, gdf_column& out_ts)
    {
        CUDF_FAIL("Non-floating point operation is not supported");
    }
};

} // namespace anonymous

namespace cuspatial {

gdf_size_type subset_trajectory_id(const gdf_column& id,
                                   const gdf_column& in_x,
                                   const gdf_column& in_y,
                                   const gdf_column& in_id,
                                   const gdf_column& in_timestamp,
                                   gdf_column& out_x,
                                   gdf_column& out_y,
                                   gdf_column& out_id,
                                   gdf_column& out_timestamp)
{
    CUDF_EXPECTS(in_x.data != nullptr && in_x.data != nullptr &&
                 in_id.data != nullptr && in_timestamp.data != nullptr,
                 "Null input data");
    CUDF_EXPECTS(in_x.size == in_y.size && in_x.size == in_id.size &&
                 in_x.size == in_timestamp.size,
                 "Data size mismatch");
    CUDF_EXPECTS(in_id.dtype == GDF_INT32,
                 "Invalid trajectory ID datatype");
    CUDF_EXPECTS(id.dtype == in_id.dtype,
                 "Trajectory ID datatype mismatch");
    CUDF_EXPECTS(in_timestamp.dtype == GDF_TIMESTAMP,
                 "Invalid timestamp datatype");
    CUDF_EXPECTS(in_x.null_count == 0 && in_y.null_count == 0 &&
                 in_id.null_count==0 && in_timestamp.null_count==0,
                 "NULL support unimplemented");

    out_x = cudf::empty_like(in_x);
    out_y = cudf::empty_like(in_y);
    out_id = cudf::empty_like(in_id);
    out_timestamp = cudf::empty_like(in_timestamp);

    return cudf::type_dispatcher(in_x.dtype, subset_functor(), id,
                                 in_x, in_y, in_id, in_timestamp,
                                 out_x, out_y, out_id, out_timestamp);
}

}// namespace cuspatial
