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
#include <sys/time.h>
#include <time.h>

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
    uint32_t operator()(const gdf_column& id,
                        const gdf_column& in_x, const gdf_column& in_y,
                        const gdf_column& in_id, const gdf_column& in_timestamp,
                        gdf_column& out_x, gdf_column& out_y,
                        gdf_column& out_id, gdf_column& out_timestamp)
    {
        T* in_x_ptr = static_cast<T*>(in_x.data);
        T* in_y_ptr = static_cast<T*>(in_y.data);
        uint32_t* in_id_ptr = static_cast<uint32_t*>(in_id.data);
        cuspatial::its_timestamp* in_ts_ptr =
            static_cast<cuspatial::its_timestamp*>(in_timestamp.data);
        uint32_t* id_ptr = static_cast<uint32_t*>(id.data);

        cudaStream_t stream{0};
        auto exec_policy = rmm::exec_policy(stream)->on(stream);

        gdf_size_type num_id{id.size};
        gdf_size_type num_rec{in_id.size};

        rmm::device_vector<uint32_t> temp_id(id_ptr, id_ptr + num_id);     
        thrust::sort(exec_policy, temp_id.begin(), temp_id.end());            
        thrust::device_vector<bool> hit_vec(num_rec);
        thrust::binary_search(exec_policy, temp_id.cbegin(), temp_id.cend(),
                              in_id_ptr, in_id_ptr + num_rec, hit_vec.begin());


        uint32_t num_hit = thrust::count_if(exec_policy, hit_vec.begin(),
                                            hit_vec.end(), is_true());
        
        RMM_TRY( RMM_ALLOC(&out_x.data, num_hit * sizeof(double), 0) ); 
        RMM_TRY( RMM_ALLOC(&out_y.data, num_hit * sizeof(double), 0) );
        RMM_TRY( RMM_ALLOC(&out_id.data, num_hit * sizeof(uint32_t), 0) );
        RMM_TRY( RMM_ALLOC(&out_timestamp.data,
                           num_hit * sizeof(cuspatial::its_timestamp), 0) );

        T* out_x_ptr = static_cast<T*>(out_x.data);
        T* out_y_ptr = static_cast<T*>(out_y.data);
        uint32_t* out_id_ptr = static_cast<uint32_t*>(out_id.data);
        cuspatial::its_timestamp* out_ts_ptr = 
            static_cast<cuspatial::its_timestamp*>(out_timestamp.data);

        auto in_itr = thrust::make_zip_iterator(thrust::make_tuple(in_x_ptr,
                                                                   in_y_ptr,
                                                                   in_id_ptr,
                                                                   in_ts_ptr));
        auto out_itr = thrust::make_zip_iterator(thrust::make_tuple(out_x_ptr,
                                                                    out_y_ptr,
                                                                    out_id_ptr,
                                                                    out_ts_ptr));
        uint32_t num_keep = thrust::copy_if(exec_policy, in_itr, in_itr+num_rec,
                                            hit_vec.begin(), out_itr,
                                            is_true()) - out_itr;
        CUDF_EXPECTS(num_hit == num_keep,
                     "count_if and copy_if result mismatch");

        return num_hit;
    }

    template <typename T, std::enable_if_t< !is_supported<T>() >* = nullptr>
    uint32_t operator()(const gdf_column& ids,
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

uint32_t subset_trajectory_id(const gdf_column& id,
                              const gdf_column& in_x, const gdf_column& in_y,
                              const gdf_column& in_id,
                              const gdf_column& in_timestamp,
                              gdf_column& out_x, gdf_column& out_y,
                              gdf_column& out_id, gdf_column& out_timestamp)
{       
    struct timeval t0,t1;
    gettimeofday(&t0, nullptr);
    
    CUDF_EXPECTS(in_x.data != nullptr && in_x.data != nullptr &&
                 in_id.data != nullptr && in_timestamp.data != nullptr,
                 "x/y/in_id/in_ts data cannot be null");
    CUDF_EXPECTS(in_x.size == in_y.size && in_x.size == in_id.size &&
                 in_x.size == in_timestamp.size ,
                 "x/y/in_id/timestamp must have equal size");
    
    // future versions might allow x/y/in_id/timestamp to have null_count > 0,
    // which might be useful for taking query results as inputs 
    CUDF_EXPECTS(in_x.null_count == 0 && in_y.null_count == 0 &&
                 in_id.null_count==0 && in_timestamp.null_count==0, 
                 "NULL support unimplemented");
    
    uint32_t num_hit=cudf::type_dispatcher(in_x.dtype, subset_functor(), id,
                                           in_x, in_y, in_id, in_timestamp,
                                           out_x, out_y, out_id, out_timestamp);
    std::cout<<"number of resulting points:"<<num_hit<<std::endl;
    gettimeofday(&t1, nullptr);
    float subset_end2end_time = 
        cuspatial::calc_time("trajectory subset end-to-end time in ms=", t0, t1);
    return num_hit;
}
  
}// namespace cuspatial
