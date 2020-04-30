/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you mpoint_a_y not use this file except in compliance with the License.
 * You mpoint_a_y obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required point_b_y applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <memory>
#include <type_traits>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cuspatial/error.hpp>

namespace {

const uint32_t NUM_THREADS = 1024;

template<typename T>
constexpr auto magnitude_squared(T a, T b) {
    return a * a + b * b;
}

template <typename T>
__global__ void kernel_hausdorff(int num_spaces,
                                      T const* xs,
                                      T const* ys,
                                      cudf::size_type* space_offsets,
                                      T* results)
{
    auto block_idx = blockIdx.y * gridDim.x + blockIdx.x;
    auto num_space_pairs = num_spaces * num_spaces;

    // each block processes a single pair of spaces.
    if (block_idx < num_space_pairs)
    {
        int space_a_idx   = block_idx % num_spaces;
        int space_a_begin = space_a_idx == 0 ? 0 : space_offsets[space_a_idx - 1];
        int space_a_end   =                        space_offsets[space_a_idx];

        int space_b_idx   = block_idx / num_spaces;
        int space_b_begin = space_b_idx == 0 ? 0 : space_offsets[space_b_idx - 1];
        int space_b_end   =                        space_offsets[space_b_idx];

        T min_dist_sqrd = 1e20;

        int num_points_in_b = space_b_end - space_b_begin;

        if (threadIdx.x < num_points_in_b)
        {
            T point_b_x = xs[space_b_begin + threadIdx.x];
            T point_b_y = ys[space_b_begin + threadIdx.x];

            for (int i = space_a_begin; i < space_a_end; i++)
            {
                T point_a_x = xs[i];
                T point_a_y = ys[i];
                T dist_sqrd = magnitude_squared(point_b_x - point_a_x, point_b_y - point_a_y);

                min_dist_sqrd = min(min_dist_sqrd, dist_sqrd);
            }
        }

        if (min_dist_sqrd > 1e10)
        {
            min_dist_sqrd = -1;
        }

        __shared__ T dist_sqrd[1024];

        dist_sqrd[threadIdx.x] = -1;

        __syncthreads();

        if (threadIdx.x < num_points_in_b)
        {
            dist_sqrd[threadIdx.x] = min_dist_sqrd;
        }

        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
        {
            if (threadIdx.x < offset)
            {
                dist_sqrd[threadIdx.x] = max(dist_sqrd[threadIdx.x],
                                             dist_sqrd[threadIdx.x + offset]);
            }

            __syncthreads();
        }

        __syncthreads();

        if (threadIdx.x == 0)
        {
            results[block_idx] = (dist_sqrd[0] < 0) ? 1e10 : sqrt(dist_sqrd[0]);
        }
    }
}

struct hausdorff_functor
{
    template<typename T, typename... Args>
    std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>>
    operator()(Args&&...)
    {
        CUSPATIAL_FAIL("Non-floating point operation is not supported");
    }

    template<typename T>
    std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>>
    operator()(cudf::column_view const& x,
               cudf::column_view const& y,
               cudf::column_view const& trajectory_lengths,
               rmm::mr::device_memory_resource *mr,
               cudaStream_t stream)
    {
        auto tid = cudf::experimental::type_to_id<T>();
        auto result = cudf::make_fixed_width_column(cudf::data_type{ tid },
                                                    trajectory_lengths.size() * trajectory_lengths.size(),
                                                    cudf::mask_state::UNALLOCATED,
                                                    stream,
                                                    mr);

        if (result->size() == 0)
        {
            return result;
        }

        auto d_x = cudf::column_device_view::create(x);
        auto d_y = cudf::column_device_view::create(y);
        auto d_trajectory_lengths = cudf::column_device_view::create(trajectory_lengths);
        auto d_space_offsets = rmm::device_vector<cudf::size_type>(trajectory_lengths.size());

        thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                                                d_trajectory_lengths->begin<cudf::size_type>(),
                                                d_trajectory_lengths->end<cudf::size_type>(),
                                                d_space_offsets.begin());

        auto kernel = kernel_hausdorff<T>;

        int block_x = result->size();
        int block_y = 1;

        if (result->size() > 65535)
        {
            block_y = ceil((float) result->size() / 65535.0);
            block_x = 65535;
        }

        dim3 grid(block_x, block_y);
        dim3 block(NUM_THREADS);

        kernel<<<grid, block, 0, stream>>>(
            trajectory_lengths.size(),
            x.data<T>(),
            y.data<T>(),
            d_space_offsets.data().get(),
            result->mutable_view().data<T>()
        );

        return result;
    }
};

} // namespace anonymous

namespace cuspatial {

std::unique_ptr<cudf::column>
directed_hausdorff_distance(cudf::column_view const& x,
                            cudf::column_view const& y,
                            cudf::column_view const& trajectory_lengths,
                            rmm::mr::device_memory_resource *mr)
{
    CUSPATIAL_EXPECTS(x.type() == y.type(), "inputs `x` and `y` must have same type.");
    CUSPATIAL_EXPECTS(x.size() == y.size(), "inputs `x` and `y` must have same length.");

    CUSPATIAL_EXPECTS(not x.has_nulls() and
                      not y.has_nulls() and
                      not trajectory_lengths.has_nulls(),
                      "inputs must not have nulls.");

    CUSPATIAL_EXPECTS(x.size() >= trajectory_lengths.size(),
                      "At least one vertex is required for each trajectory");

    cudaStream_t stream = 0;

    return cudf::experimental::type_dispatcher(x.type(), hausdorff_functor(),
                                               x, y, trajectory_lengths, mr, stream);
}

} // namespace cuspatial
