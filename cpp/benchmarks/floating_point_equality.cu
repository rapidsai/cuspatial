/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <benchmarks/fixture/rmm_pool_raii.hpp>
#include <cuspatial_test/random.cuh>

#include <cuspatial/detail/utility/floating_point.cuh>
#include <cuspatial/error.hpp>

#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <nvbench/nvbench.cuh>

#include <thrust/tabulate.h>

#include <memory>
#include <type_traits>

using namespace cuspatial;

/**
 * @brief Helper to generate floats
 *
 * @p begin and @p end must be iterators to device-accessible memory
 *
 * @tparam FloatsIter The type of the iterator to the output floats container
 * @param begin The start of the sequence of floats to generate
 * @param end The end of the sequence of floats to generate
 */
template <class FloatsIter>
void generate_floats(FloatsIter begin, FloatsIter end)
{
  using T       = typename std::iterator_traits<FloatsIter>::value_type;
  auto engine_x = deterministic_engine(std::distance(begin, end));

  auto lo = std::numeric_limits<T>::min();
  auto hi = std::numeric_limits<T>::max();

  auto x_dist = make_uniform_dist(lo, hi);

  auto x_gen = value_generator{lo, hi, engine_x, x_dist};

  thrust::tabulate(
    rmm::exec_policy(), begin, end, [x_gen] __device__(size_t n) mutable { return x_gen(n); });
}

template <typename Float>
struct eq_comp {
  using element_t = Float;
  bool __device__ operator()(Float lhs, Float rhs)
  {
    // return lhs == rhs;
    return detail::float_equal(lhs, rhs);
  }
};

template <typename T>
void floating_point_equivalence_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cuspatial::rmm_pool_raii rmm_pool;

  int64_t const num_floats{state.get_int64("NumFloats")};
  rmm::device_vector<T> floats(num_floats);
  rmm::device_vector<bool> results(num_floats);

  generate_floats(floats.begin(), floats.end());

  CUSPATIAL_CUDA_TRY(cudaDeviceSynchronize());

  state.add_element_count(num_floats);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto stream = rmm::cuda_stream_view(launch.get_stream());
    thrust::transform(floats.begin(), floats.begin(), floats.end(), results.begin(), eq_comp<T>{});
  });
}

using floating_point_type = nvbench::type_list<float, double>;
NVBENCH_BENCH_TYPES(floating_point_equivalence_benchmark, NVBENCH_TYPE_AXES(floating_point_type))
  .set_type_axes_names({"FloatingPointType"})
  .add_int64_axis("NumFloats", {100'000, 1'000'000, 10'000'000, 100'000'000});
