/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cuspatial/trajectory.hpp>

#include <cudf/detail/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/timestamp_utilities.cuh>

namespace cuspatial {
namespace test {

template <typename T>
std::unique_ptr<cudf::table> make_test_trajectories_table(
  cudf::size_type size,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  std::vector<int32_t> ids(size);
  std::vector<int32_t> map(size);
  std::iota(map.begin(), map.end(), 0);
  // three sorted trajectories: one with 2/3 of the points, two with 1/6
  std::transform(map.cbegin(), map.cend(), ids.begin(), [&size](int32_t i) {
    return (i < 2 * size / 3) ? 0 : (i < 5 * size / 6) ? 1 : 2;
  });

  // randomize sequence
  std::seed_seq seed{0};
  std::shuffle(map.begin(), map.end(), std::mt19937{seed});

  auto rand_float = cudf::test::UniformRandomGenerator<T>{};
  auto ids_iter =
    cudf::test::make_counting_transform_iterator(0, [&](auto i) { return ids[map[i]]; });
  auto floats_iter = cudf::test::make_counting_transform_iterator(0, [&](auto i) {
    return static_cast<T>(40000 * rand_float.generate() * (rand_float.generate() > 0.5 ? 1 : -1));
  });

  using duration_ms = typename cudf::timestamp_ms::duration;

  auto id = cudf::test::fixed_width_column_wrapper<int32_t>(ids_iter, ids_iter + size);
  auto x  = cudf::test::fixed_width_column_wrapper<T>(floats_iter, floats_iter + size);
  auto y  = cudf::test::fixed_width_column_wrapper<T>(floats_iter, floats_iter + size);
  auto ts = cudf::test::generate_timestamps<cudf::timestamp_ms>(
    size,
    cudf::timestamp_ms{duration_ms{-2500000000000}},  // Sat, 11 Oct 1890 19:33:20 GMT
    cudf::timestamp_ms{duration_ms{2500000000000}}    // Mon, 22 Mar 2049 04:26:40 GMT
  );

  auto sorted = cudf::detail::sort_by_key(
    cudf::table_view{{id, x, y, ts}}, cudf::table_view{{id, ts}}, {}, {}, 0, mr);

  return sorted;
}

}  // namespace test
}  // namespace cuspatial
