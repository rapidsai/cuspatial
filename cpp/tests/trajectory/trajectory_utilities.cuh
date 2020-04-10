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

#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cuspatial/trajectory.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/timestamp_utilities.cuh>

#include "tests/utilities/cuspatial_gmock.hpp"

namespace cuspatial {
namespace test {

inline std::unique_ptr<cudf::experimental::table> make_test_trajectories_table(
    cudf::size_type size) {
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

  auto rand_double = cudf::test::UniformRandomGenerator<double>{};
  auto ids_iter = cudf::test::make_counting_transform_iterator(
      0, [&](auto i) { return ids[map[i]]; });
  auto doubles_iter = cudf::test::make_counting_transform_iterator(
      0, [&](auto i) { return 10000 * rand_double.generate(); });

  auto id = cudf::test::fixed_width_column_wrapper<int32_t>(ids_iter,
                                                            ids_iter + size);
  auto x = cudf::test::fixed_width_column_wrapper<double>(doubles_iter,
                                                          doubles_iter + size);
  auto y = cudf::test::fixed_width_column_wrapper<double>(doubles_iter,
                                                          doubles_iter + size);
  auto ts = cudf::test::generate_timestamps<cudf::timestamp_ms>(
      size,
      cudf::timestamp_ms{-2500000000000},  // Sat, 11 Oct 1890 19:33:20 GMT
      cudf::timestamp_ms{2500000000000}    // Mon, 22 Mar 2049 04:26:40 GMT
  );

  auto sorted = cudf::experimental::sort_by_key(
      cudf::table_view{{id, ts, x, y}}, cudf::table_view{{id, ts}});

  return sorted;
}

}  // namespace test
}  // namespace cuspatial
