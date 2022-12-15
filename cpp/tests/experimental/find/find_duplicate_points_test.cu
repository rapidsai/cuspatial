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

#include <cuspatial_test/base_fixture.hpp>
#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>

#include <cuspatial/experimental/detail/find/find_duplicate_points.cuh>
#include <cuspatial/vec_2d.hpp>

using namespace cuspatial;
using namespace cuspatial::detail;
using namespace cuspatial::test;

template <typename T>
struct FindDuplicatePointsTest : public BaseFixture {
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(FindDuplicatePointsTest, TestTypes);

TYPED_TEST(FindDuplicatePointsTest, simple)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto multipoints = make_multipoints_array({{P{0.0, 0.0}, P{1.0, 0.0}, P{0.0, 0.0}}});

  rmm::device_vector<uint8_t> flags(multipoints.range().num_points());
  std::vector<uint8_t> expected_flags{0, 0, 1};

  find_duplicate_points(multipoints.range(), flags.begin(), this->stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(flags, expected_flags);
}

TYPED_TEST(FindDuplicatePointsTest, empty)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto multipoints = make_multipoints_array<T>({});

  rmm::device_vector<uint8_t> flags(multipoints.range().num_points());
  std::vector<uint8_t> expected_flags{};

  find_duplicate_points(multipoints.range(), flags.begin(), this->stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(flags, expected_flags);
}

TYPED_TEST(FindDuplicatePointsTest, multi)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto multipoints = make_multipoints_array<T>(
    {{P{0.0, 0.0}, P{1.0, 0.0}, P{0.0, 0.0}, P{0.0, 0.0}, P{1.0, 0.0}, P{2.0, 0.0}},
     {P{5.0, 5.0}, P{5.0, 5.0}},
     {P{0.0, 0.0}}});

  rmm::device_vector<uint8_t> flags(multipoints.range().num_points());
  std::vector<uint8_t> expected_flags{0, 0, 1, 1, 1, 0, 0, 1, 0};

  find_duplicate_points(multipoints.range(), flags.begin(), this->stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(flags, expected_flags);
}
