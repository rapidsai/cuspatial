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

#include <cuspatial/error.hpp>
#include <cuspatial/experimental/linestring_distance.cuh>
#include <cuspatial/utility/vec_2d.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

namespace cuspatial {
namespace test {

using namespace cudf;
using namespace cudf::test;

template <typename T>
struct PairwiseLinestringDistanceTest : public BaseFixture {
};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(PairwiseLinestringDistanceTest, TestTypes);

TYPED_TEST(PairwiseLinestringDistanceTest, FromSeparateArrayInputs)
{
  using T       = TypeParam;
  using CartVec = std::vector<cartesian_2d<T>>;

  auto a_cart2d = rmm::device_vector<cartesian_2d<T>>{
    CartVec({{0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}})};
  auto b_cart2d = rmm::device_vector<cartesian_2d<T>>{
    CartVec({{0.0f, 1.0f}, {1.0f, 1.0f}, {2.0f, 1.0f}, {3.0f, 1.0f}, {4.0f, 1.0f}})};
  auto offset = rmm::device_vector<int32_t>{0};

  auto distance = rmm::device_vector<T>{std::vector<T>{0.0}};
  auto expected = rmm::device_vector<T>{std::vector<T>{1.0}};

  pairwise_linestring_distance(offset.begin(),
                               offset.end(),
                               a_cart2d.begin(),
                               a_cart2d.end(),
                               offset.begin(),
                               b_cart2d.begin(),
                               b_cart2d.end(),
                               distance.begin());

  EXPECT_EQ(distance, expected);
}

}  // namespace test
}  // namespace cuspatial
