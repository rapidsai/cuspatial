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

#include <vector>

#include <cuspatial/error.hpp>

#include <spatial/detail/cartesian_product_group_index_iterator.cuh>

#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <gtest/gtest.h>

using namespace cudf;
using namespace test;
using cuspatial::detail::cartesian_product_group;
using cuspatial::detail::cartesian_product_group_index;

void test_equality(int32_t idx,
                   cartesian_product_group_index lhs,
                   cartesian_product_group_index rhs)
{
  EXPECT_EQ(lhs.group_a.idx, rhs.group_a.idx) << "idx: " << idx;
  EXPECT_EQ(lhs.group_b.idx, rhs.group_b.idx) << "idx: " << idx;

  EXPECT_EQ(lhs.group_a.size, rhs.group_a.size) << "idx: " << idx;
  EXPECT_EQ(lhs.group_b.size, rhs.group_b.size) << "idx: " << idx;

  EXPECT_EQ(lhs.group_a.offset, rhs.group_a.offset) << "idx: " << idx;
  EXPECT_EQ(lhs.group_b.offset, rhs.group_b.offset) << "idx: " << idx;

  EXPECT_EQ(lhs.element_a_idx, rhs.element_a_idx) << "idx: " << idx;
  EXPECT_EQ(lhs.element_b_idx, rhs.element_b_idx) << "idx: " << idx;
}

template <typename T>
struct CartesianProductTest : public BaseFixture {
};

using TestTypes = Types<double>;

TYPED_TEST_CASE(CartesianProductTest, TestTypes);

TYPED_TEST(CartesianProductTest, Traversal)
{
  auto group_a_offsets_end = 6;
  auto group_a_offsets     = std::vector<int32_t>{0, 3, 4};
  auto group_b_offsets_end = 6;
  auto group_b_offsets     = std::vector<int32_t>{0, 2, 5};

  //     A    A    B    B    B    C
  //   +----+----+----+----+----+----+
  // D : 00   10 : 20   30   40 : 50 :
  //   +    +    +    +    +    +    +
  // D : 01   11 : 21   31   41 : 51 :
  //   +    +    +    +    +    +    +
  // D : 02   12 : 22   32   42 : 52 :
  //   +----+----+----+----+----+----+
  // E : 03   13 : 23   33   43 : 53 :
  //   +----+----+----+----+----+----+
  // F : 04   14 : 24   34   44 : 54 :
  //   +    +    +    +    +    +    +
  // F : 05   15 : 25   35   45 : 55 :
  //   +----+----+----+----+----+----+

  auto group_a_0 = cartesian_product_group{0, 3, 0};
  auto group_a_1 = cartesian_product_group{1, 1, 3};
  auto group_a_2 = cartesian_product_group{2, 2, 4};

  auto group_b_0 = cartesian_product_group{0, 2, 0};
  auto group_b_1 = cartesian_product_group{1, 3, 2};
  auto group_b_2 = cartesian_product_group{2, 1, 5};

  auto expected = std::vector<cartesian_product_group_index>{
    {group_a_0, group_b_0, 0, 0},  // 0
    {group_a_0, group_b_0, 1, 0},  //
    {group_a_0, group_b_0, 2, 0},  //
    {group_a_0, group_b_0, 0, 1},  //
    {group_a_0, group_b_0, 1, 1},  //
    {group_a_0, group_b_0, 2, 1},  //
    {group_a_0, group_b_1, 0, 0},  // 6
    {group_a_0, group_b_1, 1, 0},  //
    {group_a_0, group_b_1, 2, 0},  //
    {group_a_0, group_b_1, 0, 1},  //
    {group_a_0, group_b_1, 1, 1},  //
    {group_a_0, group_b_1, 2, 1},  //
    {group_a_0, group_b_1, 0, 2},  //
    {group_a_0, group_b_1, 1, 2},  //
    {group_a_0, group_b_1, 2, 2},  //
    {group_a_0, group_b_2, 0, 0},  // 15
    {group_a_0, group_b_2, 1, 0},  //
    {group_a_0, group_b_2, 2, 0},  //
    {group_a_1, group_b_0, 0, 0},  // 18
    {group_a_1, group_b_0, 0, 1},  //
    {group_a_1, group_b_1, 0, 0},  // 20
    {group_a_1, group_b_1, 0, 1},  //
    {group_a_1, group_b_1, 0, 2},  //
    {group_a_1, group_b_2, 0, 0},  // 23
    {group_a_2, group_b_0, 0, 0},  // 24
    {group_a_2, group_b_0, 1, 0},  //
    {group_a_2, group_b_0, 0, 1},  //
    {group_a_2, group_b_0, 1, 1},  //
    {group_a_2, group_b_1, 0, 0},  // 28
    {group_a_2, group_b_1, 1, 0},  //
    {group_a_2, group_b_1, 0, 1},  //
    {group_a_2, group_b_1, 1, 1},  //
    {group_a_2, group_b_1, 0, 2},  //
    {group_a_2, group_b_1, 1, 2},  //
    {group_a_2, group_b_2, 0, 0},  // 34
    {group_a_2, group_b_2, 1, 0},  //
  };

  auto gcp_iter =
    cuspatial::detail::make_cartesian_product_group_index_iterator(group_a_offsets_end,
                                                                   group_b_offsets_end,
                                                                   group_a_offsets.size(),
                                                                   group_b_offsets.size(),
                                                                   group_a_offsets.cbegin(),
                                                                   group_b_offsets.cbegin());

  auto num_cartesian = group_a_offsets_end * group_b_offsets_end;

  for (auto i = 0; i < num_cartesian; i++) { test_equality(i, expected[i], *(gcp_iter + i)); }
}
