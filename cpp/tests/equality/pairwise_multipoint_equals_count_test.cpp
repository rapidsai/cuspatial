/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuspatial_test/column_factories.hpp>

#include <cuspatial/error.hpp>
#include <cuspatial/pairwise_multipoint_equals_count.hpp>

#include <cudf/utilities/default_stream.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <type_traits>

using namespace cuspatial;
using namespace cuspatial::test;

using namespace cudf::test;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

template <typename T>
struct PairwiseMultipointEqualsCountTestTyped : public BaseFixture {
  rmm::cuda_stream_view stream() { return cudf::get_default_stream(); }
};

struct PairwiseMultipointEqualsCountTestUntyped : public BaseFixture {
  rmm::cuda_stream_view stream() { return cudf::get_default_stream(); }
};

// float and double are logically the same but would require separate tests due to precision.
using TestTypes = Types<double>;
TYPED_TEST_CASE(PairwiseMultipointEqualsCountTestTyped, TestTypes);

TYPED_TEST(PairwiseMultipointEqualsCountTestTyped, Empty)
{
  using T           = TypeParam;
  auto [ptype, lhs] = make_point_column<T>(std::initializer_list<T>{}, this->stream());
  auto [pytpe, rhs] = make_point_column<T>(std::initializer_list<T>{}, this->stream());

  auto lhs_gcv = geometry_column_view(lhs->view(), ptype, geometry_type_id::POINT);
  auto rhs_gcv = geometry_column_view(rhs->view(), ptype, geometry_type_id::POINT);

  auto output = cuspatial::pairwise_multipoint_equals_count(lhs_gcv, rhs_gcv);

  auto expected = fixed_width_column_wrapper<uint32_t>({});

  expect_columns_equivalent(expected, output->view(), verbosity);
}

TYPED_TEST(PairwiseMultipointEqualsCountTestTyped, InvalidLength)
{
  using T           = TypeParam;
  auto [ptype, lhs] = make_point_column<T>({0, 1}, {0.0, 0.0}, this->stream());
  auto [pytpe, rhs] = make_point_column<T>({0, 1, 2}, {1.0, 1.0, 0.0, 0.0}, this->stream());

  auto lhs_gcv = geometry_column_view(lhs->view(), ptype, geometry_type_id::POINT);
  auto rhs_gcv = geometry_column_view(rhs->view(), ptype, geometry_type_id::POINT);

  EXPECT_THROW(auto output = cuspatial::pairwise_multipoint_equals_count(lhs_gcv, rhs_gcv),
               cuspatial::logic_error);
}

TEST_F(PairwiseMultipointEqualsCountTestUntyped, InvalidTypes)
{
  auto [ptype, lhs] = make_point_column<float>(std::initializer_list<float>{}, this->stream());
  auto [pytpe, rhs] = make_point_column<double>(std::initializer_list<double>{}, this->stream());

  auto lhs_gcv = geometry_column_view(lhs->view(), ptype, geometry_type_id::POINT);
  auto rhs_gcv = geometry_column_view(rhs->view(), ptype, geometry_type_id::POINT);

  EXPECT_THROW(auto output = cuspatial::pairwise_multipoint_equals_count(lhs_gcv, rhs_gcv),
               cuspatial::logic_error);
}
