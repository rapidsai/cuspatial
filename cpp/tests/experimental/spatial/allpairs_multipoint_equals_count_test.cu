/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cuspatial/constants.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/sinusoidal_projection.cuh>
#include <cuspatial/vec_2d.hpp>
#include <cuspatial_test/base_fixture.hpp>
#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <rmm/device_vector.hpp>

#include <gtest/gtest.h>

#include <thrust/iterator/transform_iterator.h>

template <typename T>
struct AllpairsMultipointEqualsCountTest : public ::testing::Test {
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }
  rmm::mr::device_memory_resource* mr() { return rmm::mr::get_current_device_resource(); }

  void run_single(std::initializer_list<std::initializer_list<vec_2d<T>>> lhs_coordinates,
                  std::initializer_list<std::initializer_list<vec_2d<T>>> rhs_coordinates,
                  std::initializer_list<T> expected)
  {
    std::vector<vec_d2<T>> lhs_ref(lhs_coordinates);
    std::vector<vec_d2<T>> rhs_ref(rhs_coordinates);
    // std::vector<vec_2d<T>> multipolygon_coordinates_vec(multipolygon_coordinates);
    return this->run_single(lhs_ref, rhs_ref, expected);
  }

  void run_single(std::initializer_list<std::initializer_list<vec_2d<T>>> lhs_coordinates,
                  std::initializer_list<std::initializer_list<vec_2d<T>>> rhs_coordinates,
                  std::initializer_list<size_t> expected)
  {
    auto d_lhs = make_multipoints_array(lhs_coordinates).ref;
    auto d_rhs = make_multipoints_array(rhs_coordinates).ref;
    auto got   = rmm::device_uvector<size_t>(d_lhs.size(), stream());

    auto ret = allpairs_multipoint_equals_count(d_lhs, d_rhs, got.begin(), stream());

    auto d_expected = cuspatial::test::make_device_vector(expected);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(got, d_expected);
    EXPECT_EQ(ret, got.end());
  }
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(AllpairsMultipointEqualsCountTest, TestTypes);

// Inputs are empty columns
TYPED_TEST(AllpairsMultipointEqualsCountTest, ZeroPairs)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     std::initializer_list<std::initializer_list<P>>{},
                     {0},
                     {0},
                     {0},
                     std::initializer_list<P>{},
                     std::initializer_list<T>{});
}
