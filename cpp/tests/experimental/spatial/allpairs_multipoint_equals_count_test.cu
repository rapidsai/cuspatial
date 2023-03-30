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
#include <cuspatial_test/base_fixture.hpp>
#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <rmm/device_vector.hpp>

#include <gtest/gtest.h>

#include <thrust/iterator/transform_iterator.h>

template <typename T>
struct AllPairsMultipointEqualsCount : public cuspatial::test::BaseFixture {
  using Vec = cuspatial::vec_2d<T>;

  void run_test(std::initializer_list<Vec> lhs, std::initializer_list<Vec> rhs)
  {
    auto _lhs = *make_multipoint_array(lhs);

    auto d_count = rmm::device_scalar<size_t>{};

    cuspatial::experimental::all_pairs_multipoint_equals_count(
      d_lhs.begin(), d_lhs.end(), d_rhs.begin(), d_rhs.end(), d_count.data());

    EXPECT_EQ(d_count.value(), lhs.size());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(h_expected, xy_output);
    EXPECT_EQ(h_expected.size(), std::distance(xy_output.begin(), xy_end));
  }
};

// float and double are logically the same but would require separate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(SinusoidalProjectionTest, TestTypes);

TYPED_TEST(SinusoidalProjectionTest, Empty)
{
  using T    = TypeParam;
  using Loc  = cuspatial::vec_2d<T>;
  using Cart = cuspatial::vec_2d<T>;

  auto origin = Loc{-90.66511046, 42.49197018};

  auto h_point_lonlat = std::vector<Loc>{};
  CUSPATIAL_RUN_TEST(this->run_test, h_point_lonlat, origin);
}

TYPED_TEST(SinusoidalProjectionTest, Single)
{
  using T    = TypeParam;
  using Loc  = cuspatial::vec_2d<T>;
  using Cart = cuspatial::vec_2d<T>;

  auto origin = Loc{-90.66511046, 42.49197018};

  auto h_point_lonlat = std::vector<Loc>({{-90.664973, 42.493894}});
  CUSPATIAL_RUN_TEST(this->run_test, h_point_lonlat, origin);
}

TYPED_TEST(SinusoidalProjectionTest, Extremes)
{
  using T    = TypeParam;
  using Loc  = cuspatial::vec_2d<T>;
  using Cart = cuspatial::vec_2d<T>;

  auto origin = Loc{0, 0};

  auto h_points_lonlat = std::vector<Loc>(
    {{0.0, -90.0}, {0.0, 90.0}, {-180.0, 0.0}, {180.0, 0.0}, {45.0, 0.0}, {-180.0, -90.0}});
  CUSPATIAL_RUN_TEST(this->run_test, h_points_lonlat, origin);
}

TYPED_TEST(SinusoidalProjectionTest, Multiple)
{
  using T    = TypeParam;
  using Loc  = cuspatial::vec_2d<T>;
  using Cart = cuspatial::vec_2d<T>;

  auto origin = Loc{-90.66511046, 42.49197018};

  auto h_points_lonlat = std::vector<Loc>({{-90.664973, 42.493894},
                                           {-90.665393, 42.491520},
                                           {-90.664976, 42.491420},
                                           {-90.664537, 42.493823}});
  CUSPATIAL_RUN_TEST(this->run_test, h_points_lonlat, origin);
}

TYPED_TEST(SinusoidalProjectionTest, OriginOutOfBounds)
{
  using T    = TypeParam;
  using Loc  = cuspatial::vec_2d<T>;
  using Cart = cuspatial::vec_2d<T>;

  auto origin = Loc{-181, -91};

  auto h_point_lonlat = std::vector<Loc>{};
  auto h_expected     = std::vector<Cart>{};

  auto point_lonlat = rmm::device_vector<Loc>{};
  auto expected     = rmm::device_vector<Cart>{};

  auto xy_output = rmm::device_vector<Cart>{};

  EXPECT_THROW(cuspatial::sinusoidal_projection(
                 point_lonlat.begin(), point_lonlat.end(), xy_output.begin(), origin),
               cuspatial::logic_error);
}

template <typename T>
struct identity_xform {
  using Location = cuspatial::vec_2d<T>;
  __device__ Location operator()(Location const& loc) { return loc; };
};

// This test verifies that fancy iterators can be passed by using a pass-through transform_iterator
TYPED_TEST(SinusoidalProjectionTest, TransformIterator)
{
  using T    = TypeParam;
  using Loc  = cuspatial::vec_2d<T>;
  using Cart = cuspatial::vec_2d<T>;

  auto origin = Loc{-90.66511046, 42.49197018};

  auto h_points_lonlat = std::vector<Loc>({{-90.664973, 42.493894},
                                           {-90.665393, 42.491520},
                                           {-90.664976, 42.491420},
                                           {-90.664537, 42.493823}});
  auto h_expected      = std::vector<Cart>(h_points_lonlat.size());

  std::transform(h_points_lonlat.begin(),
                 h_points_lonlat.end(),
                 h_expected.begin(),
                 sinusoidal_projection_functor(origin));

  auto points_lonlat = rmm::device_vector<Loc>{h_points_lonlat};
  auto expected      = rmm::device_vector<Cart>{h_expected};

  auto xy_output = rmm::device_vector<Cart>(4, Cart{-1, -1});

  auto xform_begin = thrust::make_transform_iterator(points_lonlat.begin(), identity_xform<T>{});
  auto xform_end   = thrust::make_transform_iterator(points_lonlat.end(), identity_xform<T>{});

  auto xy_end = cuspatial::sinusoidal_projection(xform_begin, xform_end, xy_output.begin(), origin);

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, xy_output);
  EXPECT_EQ(4, std::distance(xy_output.begin(), xy_end));
}
