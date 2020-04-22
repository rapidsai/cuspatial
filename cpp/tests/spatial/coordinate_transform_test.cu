/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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


#include <type_traits>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>
#include <cuspatial/coordinate_transform.hpp>
#include <cuspatial/error.hpp>

using cudf::test::fixed_width_column_wrapper;

template <typename T>
struct LonLatToCartesianTest : public cudf::test::BaseFixture {};

// float and double are logically the same but would require seperate tests due to precision.
TYPED_TEST_CASE(LonLatToCartesianTest, cudf::test::Types<double>);
   
TYPED_TEST(LonLatToCartesianTest, Single)
{
    using T = TypeParam;
    auto camera_lon = -90.66511046;
    auto camera_lat =  42.49197018;
    auto point_lon = fixed_width_column_wrapper<T>({ -90.664973 });
    auto point_lat = fixed_width_column_wrapper<T>({  42.493894 });

    auto res_pair = cuspatial::lonlat_to_cartesian(camera_lon, camera_lat, point_lon, point_lat);

    auto expected_lon = fixed_width_column_wrapper<T>({ -0.01126195531216838 });
    auto expected_lat = fixed_width_column_wrapper<T>({ -0.21375777777718794 });

    cudf::test::expect_columns_equal(expected_lon, res_pair.first->view(), true);
    cudf::test::expect_columns_equal(expected_lat, res_pair.second->view(), true);
}

TYPED_TEST(LonLatToCartesianTest, Extremes)
{
    using T = TypeParam;
    auto camera_lon = 0;
    auto camera_lat = 0;
    auto point_lon = fixed_width_column_wrapper<T>({   0.0,  0.0, -180.0, 180.0, 45.0, -180.0 });
    auto point_lat = fixed_width_column_wrapper<T>({ -90.0, 90.0,    0.0,   0.0,  0.0,  -90.0 });

    auto res_pair = cuspatial::lonlat_to_cartesian(camera_lon, camera_lat, point_lon, point_lat);

    auto expected_lon = fixed_width_column_wrapper<T>({     0.0,      0.0, 20000.0, -20000.0, -5000.0, 14142.13562373095192015 });
    auto expected_lat = fixed_width_column_wrapper<T>({ 10000.0, -10000.0,     0.0,      0.0,     0.0, 10000.0 });

    cudf::test::expect_columns_equal(expected_lon, res_pair.first->view(), true);
    cudf::test::expect_columns_equal(expected_lat, res_pair.second->view(), true);
}
   
TYPED_TEST(LonLatToCartesianTest, Multiple)
{
    using T = TypeParam;
    auto camera_lon = -90.66511046;
    auto camera_lat =  42.49197018;
    auto point_lon = fixed_width_column_wrapper<T>({ -90.664973, -90.665393, -90.664976, -90.664537 });
    auto point_lat = fixed_width_column_wrapper<T>({  42.493894,  42.491520,  42.491420,  42.493823 });

    auto res_pair = cuspatial::lonlat_to_cartesian(camera_lon, camera_lat, point_lon, point_lat);

                                                        
    auto expected_lon = fixed_width_column_wrapper<T>({ -0.01126195531216838, 0.02314864865181343, -0.01101638630252916, -0.04698301003584082 });
    auto expected_lat = fixed_width_column_wrapper<T>({ -0.21375777777718794, 0.05002000000015667,  0.06113111111163663, -0.20586888888847929 });

    cudf::test::expect_columns_equal(expected_lon, res_pair.first->view(), true);
    cudf::test::expect_columns_equal(expected_lat, res_pair.second->view(), true);
}

TYPED_TEST(LonLatToCartesianTest, Empty)
{
    using T = TypeParam;
    auto camera_lon = -90.66511046;
    auto camera_lat =  42.49197018;
    auto point_lon = fixed_width_column_wrapper<T>({ });
    auto point_lat = fixed_width_column_wrapper<T>({ });

    auto res_pair = cuspatial::lonlat_to_cartesian(camera_lon, camera_lat, point_lon, point_lat);

    auto expected_lon = fixed_width_column_wrapper<T>({ });
    auto expected_lat = fixed_width_column_wrapper<T>({ });

    cudf::test::expect_columns_equal(expected_lon, res_pair.first->view(), true);
    cudf::test::expect_columns_equal(expected_lat, res_pair.second->view(), true);
}
   
TYPED_TEST(LonLatToCartesianTest, NullableNoNulls)
{
    using T = TypeParam;
    auto camera_lon = -90.66511046;
    auto camera_lat =  42.49197018;
    auto point_lon = fixed_width_column_wrapper<T>({ -90.664973 }, { 1 });
    auto point_lat = fixed_width_column_wrapper<T>({  42.493894 }, { 1 });

    auto res_pair = cuspatial::lonlat_to_cartesian(camera_lon, camera_lat, point_lon, point_lat);

    auto expected_lon = fixed_width_column_wrapper<T>({ -0.01126195531216838 });
    auto expected_lat = fixed_width_column_wrapper<T>({ -0.21375777777718794 });

    cudf::test::expect_columns_equal(expected_lon, res_pair.first->view(), true);
    cudf::test::expect_columns_equal(expected_lat, res_pair.second->view(), true);
}
   
TYPED_TEST(LonLatToCartesianTest, NullabilityMixedNoNulls)
{
    using T = TypeParam;
    auto camera_lon = -90.66511046;
    auto camera_lat =  42.49197018;
    auto point_lon = fixed_width_column_wrapper<T>({ -90.664973 });
    auto point_lat = fixed_width_column_wrapper<T>({  42.493894 }, { 1 });

    auto res_pair = cuspatial::lonlat_to_cartesian(camera_lon, camera_lat, point_lon, point_lat);

    auto expected_lon = fixed_width_column_wrapper<T>({ -0.01126195531216838 });
    auto expected_lat = fixed_width_column_wrapper<T>({ -0.21375777777718794 });

    cudf::test::expect_columns_equal(expected_lon, res_pair.first->view(), true);
    cudf::test::expect_columns_equal(expected_lat, res_pair.second->view(), true);
}
   
TYPED_TEST(LonLatToCartesianTest, NullableWithNulls)
{
    using T = TypeParam;
    auto camera_lon = 0;
    auto camera_lat = 0;
    auto point_lon = fixed_width_column_wrapper<T>({ 0 }, { 0 });
    auto point_lat = fixed_width_column_wrapper<T>({ 0 }, { 1 });

    EXPECT_THROW(cuspatial::lonlat_to_cartesian(camera_lon, camera_lat, point_lon, point_lat),
                 cuspatial::logic_error);
}

TYPED_TEST(LonLatToCartesianTest, OriginOutOfBounds)
{
    using T = TypeParam;
    auto camera_lon = -181;
    auto camera_lat =  -91;
    auto point_lon = fixed_width_column_wrapper<T>({ 0 });
    auto point_lat = fixed_width_column_wrapper<T>({ 0 });

    EXPECT_THROW(cuspatial::lonlat_to_cartesian(camera_lon, camera_lat, point_lon, point_lat),
                 cuspatial::logic_error);
}
   
TYPED_TEST(LonLatToCartesianTest, MismatchType)
{
    auto camera_lon = 0;
    auto camera_lat = 0;
    auto point_lon = fixed_width_column_wrapper<double>({ 0 });
    auto point_lat = fixed_width_column_wrapper<float>({ 0 });

    EXPECT_THROW(cuspatial::lonlat_to_cartesian(camera_lon, camera_lat, point_lon, point_lat),
                 cuspatial::logic_error);
}
   
TYPED_TEST(LonLatToCartesianTest, MismatchSize)
{
    using T = TypeParam;
    auto camera_lon = 0;
    auto camera_lat = 0;
    auto point_lon = fixed_width_column_wrapper<T>({ 0, 0 });
    auto point_lat = fixed_width_column_wrapper<T>({ 0 });

    EXPECT_THROW(cuspatial::lonlat_to_cartesian(camera_lon, camera_lat, point_lon, point_lat),
                 cuspatial::logic_error);
}
