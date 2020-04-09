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

using cudf::test::fixed_width_column_wrapper;

template <typename T>
struct ShiftTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(ShiftTest, cudf::test::FloatingPointTypes);
   
TYPED_TEST(ShiftTest, CoordinateTest)
{
    using T = TypeParam;
    auto camera_lon = -90.66511046;
    auto camera_lat =  42.49197018;
    auto point_lon = fixed_width_column_wrapper<T>({ -90.664973, -90.665393, -90.664976, -90.664537 });
    auto point_lat = fixed_width_column_wrapper<T>({  42.493894,  42.491520,  42.491420,  42.493823 });

    auto res_pair = cuspatial::lonlat_to_coord(camera_lon, camera_lat, point_lon, point_lat);

    auto expected_lon = fixed_width_column_wrapper<T>({ -0.01394348958210479, 0.02865987990558439, -0.01363917930576489, -0.05816999226454722 });
    auto expected_lat = fixed_width_column_wrapper<T>({ -0.21375777777718794, 0.05002000000015667,  0.06113111111163663, -0.20586888888847929 });

    cudf::test::expect_columns_equal(expected_lon, res_pair.first->view(), true);
    cudf::test::expect_columns_equal(expected_lat, res_pair.second->view(), true);
}
