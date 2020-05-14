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

#include <cuspatial/error.hpp>
#include <cuspatial/soa_readers.hpp>

#include <utility/legacy/utility.hpp>

#include <rmm/mr/device/default_memory_resource.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

#include <initializer_list>
#include <utility>

using namespace cudf::test;

template <typename T>
struct SOAReaderTest : public BaseFixture {
  template <typename Reader>
  void run_soa_io_test(Reader read, std::string const filename, std::initializer_list<T> testdata)
  {
    TempDirTestEnvironment* const temp_env = static_cast<TempDirTestEnvironment*>(
      ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment));

    temp_env->SetUp();

    auto path     = temp_env->get_temp_filepath(filename);
    auto expected = fixed_width_column_wrapper<T>(testdata);
    auto h_col    = to_host<T>(expected).first;

    CUSPATIAL_EXPECTS(h_col.size() == cuspatial::detail::write_field_from_vec(
                                        path.data(), std::vector<T>(h_col.begin(), h_col.end())),
                      "Wrote an empty vec");

    auto actual = read(path, this->mr());
    expect_columns_equal(*actual, expected, true);

    temp_env->TearDown();
  }
};

template <typename T>
struct ITSTimestamp64Test : public SOAReaderTest<T> {
};

using TestTypesInt64 = Types<int64_t>;
TYPED_TEST_CASE(ITSTimestamp64Test, TestTypesInt64);

template <typename T>
struct INTTest : public SOAReaderTest<T> {
};

using TestTypesInt32 = Types<int32_t>;
TYPED_TEST_CASE(INTTest, TestTypesInt32);

template <typename T>
struct POLYGONSOATest : public SOAReaderTest<T> {
};

using TestTypesPolygonSoa = Types<double>;
TYPED_TEST_CASE(POLYGONSOATest, TestTypesPolygonSoa);

TYPED_TEST(ITSTimestamp64Test, Empty)
{
  this->run_soa_io_test(cuspatial::experimental::read_timestamp_soa, "soa_its.tmp", {});
}

TYPED_TEST(ITSTimestamp64Test, Single)
{
  this->run_soa_io_test(cuspatial::experimental::read_timestamp_soa, "soa_its.tmp", {0});
}

TYPED_TEST(ITSTimestamp64Test, Triple)
{
  this->run_soa_io_test(cuspatial::experimental::read_timestamp_soa, "soa_its.tmp", {0, 1, 2});
}

TYPED_TEST(ITSTimestamp64Test, Negative)
{
  this->run_soa_io_test(cuspatial::experimental::read_timestamp_soa, "soa_its.tmp", {0, -1, -2});
}

TYPED_TEST(INTTest, EmptyUint32)
{
  this->run_soa_io_test(cuspatial::experimental::read_int32_soa, "soa_int32.tmp", {});
}

TYPED_TEST(INTTest, SingleUint32)
{
  this->run_soa_io_test(cuspatial::experimental::read_int32_soa, "soa_int32.tmp", {0});
}

TYPED_TEST(INTTest, TripleUint32)
{
  this->run_soa_io_test(cuspatial::experimental::read_int32_soa, "soa_int32.tmp", {0, 1, 2});
}

TYPED_TEST(INTTest, NegativeUint32)
{
  this->run_soa_io_test(cuspatial::experimental::read_int32_soa, "soa_int32.tmp", {0, -1, -2});
}

// TODO:
// Test read_points_lonlat
// Test read_points_xy

TYPED_TEST(POLYGONSOATest, PolygonSoaTest)
{
  using T = TypeParam;

  TempDirTestEnvironment* const temp_env = static_cast<TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment));
  temp_env->SetUp();

  uint32_t length[]{4};
  T points[]{1.0, 2.0, 3.0, 4.0};

  // create a polygon
  struct cuspatial::detail::polygons<T> polygon;
  polygon.num_group      = 1;
  polygon.num_feature    = 1;
  polygon.num_ring       = 1;
  polygon.num_vertex     = 1;
  polygon.group_length   = length;
  polygon.feature_length = length;
  polygon.ring_length    = length;
  polygon.x              = points;
  polygon.y              = points;

  // write polygon
  cuspatial::detail::write_polygon_soa<T>(temp_env->get_temp_filepath("soa_polygons.tmp"),
                                          &polygon);

  // read polygon
  struct cuspatial::detail::polygons<T> new_poly;
  cuspatial::detail::read_polygon_soa<T>(temp_env->get_temp_filepath("soa_polygons.tmp").c_str(),
                                         &new_poly);

  // validate read polygon is same as written polygon
  CUSPATIAL_EXPECTS(polygon.num_group == new_poly.num_group, "Number of groups inequal");
}
