/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cuspatial_test/column_factories.hpp>
#include <cuspatial_test/geometry_fixtures.hpp>

#include <cuspatial/column/mixed_geometry_column_view.hpp>
#include <cuspatial/types.hpp>

#include <cudf_test/column_wrapper.hpp>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename T>
struct mixed_geometry_column_test : CommonGeometryColumnFixture<T> {};

TYPED_TEST_CASE(mixed_geometry_column_test, FloatingPointTypes);

TYPED_TEST(mixed_geometry_column_test, ConstructorLegalEmptyColumn)
{
  auto types_buffer   = cudf::test::fixed_width_column_wrapper<mixed_geometry_type_t>{};
  auto offsets_buffer = cudf::test::fixed_width_column_wrapper<cudf::size_type>{};

  auto mixed_geometry_column = mixed_geometry_column_view(types_buffer,
                                                          offsets_buffer,
                                                          this->empty_point(),
                                                          this->empty_linestring(),
                                                          this->empty_polygon(),
                                                          this->empty_multipoint(),
                                                          this->empty_multilinestring(),
                                                          this->empty_multipolygon());
}

TYPED_TEST(mixed_geometry_column_test, ConstructorIllegalWrongTypesBufferTypes)
{
  auto types_buffer   = cudf::test::fixed_width_column_wrapper<int32_t>{};
  auto offsets_buffer = cudf::test::fixed_width_column_wrapper<cudf::size_type>{};

  EXPECT_THROW(mixed_geometry_column_view(types_buffer,
                                          offsets_buffer,
                                          this->empty_point(),
                                          this->empty_linestring(),
                                          this->empty_polygon(),
                                          this->empty_multipoint(),
                                          this->empty_multilinestring(),
                                          this->empty_multipolygon()),
               cuspatial::logic_error);
}

TYPED_TEST(mixed_geometry_column_test, ConstructorIllegalWrongOffsetsBufferTypes)
{
  auto types_buffer   = cudf::test::fixed_width_column_wrapper<mixed_geometry_type_t>{};
  auto offsets_buffer = cudf::test::fixed_width_column_wrapper<int64_t>{};

  EXPECT_THROW(mixed_geometry_column_view(types_buffer,
                                          offsets_buffer,
                                          this->empty_point(),
                                          this->empty_linestring(),
                                          this->empty_polygon(),
                                          this->empty_multipoint(),
                                          this->empty_multilinestring(),
                                          this->empty_multipolygon()),
               cuspatial::logic_error);
}

TYPED_TEST(mixed_geometry_column_test, ConstructorIllegalWrongPointColumnType)
{
  auto types_buffer   = cudf::test::fixed_width_column_wrapper<mixed_geometry_type_t>{};
  auto offsets_buffer = cudf::test::fixed_width_column_wrapper<int64_t>{};

  EXPECT_THROW(mixed_geometry_column_view(types_buffer,
                                          offsets_buffer,
                                          this->empty_linestring(),
                                          this->empty_linestring(),
                                          this->empty_polygon(),
                                          this->empty_multipoint(),
                                          this->empty_multilinestring(),
                                          this->empty_multipolygon()),
               cuspatial::logic_error);
}

TYPED_TEST(mixed_geometry_column_test, ConstructorIllegalWrongLinestringColumnType)
{
  auto types_buffer   = cudf::test::fixed_width_column_wrapper<mixed_geometry_type_t>{};
  auto offsets_buffer = cudf::test::fixed_width_column_wrapper<int64_t>{};

  EXPECT_THROW(mixed_geometry_column_view(types_buffer,
                                          offsets_buffer,
                                          this->empty_point(),
                                          this->empty_polygon(),
                                          this->empty_polygon(),
                                          this->empty_multipoint(),
                                          this->empty_multilinestring(),
                                          this->empty_multipolygon()),
               cuspatial::logic_error);
}

TYPED_TEST(mixed_geometry_column_test, ConstructorIllegalWrongPolygonColumnType)
{
  auto types_buffer   = cudf::test::fixed_width_column_wrapper<mixed_geometry_type_t>{};
  auto offsets_buffer = cudf::test::fixed_width_column_wrapper<int64_t>{};

  EXPECT_THROW(mixed_geometry_column_view(types_buffer,
                                          offsets_buffer,
                                          this->empty_point(),
                                          this->empty_linestring(),
                                          this->empty_point(),
                                          this->empty_multipoint(),
                                          this->empty_multilinestring(),
                                          this->empty_multipolygon()),
               cuspatial::logic_error);
}

TYPED_TEST(mixed_geometry_column_test, ConstructorIllegalWrongMultipointColumnType)
{
  auto types_buffer   = cudf::test::fixed_width_column_wrapper<mixed_geometry_type_t>{};
  auto offsets_buffer = cudf::test::fixed_width_column_wrapper<int64_t>{};

  EXPECT_THROW(mixed_geometry_column_view(types_buffer,
                                          offsets_buffer,
                                          this->empty_point(),
                                          this->empty_linestring(),
                                          this->empty_polygon(),
                                          this->empty_multipolygon(),
                                          this->empty_multilinestring(),
                                          this->empty_multipolygon()),
               cuspatial::logic_error);
}

TYPED_TEST(mixed_geometry_column_test, ConstructorIllegalWrongMultilinestringColumnType)
{
  auto types_buffer   = cudf::test::fixed_width_column_wrapper<mixed_geometry_type_t>{};
  auto offsets_buffer = cudf::test::fixed_width_column_wrapper<int64_t>{};

  EXPECT_THROW(mixed_geometry_column_view(types_buffer,
                                          offsets_buffer,
                                          this->empty_point(),
                                          this->empty_linestring(),
                                          this->empty_polygon(),
                                          this->empty_multipoint(),
                                          this->empty_multipolygon(),
                                          this->empty_multipolygon()),
               cuspatial::logic_error);
}

TYPED_TEST(mixed_geometry_column_test, ConstructorIllegalWrongMultipolygonColumnType)
{
  auto types_buffer   = cudf::test::fixed_width_column_wrapper<mixed_geometry_type_t>{};
  auto offsets_buffer = cudf::test::fixed_width_column_wrapper<int64_t>{};

  EXPECT_THROW(mixed_geometry_column_view(types_buffer,
                                          offsets_buffer,
                                          this->empty_point(),
                                          this->empty_linestring(),
                                          this->empty_polygon(),
                                          this->empty_multipoint(),
                                          this->empty_multilinestring(),
                                          this->empty_multipoint()),
               cuspatial::logic_error);
}
