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

#include <proj.h>

#include <gtest/gtest.h>

#include <iostream>

template <typename T>
struct ProjectionTest : public ::testing::Test {};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(ProjectionTest, TestTypes);

TYPED_TEST(ProjectionTest, Empty)
{
  PJ_CONTEXT* C;
  PJ* P;

  C = proj_context_create();

  P = proj_create_crs_to_crs(C, "EPSG:4326", "EPSG:32756", NULL);

  PJ_COORD input_coords,
    output_coords;  // https://proj.org/development/reference/datatypes.html#c.PJ_COORD

  input_coords = proj_coord(-28.667003, 153.090959, 0, 0);

  output_coords = proj_trans(P, PJ_FWD, input_coords);

  std::cout << output_coords.xy.x << " " << output_coords.xy.y << std::endl;

  /* Clean up */
  proj_destroy(P);
  proj_context_destroy(C);  // may be omitted in the single threaded case
}
