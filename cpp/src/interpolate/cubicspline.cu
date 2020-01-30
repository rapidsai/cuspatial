/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "cuspatial/cubicspline.hpp"
#include "cudf/column/column_factories.hpp"

namespace cuspatial
{

std::unique_ptr<cudf::experimental::table> cubicspline(
    cudf::column_view t,
    cudf::column_view x,
    cudf::column_view ids
)
{
    std::unique_ptr<cudf::column> column_1 = cudf::make_numeric_column(cudf::data_type{cudf::FLOAT64}, ids.size());
    std::unique_ptr<cudf::column> column_2 = cudf::make_numeric_column(cudf::data_type{cudf::FLOAT64}, ids.size());
    std::vector<std::unique_ptr<cudf::column>> table;
    table.push_back(std::move(column_1));
    table.push_back(std::move(column_2));
    std::unique_ptr<cudf::experimental::table> result = std::make_unique<cudf::experimental::table>(move(table));
    return result;

    // steps
    // 1. allocate return Table
    // return is m x n where m is len(ids_and_end_coordinates) and
    // n is 4 * len(y.columns)
    // 2. iterate over ids_and_end_coordinates, calling kernel function
    // for each
    // 3. return table

    // first steps:
    // 1. allocate fake return table
    // fake return table is m x n where m is len(ids_and_end_coordinates)
    // and n is 2
    // 2. write kernel function that writes the current id to the first
    // column and writes the id * the end_coordinate into the second column
    // 3. iterate over ids_and_end_coordinates, calling kernel function
    // for each
    // 4. return table
}

std::unique_ptr<cudf::experimental::table> cubicspline(
    cudf::column_view x,
    cudf::table_view y,
    cudf::table_view ids_and_end_coordinates
)
{
    std::unique_ptr<cudf::column> column = cudf::make_numeric_column(cudf::data_type{cudf::FLOAT64}, ids_and_end_coordinates.num_rows());
    std::unique_ptr<cudf::experimental::table> result = std::make_unique<cudf::experimental::table>(y);
    return result;

    // steps
    // 1. allocate return Table
    // return is m x n where m is len(ids_and_end_coordinates) and
    // n is 4 * len(y.columns)
    // 2. iterate over ids_and_end_coordinates, calling kernel function
    // for each
    // 3. return table

    // first steps:
    // 1. allocate fake return table
    // fake return table is m x n where m is len(ids_and_end_coordinates)
    // and n is 2
    // 2. write kernel function that writes the current id to the first
    // column and writes the id * the end_coordinate into the second column
    // 3. iterate over ids_and_end_coordinates, calling kernel function
    // for each
    // 4. return table
}
}
