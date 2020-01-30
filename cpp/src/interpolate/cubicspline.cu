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
#include "cusparse.h"
#include <thrust/device_vector.h>

namespace cuspatial
{

std::unique_ptr<cudf::experimental::table> cubicspline(
    cudf::column_view t,
    cudf::column_view x,
    cudf::column_view ids
)
{
    // steps
    // compute array values
    // 1. compute precursor values for tridiagonal matrix
    // 2. allocate sparse matrix inputs
    // 3. fill tridiagonal matrix
    // 4. call cusparse<T>gtsv2() to solve
    // 5. finish coefficient calculations
    
    // 1. compute precursor values for tridiagonal matrix
    // DO
    thrust::device_vector<float> t_(4);
    t_[0] = 0;
    t_[1] = 1;
    t_[2] = 2;
    t_[3] = 3;
    thrust::device_vector<float> x_(4);
    x_[0] = 3;
    x_[1] = 2;
    x_[2] = 3;
    x_[3] = 4;
    thrust::device_vector<float> h(t_.begin(), t_.end()-1);
    thrust::transform(t_.begin()+1, t_.end(), h.begin(), h.begin(), thrust::minus<float>()); 
    thrust::copy(h.begin(), h.end(), std::ostream_iterator<float>(std::cout, "\n"));
    thrust::copy(x_.begin(), x_.end(), std::ostream_iterator<float>(std::cout, "\n"));
    thrust::device_vector<float> b(x_.begin(), x_.end()-1);
    thrust::transform(x_.begin()+1, x_.end(), b.begin(), b.begin(), thrust::minus<float>());
    thrust::transform(b.begin(), b.end(), h.begin(), b.begin(), thrust::divides<float>());
    thrust::copy(b.begin(), b.end(), std::ostream_iterator<float>(std::cout, "\n"));
    thrust::device_vector<float> v(h.begin(), h.end()-1);
    thrust::device_vector<float> two(v.size(), 2);
    thrust::transform(h.begin()+1, h.end(), v.begin(), v.begin(), thrust::plus<float>());
    thrust::transform(v.begin(), v.end(), two.begin(), v.begin(), thrust::multiplies<float>());
    thrust::copy(v.begin(), v.end(), std::ostream_iterator<float>(std::cout, "\n"));
    thrust::device_vector<float> u(b.begin(), b.end()-1);
    thrust::device_vector<float> six(b.size(), 6);
    thrust::transform(b.begin()+1, b.end(), u.begin(), u.begin(), thrust::minus<float>());
    thrust::transform(u.begin(), u.end(), six.begin(), u.begin(), thrust::multiplies<float>());
    thrust::copy(u.begin(), u.end(), std::ostream_iterator<float>(std::cout, "\n"));


    std::unique_ptr<cudf::column> column_1 = cudf::make_numeric_column(cudf::data_type{cudf::FLOAT64}, ids.size());
    std::unique_ptr<cudf::column> column_2 = cudf::make_numeric_column(cudf::data_type{cudf::FLOAT64}, ids.size());
    std::vector<std::unique_ptr<cudf::column>> table;
    table.push_back(std::move(column_1));
    table.push_back(std::move(column_2));
    std::unique_ptr<cudf::experimental::table> result = std::make_unique<cudf::experimental::table>(move(table));
    return result;
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
