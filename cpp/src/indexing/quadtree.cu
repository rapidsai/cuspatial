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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>

namespace cuspatial {

/*
 * Return 
 * see indexing.hpp
 */
cudf::column_view quadtree_on_points(cudf::column_view x,cudf::column_view y)
{
    std::vector<column_view> children;
    std::unique_ptr<column> key_col=cudf::make_numeric_column(data_type{INT32}, 0);
    children->push_back(key_col->view());
    std::unique_ptr<column> indicator_col=cudf::make_numeric_column(data_type{BOOL8}, 0);
    children->push_back(indicator_col->view());
    std::unique_ptr<column> fpos_col=cudf::make_numeric_column(data_type{INT32}, 0);
    children->push_back(fpos_col->view());
    std::unique_ptr<column> len_col=cudf::make_numeric_column(data_type{INT32}, 0);
    children->push_back(len_col->view());
    
    children->push_back(x);
    children->push_back(y);
    
    cudf::column_view ret=cudf::column_view(
    	cudf::data_type{cudf::EMPTY},0,nullptr,nullptr,cudf::UNKNOWN_NULL_COUNT,0,children);
}

}// namespace cuspatial
