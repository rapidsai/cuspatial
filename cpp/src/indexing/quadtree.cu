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

#include <vector>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>
#include <cuspatial/quadtree.hpp>

namespace cuspatial {

std::unique_ptr<cudf::experimental::table> quadtree_on_points(cudf::column_view x,cudf::column_view y)
{
    //x/y not used for now
    std::vector<std::unique_ptr<cudf::column>> src_cols;
    
    std::unique_ptr<cudf::column> key_col=cudf::make_numeric_column(cudf::data_type{cudf::INT32}, 1); 
    src_cols.push_back(key_col);
    
    std::unique_ptr<cudf::column> indicator_col=cudf::make_numeric_column(cudf::data_type{cudf::BOOL8}, 1);
    src_cols.push_back(indicator_col);
    
    std::unique_ptr<cudf::column> fpos_col=cudf::make_numeric_column(cudf::data_type{cudf::INT32}, 1);
    src_cols.push_back(fpos_col);
    
    std::unique_ptr<cudf::column> len_col=cudf::make_numeric_column(cudf::data_type{cudf::INT32}, 1);
    src_cols.push_back(len_col);

    
    std::unique_ptr<cudf::experimental::table> destination_table = std::make_unique<cudf::experimental::table>(std::move(src_cols));
        
    return destination_table;
}

}// namespace cuspatial
