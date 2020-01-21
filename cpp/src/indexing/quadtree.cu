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
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>
#include <cuspatial/quadtree.hpp>

namespace cuspatial {

std::unique_ptr<cudf::column> nested_column_test(cudf::column_view x,cudf::column_view y)
{ 
    std::vector<std::unique_ptr<cudf::column>> children;
    
    std::unique_ptr<cudf::column> key_col=cudf::make_numeric_column(cudf::data_type{cudf::INT32}, 1);
    children.push_back(std::move(key_col));

    std::unique_ptr<cudf::column> indicator_col=cudf::make_numeric_column(cudf::data_type{cudf::BOOL8}, 1);
    children.push_back(std::move(indicator_col));

    std::unique_ptr<cudf::column> fpos_col=cudf::make_numeric_column(cudf::data_type{cudf::INT32}, 1);
    children.push_back(std::move(fpos_col));

    std::unique_ptr<cudf::column> len_col=cudf::make_numeric_column(cudf::data_type{cudf::INT32}, 1);
    children.push_back(std::move(len_col));

    //children.push_back(x);
    //children.push_back(y);
    
    //cudf::data_type type=cudf::data_type{cudf::EMPTY};
    cudf::data_type type=cudf::data_type{cudf::INT32};
    cudf::size_type size=1;
    cudf::mask_state state=cudf::mask_state::ALL_NULL;
    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();
    rmm::device_buffer  buffer{size * cudf::size_of(type), stream, mr};
    rmm::device_buffer nmask=create_null_mask(size, state, stream, mr);
    cudf::size_type ncount=state_null_count(state, size);
    
    std::unique_ptr<cudf::column> ret=std::make_unique<cudf::column>(type,size,buffer,nmask,ncount,std::move(children));
    return ret;
}

std::unique_ptr<cudf::experimental::table> quadtree_on_points(cudf::column_view x,cudf::column_view y)
{
    //x/y not used for now
    std::vector<std::unique_ptr<cudf::column>> src_cols;
    
    std::unique_ptr<cudf::column> key_col=cudf::make_numeric_column(cudf::data_type{cudf::INT32}, 1); 
    src_cols.push_back(std::move(key_col));
    
    std::unique_ptr<cudf::column> indicator_col=cudf::make_numeric_column(cudf::data_type{cudf::BOOL8}, 1);
    src_cols.push_back(std::move(indicator_col));
    
    std::unique_ptr<cudf::column> fpos_col=cudf::make_numeric_column(cudf::data_type{cudf::INT32}, 1);
    src_cols.push_back(std::move(fpos_col));
    
    std::unique_ptr<cudf::column> len_col=cudf::make_numeric_column(cudf::data_type{cudf::INT32}, 1);
    src_cols.push_back(std::move(len_col));

    
    std::unique_ptr<cudf::experimental::table> destination_table = std::make_unique<cudf::experimental::table>(std::move(src_cols));
        
    return destination_table;
}

}// namespace cuspatial
