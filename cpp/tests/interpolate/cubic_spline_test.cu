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

#include <time.h>
#include <sys/time.h>
#include <string>

#include <gtest/gtest.h>
#include <utilities/legacy/error_utils.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>
#include <cuspatial/quadtree.hpp>
#include <cuspatial/cubicspline.hpp>

struct CubicSplineTest : public GdfTest 
{
 
};

TEST_F(CubicSplineTest, test1)
{
    int point_len = 5;
    float t[5] = {0, 1, 2, 3, 4};
    assert(sizeof(t) / sizeof(int)==point_len);
    float x[5] = {3, 2, 3, 4, 3};
    assert(sizeof(x) / sizeof(int)==point_len);
    int ids[5] = {0, 1, 2, 3, 4};
    
    float *d_p_t = NULL;
    float *d_p_x = NULL;
    float *d_p_ids = NULL;
    RMM_TRY( RMM_ALLOC( &d_p_t,point_len* sizeof(float), 0));
    assert(d_p_t != NULL);    
    RMM_TRY( RMM_ALLOC( &d_p_x,point_len* sizeof(float), 0));
    assert(d_p_x != NULL);
    RMM_TRY( RMM_ALLOC( &d_p_ids,point_len* sizeof(int), 0));
    assert(d_p_ids != NULL);    
 
    HANDLE_CUDA_ERROR( cudaMemcpy( d_p_t, t, point_len * sizeof(float), cudaMemcpyHostToDevice ) );    
    HANDLE_CUDA_ERROR( cudaMemcpy( d_p_x, x, point_len * sizeof(float), cudaMemcpyHostToDevice ) );     
    HANDLE_CUDA_ERROR( cudaMemcpy( d_p_ids, ids, point_len * sizeof(int), cudaMemcpyHostToDevice ) );     
    
    cudf::column_view t_column(cudf::data_type{cudf::FLOAT64},point_len,d_p_t);
    cudf::column_view x_column(cudf::data_type{cudf::FLOAT64},point_len,d_p_x);
    cudf::column_view ids_column(cudf::data_type{cudf::FLOAT64},point_len,d_p_ids);
    std::unique_ptr<cudf::experimental::table> splines = cuspatial::cubicspline_column(t_column, x_column, ids_column);
    std::cout<<"num cols="<<splines->view().num_columns()<<std::endl;
    std::cout<<"len table="<<splines->view().num_rows()<<std::endl;
}

