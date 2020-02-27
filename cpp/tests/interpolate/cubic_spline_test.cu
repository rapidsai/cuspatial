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
#include <cudf/utilities/type_dispatcher.hpp>
#include <cuspatial/cubicspline.hpp>

static void HandleCudaError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_CUDA_ERROR( err ) (HandleCudaError( err, __FILE__, __LINE__ ))


struct CubicSplineTest : public GdfTest 
{
 
};

template<typename T>
auto make_col(T* const points, int length) {
    T *d_p = NULL;
    RMM_TRY( RMM_ALLOC( &d_p, length * sizeof(T), 0));
    assert(d_p != NULL);    
    HANDLE_CUDA_ERROR( cudaMemcpy( d_p, points, length * sizeof(T), cudaMemcpyHostToDevice ) );
    cudf::column_view col(cudf::data_type{cudf::experimental::type_to_id<T>()}, length, d_p);
    cudf::column result(col);
    RMM_FREE(d_p, 0);
    return result;
}

TEST_F(CubicSplineTest, test_coefficients_full)
{
    int point_len = 15;
    float t[point_len] = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
    float y[point_len] = {3, 2, 3, 4, 3, 3, 2, 3, 4, 3, 3, 2, 3, 4, 3};
    int ids_len = 4;
    int ids[ids_len] = {0, 0, 1, 2};
    int prefix[ids_len] = {0, 5, 10, 15};
   
    cudf::column t_column = make_col<float>(t, point_len);
    cudf::column y_column = make_col<float>(y, point_len);
    cudf::column ids_column = make_col<int>(ids, ids_len);
    cudf::column prefix_column = make_col<int>(prefix, ids_len);

    std::unique_ptr<cudf::experimental::table> splines =
        cuspatial::cubicspline_full(
            t_column, y_column, ids_column, prefix_column
        );
    cudf::column_view device_column = splines->view().column(0);
    std::vector<float> host_data;
    host_data.resize(device_column.size());
    cudaMemcpy(host_data.data(), device_column.data<float>(),
          device_column.size() * sizeof(float),
          cudaMemcpyDeviceToHost);

    for(unsigned int i = 0 ; i < host_data.size() ; ++i ){
      std::cout << host_data[i] << " ";
    }
    std::cout << std::endl;
}

TEST_F(CubicSplineTest, test_interpolate_single)
{
    int point_len = 5;
    float t[point_len] = {0, 1, 2, 3, 4};
    float x[point_len] = {3, 2, 3, 4, 3};
    int ids_len = 2;
    int ids[ids_len] = {0, 0};
    int prefix[ids_len] = {0, 5};
    int point_prefix[point_len] = {0, 0, 0, 0, 0};
    
    cudf::column t_column = make_col<float>(t, point_len);
    cudf::column x_column = make_col<float>(x, point_len);
    cudf::column ids_column = make_col<int>(ids, ids_len);
    cudf::column prefix_column = make_col<int>(prefix, ids_len);
    cudf::column point_prefix_column = make_col<int>(point_prefix, point_len);

    std::unique_ptr<cudf::experimental::table> splines = cuspatial::cubicspline_full(t_column, x_column, ids_column, prefix_column);

    std::unique_ptr<cudf::column> interpolates = cuspatial::cubicspline_interpolate(t_column, point_prefix_column, prefix_column, t_column, splines->view());
    
    cudf::column_view device_column = interpolates->view();
    std::vector<float> host_data;
    host_data.resize(device_column.size());
    cudaMemcpy(host_data.data(), device_column.data<float>(),
          device_column.size() * sizeof(float),
          cudaMemcpyDeviceToHost);

    for(int i = 0 ; i < device_column.size() ; ++i){
      EXPECT_EQ(x[i], host_data[i]);
    }
}

