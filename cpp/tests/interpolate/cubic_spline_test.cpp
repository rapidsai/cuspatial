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
#include <cuspatial/cubic_spline.hpp>
#include <cuspatial/utility.hpp>

struct CubicSplineTest : public GdfTest 
{
 
};

template<typename T>
auto make_device_column(T* const points, int length) {
    T *d_p = nullptr;
    RMM_TRY( RMM_ALLOC( &d_p, length * sizeof(T), 0));
    assert(d_p != nullptr);    
    HANDLE_CUDA_ERROR( cudaMemcpy( d_p, points, length * sizeof(T), cudaMemcpyHostToDevice ) );
    cudf::column_view col(cudf::data_type{cudf::experimental::type_to_id<T>()}, length, d_p);
    cudf::column result(col);
    RMM_FREE(d_p, 0);
    return result;
}

auto get_d_expect() {
  std::array<float, 4> d3_expect{{0.5, -0.5, -0.5, 0.5}};
  std::array<float, 4> d2_expect{{0, 3, 3, -6}};
  std::array<float, 4> d1_expect{{-1.5, -4.5, -4.5, 22.5}};
  std::array<float, 4> d0_expect{{3, 4, 4, -23}};
  std::vector<std::array<float, 4>> d_expect;
  d_expect.push_back(d3_expect);
  d_expect.push_back(d2_expect);
  d_expect.push_back(d1_expect);
  d_expect.push_back(d0_expect);
  return d_expect;
}

TEST_F(CubicSplineTest, test_coefficients_single)
{
    cudaStream_t stream = 0;
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource();

    int point_len = 5;
    float t[point_len] = {0, 1, 2, 3, 4};
    float y[point_len] = {3, 2, 3, 4, 3};
    int ids_len = 2;
    int ids[ids_len] = {0, 0};
    int prefix[ids_len] = {0, 5};

    cudf::column t_column = make_device_column<float>(t, point_len);
    cudf::column y_column = make_device_column<float>(y, point_len);
    cudf::column ids_column = make_device_column<int>(ids, ids_len);
    cudf::column prefix_column = make_device_column<int>(prefix, ids_len);

    std::unique_ptr<cudf::experimental::table> splines =
        cuspatial::cubicspline_coefficients(
            t_column, y_column, ids_column, prefix_column, mr, stream
        );

    auto d_expect = get_d_expect();
    for(unsigned int i = 0 ; i < d_expect.size() ; ++i){
      cudf::column_view device_column = splines->view().column(i);
      std::vector<float> host_data;
      host_data.resize(device_column.size());
      cudaMemcpy(host_data.data(), device_column.data<float>(),
            device_column.size() * sizeof(float),
            cudaMemcpyDeviceToHost);
      for(unsigned int j = 0 ; j < host_data.size() ; ++j ){
        EXPECT_EQ(d_expect[i][j], host_data[j]);
      }
    }
}

TEST_F(CubicSplineTest, test_coefficients_full)
{
    cudaStream_t stream = 0;
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource();

    int point_len = 15;
    float t[point_len] = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
    float y[point_len] = {3, 2, 3, 4, 3, 3, 2, 3, 4, 3, 3, 2, 3, 4, 3};
    int ids_len = 4;
    int ids[ids_len] = {0, 0, 1, 2};
    int prefix[ids_len] = {0, 5, 10, 15};
 
    cudf::column t_column = make_device_column<float>(t, point_len);
    cudf::column y_column = make_device_column<float>(y, point_len);
    cudf::column ids_column = make_device_column<int>(ids, ids_len);
    cudf::column prefix_column = make_device_column<int>(prefix, ids_len);

    std::unique_ptr<cudf::experimental::table> splines =
        cuspatial::cubicspline_coefficients(
            t_column, y_column, ids_column, prefix_column, mr, stream
        );

    auto d_expect = get_d_expect();
    for(unsigned int i = 0 ; i < d_expect.size() ; ++i){
      cudf::column_view device_column = splines->view().column(i);
      std::vector<float> host_data;
      host_data.resize(device_column.size());
      cudaMemcpy(host_data.data(), device_column.data<float>(),
            device_column.size() * sizeof(float),
            cudaMemcpyDeviceToHost);
      for(unsigned int j = 0 ; j < host_data.size() ; ++j ){
        EXPECT_EQ(d_expect[i][j%d_expect[i].size()], host_data[j]);
      }
    }
}

TEST_F(CubicSplineTest, test_interpolate_single)
{ 
    cudaStream_t stream = 0;
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource();

    int point_len = 5;
    float t[point_len] = {0, 1, 2, 3, 4};
    float x[point_len] = {3, 2, 3, 4, 3};
    int ids_len = 2;
    int ids[ids_len] = {0, 0};
    int prefix[ids_len] = {0, 5};
    int point_ids[point_len] = {0, 0, 0, 0, 0};

    cudf::column t_column = make_device_column<float>(t, point_len);
    cudf::column x_column = make_device_column<float>(x, point_len);
    cudf::column ids_column = make_device_column<int>(ids, ids_len);
    cudf::column prefix_column = make_device_column<int>(prefix, ids_len);
    cudf::column point_ids_column = make_device_column<int>(point_ids, point_len);

    std::unique_ptr<cudf::experimental::table> splines = cuspatial::cubicspline_coefficients(t_column, x_column, ids_column, prefix_column, mr, stream);

    std::unique_ptr<cudf::column> interpolates = cuspatial::cubicspline_interpolate(t_column, point_ids_column, prefix_column, t_column, splines->view(), mr, stream);
    
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

TEST_F(CubicSplineTest, test_interpolate_full)
{
    cudaStream_t stream = 0;
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource();

    int point_len = 15;
    float t[point_len] = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
    float x[point_len] = {3, 2, 3, 4, 3, 3, 2, 3, 4, 3, 3, 2, 3, 4, 3};
    int ids_len = 4;
    int ids[ids_len] = {0, 0, 1, 2};
    int prefix[ids_len] = {0, 5, 10, 15};
    int point_ids[point_len] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2};

    cudf::column t_column = make_device_column<float>(t, point_len);
    cudf::column x_column = make_device_column<float>(x, point_len);
    cudf::column ids_column = make_device_column<int>(ids, ids_len);
    cudf::column prefix_column = make_device_column<int>(prefix, ids_len);
    cudf::column point_ids_column = make_device_column<int>(point_ids, point_len);

    std::unique_ptr<cudf::experimental::table> splines = cuspatial::cubicspline_coefficients(t_column, x_column, ids_column, prefix_column, mr, stream);

    std::unique_ptr<cudf::column> interpolates = cuspatial::cubicspline_interpolate(t_column, point_ids_column.view(), prefix_column, t_column, splines->view(), mr, stream);
    
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

