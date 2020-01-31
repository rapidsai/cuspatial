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

namespace
{

void HANDLE_CUSPARSE_STATUS(cusparseStatus_t status) {
  if(status != CUSPARSE_STATUS_SUCCESS) {
    const char* status_string;
    switch(status) {
      case CUSPARSE_STATUS_SUCCESS:
          status_string = "CUSPARSE_STATUS_SUCCESS";
          break;
      case CUSPARSE_STATUS_NOT_INITIALIZED:
          status_string = "CUSPARSE_STATUS_NOT_INITIALIZED";
          break;
      case CUSPARSE_STATUS_ALLOC_FAILED:
          status_string = "CUSPARSE_STATUS_ALLOC_FAILED";
          break;
      case CUSPARSE_STATUS_INVALID_VALUE:
          status_string = "CUSPARSE_STATUS_INVALID_VALUE";
          break;
      case CUSPARSE_STATUS_ARCH_MISMATCH:
          status_string = "CUSPARSE_STATUS_ARCH_MISMATCH";
          break;
      case CUSPARSE_STATUS_EXECUTION_FAILED:
          status_string = "CUSPARSE_STATUS_EXECUTION_FAILED";
          break;
      case CUSPARSE_STATUS_INTERNAL_ERROR:
          status_string = "CUSPARSE_STATUS_INTERNAL_ERROR";
          break;
      case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
          status_string = "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
          break;
      default:
          status_string = "UNKNOWN";
    }
    printf("Cusparse error status %s\n", status_string);
  }
  assert(False);
}

template<typename T>
void tPrint(rmm::device_vector<T> vec, const char* name="None") {
    std::cout << name << "\n";
    thrust::copy(vec.begin(), vec.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << "\n";
}

} // anonymous namespace

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
    rmm::device_vector<float> t_(5);
    t_[0] = 0;
    t_[1] = 1;
    t_[2] = 2;
    t_[3] = 3;
    t_[4] = 4;
    rmm::device_vector<float> x_(5);
    x_[0] = 3;
    x_[1] = 2;
    x_[2] = 3;
    x_[3] = 4;
    x_[4] = 3;

    // h = t[1:] - t[:-1]
    rmm::device_vector<float> h(t_.begin(), t_.end()-1);
    thrust::transform(t_.begin()+1, t_.end(), h.begin(), h.begin(), thrust::minus<float>()); 
    tPrint(h, "h");

    // b = (y[1:]-y[:-1])/h
    rmm::device_vector<float> b(x_.begin(), x_.end()-1);
    thrust::transform(x_.begin()+1, x_.end(), b.begin(), b.begin(), thrust::minus<float>());
    thrust::transform(b.begin(), b.end(), h.begin(), b.begin(), thrust::divides<float>());
    tPrint(b, "b");

    // v = 2*(h[:-1]+h[1:])
    rmm::device_vector<float> v(h.begin(), h.end()-1);
    rmm::device_vector<float> two(v.size(), 2);
    thrust::transform(h.begin()+1, h.end(), v.begin(), v.begin(), thrust::plus<float>());
    thrust::transform(v.begin(), v.end(), two.begin(), v.begin(), thrust::multiplies<float>());
    tPrint(v, "v");

    // u = 6*(b[1:] - b[:-1])
    rmm::device_vector<float> u(b.begin(), b.end()-1);
    rmm::device_vector<float> six(b.size(), 6);
    thrust::transform(b.begin()+1, b.end(), u.begin(), u.begin(), thrust::minus<float>());
    thrust::transform(u.begin(), u.end(), six.begin(), u.begin(), thrust::multiplies<float>());
    tPrint(u, "u");

    // 2. allocate sparse matrix inputs
    // 3. fill tridiagonal matrix
    // M = cp.zeros((n-2, n-2))
    // cp.fill_diagonal(M, v)
    // cp.fill_diagonal(M[1:], h[1:-1])
    // cp.fill_diagonal(M[:, 1:], h[1:-1])
    rmm::device_vector<float> d(v.begin(), v.end());
    rmm::device_vector<float> dl(h.begin()+1, h.end()-1);
    rmm::device_vector<float> du(h.begin()+1, h.end()-1);

    // 4. call cusparse<T>gtsv2() to solve
    // 4.1 Get cuSparse library context
    // compute inputs:
    // handle: the cuSparse library context
    // m: size
    // n: number of columns of solution matrix B
    // dl, d, du: vectors of the diagonal
    // B: (ldb, n) dimensional dense matrix to be solved for
    // ldb: leading dimension of B
    // pBuffer: get size of thisu by gtsv2_bufferSizeExt 
    cusparseStatus_t cusparseStatus;
    cusparseHandle_t handle;
    cudaMalloc(&handle, sizeof(cusparseHandle_t));
    cusparseStatus = cusparseCreate(&handle);
    HANDLE_CUSPARSE_STATUS(cusparseStatus);
    size_t pBufferSize;
    float* dlp = thrust::raw_pointer_cast(dl.data());
    cusparseStatus = cusparseSgtsv2_bufferSizeExt(
        handle,
        u.size(), 
        1,
        thrust::raw_pointer_cast(dl.data()),
        thrust::raw_pointer_cast(d.data()),
        thrust::raw_pointer_cast(du.data()),
        thrust::raw_pointer_cast(u.data()),
        u.size(),
        &pBufferSize
    );
    HANDLE_CUSPARSE_STATUS(cusparseStatus);
    rmm::device_vector<float> pBuffer(pBufferSize);
    cusparseStatus = cusparseSgtsv2(
        handle,
        u.size(),
        1,
        thrust::raw_pointer_cast(dl.data()),
        thrust::raw_pointer_cast(d.data()),
        thrust::raw_pointer_cast(du.data()),
        thrust::raw_pointer_cast(u.data()),
        u.size(),
        thrust::raw_pointer_cast(pBuffer.data())
    );
    HANDLE_CUSPARSE_STATUS(cusparseStatus);
    tPrint(u, "u");

    // 5. finish coefficient calculations
    // a = y[:-1]
    rmm::device_vector<float> a(x_.begin(), x_.end()-1);
    tPrint(a, "a"); 

    // b = b - h*(z[1:] + 2*z[:-1])/6 
    rmm::device_vector<float> z(u.begin(), u.end());
    rmm::device_vector<float> two_z(z.begin(), z.end()-1);
    thrust::transform(two_z.begin(), two_z.end(), two.begin(), two_z.end(), thrust::multiplies<float>());
    rmm::device_vector<float> z_tmp(z.begin()+1, z.end());
    thrust::transform(z_tmp.begin(), z_tmp.end(), two_z.begin(), z_tmp.begin(), thrust::plus<float>());
    rmm::device_vector<float> h_over_six(h.begin(), h.end());
    thrust::transform(h.begin(), h.end(), six.begin(), h_over_six.begin(), thrust::divides<float>());
    thrust::transform(z_tmp.begin(), z_tmp.end(), h_over_six.begin(), z_tmp.begin(), thrust::multiplies<float>());
    thrust::transform(b.begin(), b.end(), z_tmp.begin(), b.begin(), thrust::minus<float>());
    tPrint(b, "b");

    // END:
    // Basic columnar operations to prepare return values to `cuspatial` DataFrame
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
