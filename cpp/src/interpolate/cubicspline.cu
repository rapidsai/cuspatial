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
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include "cusparse.h"
#include <thrust/device_vector.h>

namespace { // anonymous

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
void tPrint(thrust::detail::normal_iterator<T> start, thrust::detail::normal_iterator<T> stop, const char* name="None") {
    std::cout << name << " size: " << stop-start << " ";
    thrust::copy(start, stop, std::ostream_iterator<float>(std::cout, " "));
    std::cout << "\n";
}

template<typename T>
void tPrint(rmm::device_vector<T> vec, const char* name="None") {
  tPrint(vec.begin(), vec.end(), name);
}

#define ALLOW_PRINT 1
#if ALLOW_PRINT
#define TPRINT(vec, name) (tPrint( vec, name))
#else
#define TPRINT(vec, name) {}
#endif

struct mul_scalar_functor
{
  const float a;
  mul_scalar_functor(float _a) : a(_a) {}
  __host__ __device__
    float operator()(const float& x) const {
      return a * x;
    }
};
struct calc_d_functor
{
  template<typename Tuple>
  __host__ __device__
    void operator()(Tuple t)
    {
      thrust::get<3>(t) = (thrust::get<0>(t) - thrust::get<1>(t)) / 6.0*thrust::get<2>(t);
    }
};
struct calc_deg_2_functor
{
  template<typename Tuple>
  __host__ __device__
    void operator()(Tuple t)
    {
      // (c - 3*d*t)
      thrust::get<3>(t) = thrust::get<0>(t) - 3.0 * thrust::get<1>(t) * thrust::get<2>(t);
    }
};
struct calc_deg_1_functor
{
  template<typename Tuple>
  __host__ __device__
    void operator()(Tuple t)
    {
      // (b - (2*c*t) + (3*d*t*t))
      thrust::get<4>(t) = thrust::get<0>(t) - (2.0 * thrust::get<1>(t) * thrust::get<3>(t)) + (3.0 * thrust::get<2>(t) * thrust::get<3>(t) * thrust::get<3>(t));
    }
};
struct calc_deg_0_functor
{
  template<typename Tuple>
  __host__ __device__
    void operator()(Tuple t)
    {
      // (a - (b*t) + (c*t*t) - (d*t*t*t))
      thrust::get<5>(t) = thrust::get<0>(t) - thrust::get<1>(t) * thrust::get<4>(t) + thrust::get<2>(t) * thrust::get<4>(t) * thrust::get<4>(t) - thrust::get<3>(t) * thrust::get<4>(t) * thrust::get<4>(t) * thrust::get<4>(t);
    }
};

struct intermediate_kernel {
template<typename T, std::enable_if_t<std::is_floating_point<T>::value >* = nullptr>
    void operator()(cudf::column_view& t, cudf::column_view& y,
        cudf::mutable_column_view& buffer,
        rmm::mr::device_memory_resource *mr,
        cudaStream_t stream)
      {
          // Compute h, i, v, u
          int n = y.size();
          assert(buffer.size() == 2 * (n-1) + 2 * (n-2));
          int h = 0;
          int len_h = n-1;
          int i = h+len_h;
          int len_i = n-1;
          int v = i+len_i;
          int len_v = n-2;
          int u = v+len_v;
          int len_u = n-2;
          T* data = buffer.data<T>();
          data[0] = 1;
          const T* TT = t.data<T>();
          const T* Y = y.data<T>();
          int ci = 0;
          for(ci = 0 ; ci < len_h ; ++ci) {
              data[h+ci] = TT[h+ci+1] - TT[h+ci];
              data[i+ci] = (Y[h+ci+1] - Y[h+ci]) / data[h+ci];
          }
          for(ci = 0 ; ci < len_v ; ++ci) {
              data[v+ci] = (data[h+ci+1]+data[h+len_h-1-ci]) * 2;
          }
          for(ci = 0 ; ci < len_u ; ++ci) {
              data[u+ci] = (data[i+ci+1] - data[i+len_i-1-ci]) * 6;
          } 
      }
template<typename T, std::enable_if_t<!std::is_floating_point<T>::value >* = nullptr>
    void operator()(cudf::column_view& t, cudf::column_view& y,
        cudf::mutable_column_view& buffer,
        rmm::mr::device_memory_resource *mr,
        cudaStream_t stream)
      {
          CUDF_FAIL("Non-floating point operation is not supported.");
      }
};

} // anonymous namespace

namespace cuspatial
{

std::unique_ptr<cudf::experimental::table> cubicspline_full(
    cudf::column_view t,
    cudf::column_view y,
    cudf::column_view ids,
    cudf::column_view prefixes
)
{
    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();

    // Allocate storage for cuSparse dependency computation
    rmm::device_vector<float> t_(t.data<float>(), t.data<float>()+t.size());
    rmm::device_vector<float> y_(y.data<float>(), y.data<float>()+y.size());
    TPRINT(t_, "t_");
    TPRINT(y_, "y_");

    int64_t n = y.size();
    int64_t tcb_size = 2 * (n-1) + 2 * (n-2);
    cudf::mutable_column_view tridiagonal_creation_buffer = make_numeric_column(y.type(), tcb_size, cudf::UNALLOCATED, stream, mr)->mutable_view();
    cudf::experimental::type_dispatcher(y.type(), intermediate_kernel{}, t, y, tridiagonal_creation_buffer, mr, stream);
    
    // Placeholder result preparation
    std::unique_ptr<cudf::column> result_col = cudf::make_numeric_column(y.type(), y.size());
    float *cd1 = cudf::mutable_column_device_view::create(result_col->mutable_view())->data<float>();
    thrust::copy(y_.begin(), y_.end(), cd1);
    std::vector<std::unique_ptr<cudf::column>> table;
    table.push_back(std::move(result_col));
    std::unique_ptr<cudf::experimental::table> result = std::make_unique<cudf::experimental::table>(move(table));
    return result;
}

std::unique_ptr<cudf::experimental::table> cubicspline_column(
    cudf::column_view t,
    cudf::column_view x,
    cudf::column_view ids
)
{
    // cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();
    // steps
    // compute array values
    // 1. compute precursor values for tridiagonal matrix
    // 2. allocate sparse matrix inputs
    // 3. fill tridiagonal matrix
    // 4. call cusparse<T>gtsv2() to solve
    // 5. finish coefficient calculations
    
    // 1. compute precursor values for tridiagonal matrix
    // DO
    /*
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
    */
    //std::cout << "t Input size " << t.size() << std::endl;
    //std::cout << "x Input size " << x.size() << std::endl;
    rmm::device_vector<float> t_(t.data<float>(), t.data<float>()+t.size());
    rmm::device_vector<float> x_(x.data<float>(), x.data<float>()+x.size());
    TPRINT(t_, "t_");
    TPRINT(x_, "x_");

    // h = t[1:] - t[:-1]
    rmm::device_vector<float> h(t_.begin(), t_.end()-1);
    thrust::transform(t_.begin()+1, t_.end(), h.begin(), h.begin(), thrust::minus<float>()); 
    TPRINT(h, "h");

    // i = (y[1:]-y[:-1])/h
    rmm::device_vector<float> i(x_.begin(), x_.end()-1);
    thrust::transform(x_.begin()+1, x_.end(), i.begin(), i.begin(), thrust::minus<float>());
    thrust::transform(i.begin(), i.end(), h.begin(), i.begin(), thrust::divides<float>());
    TPRINT(i, "i");

    // v = 2*(h[:-1]+h[1:])
    rmm::device_vector<float> v(h.begin(), h.end()-1);
    rmm::device_vector<float> two(v.size(), 2);
    thrust::transform(h.begin()+1, h.end(), v.begin(), v.begin(), thrust::plus<float>());
    thrust::transform(v.begin(), v.end(), two.begin(), v.begin(), thrust::multiplies<float>());
    TPRINT(v, "v");

    // u = 6*(i[1:] - i[:-1])
    rmm::device_vector<float> u(i.begin(), i.end()-1);
    rmm::device_vector<float> six(i.size(), 6);
    thrust::transform(i.begin()+1, i.end(), u.begin(), u.begin(), thrust::minus<float>());
    thrust::transform(u.begin(), u.end(), six.begin(), u.begin(), thrust::multiplies<float>());
    TPRINT(u, "u");

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
    std::cout << "pBufferSize " << pBufferSize << std::endl;
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
    TPRINT(u, "u");

    // 5. finish coefficient calculations
    // a = y[:-1]
    rmm::device_vector<float> a(x_.begin(), x_.end()-1);
    TPRINT(a, "a"); 

    // b = b - h*(z[1:] + 2*z[:-1])/6 
    rmm::device_vector<float> b(i.begin(), i.end());
    rmm::device_vector<float> z(u.size()+2, 0);
    thrust::copy(u.begin(), u.end(), z.begin()+1);
    TPRINT(z, "z");
    rmm::device_vector<float> two_z(z.begin(), z.end()-1);
    rmm::device_vector<float> two_z_len(z.size(), 2);
    thrust::transform(two_z.begin(), two_z.end(), two_z_len.begin(), two_z.begin(), thrust::multiplies<float>());
    TPRINT(two, "two");
    TPRINT(two_z, "two_z");
    rmm::device_vector<float> z_tmp(z.begin()+1, z.end());
    thrust::transform(z_tmp.begin(), z_tmp.end(), two_z.begin(), z_tmp.begin(), thrust::plus<float>());
    TPRINT(z_tmp, "z_tmp");
    rmm::device_vector<float> h_over_six(h.begin(), h.end());
    thrust::transform(h.begin(), h.end(), six.begin(), h_over_six.begin(), thrust::divides<float>());
    TPRINT(h_over_six, "h_over_six");
    thrust::transform(z_tmp.begin(), z_tmp.end(), h_over_six.begin(), z_tmp.begin(), thrust::multiplies<float>());
    thrust::transform(b.begin(), b.end(), z_tmp.begin(), b.begin(), thrust::minus<float>());
    TPRINT(b, "b");

    rmm::device_vector<float> c(z.begin(), z.end()-1);
    thrust::transform(c.begin(), c.end(), c.begin(), mul_scalar_functor(1.0/2.0));
    TPRINT(c, "c");

    rmm::device_vector<float> d_v(h.begin(), h.end());
    thrust::for_each(
        thrust::make_zip_iterator(
          thrust::make_tuple(z.begin()+1, z.begin(), h.begin(), d_v.begin())),
        thrust::make_zip_iterator(
          thrust::make_tuple(z.end(), z.end()-1, h.end(), d_v.end())),
        calc_d_functor());
    TPRINT(d_v, "d_v");

    rmm::device_vector<float> t_final(t_.begin()+1, t_.end());

    rmm::device_vector<float> deg_2(d_v.begin(), d_v.end());
    rmm::device_vector<float> deg_1(d_v.begin(), d_v.end());
    rmm::device_vector<float> deg_0(d_v.begin(), d_v.end());

    thrust::for_each(
        thrust::make_zip_iterator(
          thrust::make_tuple(c.begin(), d_v.begin(), t_final.begin(), deg_2.begin())),
        thrust::make_zip_iterator(
          thrust::make_tuple(c.end(), d_v.end(), t_final.end(), deg_2.end())),
        calc_deg_2_functor());
    thrust::for_each(
        thrust::make_zip_iterator(
          thrust::make_tuple(b.begin(), c.begin(), d_v.begin(), t_final.begin(), deg_1.begin())),
        thrust::make_zip_iterator(
          thrust::make_tuple(b.end(), c.end(), d_v.end(), t_final.end(), deg_1.end())),
        calc_deg_1_functor());
    thrust::for_each(
        thrust::make_zip_iterator(
          thrust::make_tuple(a.begin(), b.begin(), c.begin(), d_v.begin(), t_final.begin(), deg_0.begin())),
        thrust::make_zip_iterator(
          thrust::make_tuple(a.end(), b.end(), c.end(), d_v.end(), t_final.end(), deg_0.end())),
        calc_deg_0_functor());
    TPRINT(c, "c");
    TPRINT(b, "b");
    TPRINT(a, "a");
    TPRINT(t_final, "t");
    TPRINT(d_v, "deg_3");
    TPRINT(deg_2, "deg_2");
    TPRINT(deg_1, "deg_1");
    TPRINT(deg_0, "deg_0");

    std::unique_ptr<cudf::column> column_deg_3 = cudf::make_numeric_column(cudf::data_type{cudf::FLOAT32}, d_v.size());
    float *cd3 = cudf::mutable_column_device_view::create(column_deg_3->mutable_view())->data<float>();
    thrust::copy(d_v.begin(), d_v.end(), cd3);
    std::unique_ptr<cudf::column> column_deg_2 = cudf::make_numeric_column(cudf::data_type{cudf::FLOAT32}, deg_2.size());
    float *cd2 = cudf::mutable_column_device_view::create(column_deg_2->mutable_view())->data<float>();
    thrust::copy(deg_2.begin(), deg_2.end(), cd2);
    std::unique_ptr<cudf::column> column_deg_1 = cudf::make_numeric_column(cudf::data_type{cudf::FLOAT32}, deg_1.size());
    float *cd1 = cudf::mutable_column_device_view::create(column_deg_1->mutable_view())->data<float>();
    thrust::copy(deg_1.begin(), deg_1.end(), cd1);
    std::unique_ptr<cudf::column> column_deg_0 = cudf::make_numeric_column(cudf::data_type{cudf::FLOAT32}, deg_0.size());
    float *cd0 = cudf::mutable_column_device_view::create(column_deg_0->mutable_view())->data<float>();
    thrust::copy(deg_0.begin(), deg_0.end(), cd0);
    // END:
    // Basic columnar operations to prepare return values to `cuspatial` DataFrame
    std::vector<std::unique_ptr<cudf::column>> table;
    table.push_back(std::move(column_deg_3));
    table.push_back(std::move(column_deg_2));
    table.push_back(std::move(column_deg_1));
    table.push_back(std::move(column_deg_0));
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
