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
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
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
    CUDF_EXPECTS(status != CUSPARSE_STATUS_SUCCESS, "Fail");
  }
}

#define ALLOW_PRINT 1
#if ALLOW_PRINT

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

void tPrint(cudf::mutable_column_view col, const char* name="None") {
  rmm::device_vector<float> vec = rmm::device_vector<float>(col.data<float>(), col.data<float>()+col.size());
  tPrint(vec.begin(), vec.end(), name);
}

void tPrint(cudf::column_view col, const char* name="None") {
  rmm::device_vector<float> vec = rmm::device_vector<float>(col.data<float>(), col.data<float>()+col.size());
  tPrint(vec.begin(), vec.end(), name);
}
#define TPRINT(vec, name) (tPrint( vec, name))

#else
#define TPRINT(vec, name) {}
#endif

struct parallel_search {
  template <typename T>
  void operator()(cudf::column_view const& t,
                  cudf::column_view const& prefixes,
                  cudf::column_view const& query_coords,
                  cudf::mutable_column_view const& result,
                  rmm::mr::device_memory_resource *mr,
                  cudaStream_t stream) {
      const T* T_ = t.data<T>();
      const int32_t* PREFIXES_ = prefixes.data<int32_t>();
      T* QUERY_COORDS_ = query_coords.data<T>();
      T* RESULT_ = result.data<T>();
      thrust::for_each(rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<int>(1),
        thrust::make_counting_iterator<int>(prefixes.size()),
        [T_, PREFIXES_, QUERY_COORDS_, RESULT_] __device__
        (int index) {
          int len = PREFIXES_[index] - PREFIXES_[index-1];
          int h = PREFIXES_[index];
          RESULT_[index] = len-1;
          for(int32_t i = 0 ; i < len-1 ; ++i) {
            if(QUERY_COORDS_[index] > T_[h+i]) {
              RESULT_[index] = i;
              return;
            }
          }
          RESULT_[index] = -1.0;
      });
  };
};

struct interpolate {
  template <typename T>
  void operator()(cudf::column_view const& t,
                  cudf::column_view const& positions,
                  cudf::mutable_column_view const& d3,
                  cudf::mutable_column_view const& d2,
                  cudf::mutable_column_view const& d1,
                  cudf::mutable_column_view const& d0,
                  cudf::mutable_column_view const& result,
                  rmm::mr::device_memory_resource *mr,
                  cudaStream_t stream) {
      const T* T_ = t.data<T>();
      const int32_t* POSITIONS_ = positions.data<int32_t>();
      T* D3_ = d3.data<T>();
      T* D2_ = d2.data<T>();
      T* D1_ = d1.data<T>();
      T* D0_ = d0.data<T>();
      T* RESULT_ = result.data<T>();
      thrust::for_each(rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(positions.size()-1),
        [T_, POSITIONS_, D3_, D2_, D1_, D0_, RESULT_] __device__
        (int index) {
          int h = POSITIONS_[index];
          RESULT_[index] = T_[h] * (D3_[h] + T_[h] * (D2_[h] + T_[h] * D1_[h] + D0_[h]));
      });
  };
};

struct coefficients_compute {
  template <typename T>
  void operator()(cudf::column_view const& t,
                  cudf::column_view const& y,
                  cudf::column_view const& prefixes,
                  cudf::mutable_column_view const& h,
                  cudf::mutable_column_view const& i,
                  cudf::mutable_column_view const& z,
                  cudf::mutable_column_view const& d3,
                  cudf::mutable_column_view const& d2,
                  cudf::mutable_column_view const& d1,
                  cudf::mutable_column_view const& d0,
                  rmm::mr::device_memory_resource *mr,
                  cudaStream_t stream) {
      const T* T_ = t.data<T>();
      const T* Y_ = y.data<T>();
      const int32_t* PREFIXES = prefixes.data<int32_t>();
      T* H_ = h.data<T>();
      T* I_ = i.data<T>();
      T* Z_ = z.data<T>();
      T* D3_ = d3.data<T>();
      T* D2_ = d2.data<T>();
      T* D1_ = d1.data<T>();
      T* D0_ = d0.data<T>();
      thrust::for_each(rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<int>(1),
        thrust::make_counting_iterator<int>(prefixes.size()),
        [T_, Y_, PREFIXES, H_, I_, Z_, D3_, D2_, D1_, D0_] __device__
        (int index) {
          int n = PREFIXES[index] - PREFIXES[index-1];
          int h = PREFIXES[index-1];
          int ci = 0;
          for(ci = 0 ; ci < n-1 ; ++ci) {
            T a = Y_[h+ci];
            T b = I_[h+ci] - H_[h+ci] * (Z_[h+ci+1] + 2 * Z_[h+ci]) / 6;
            T c = Z_[h+ci] / 2.0;
            T d = (Z_[h+ci+1] - Z_[h+ci]) / 6 * H_[h+ci];
            T t = T_[h+ci+1];
            D3_[h+ci] = d;
            D2_[h+ci] = c - 3 * d * t;
            D1_[h+ci] = b - t * (2*c - t * (3 * d));
            D0_[h+ci] = a - t * (b - t * (c - t * d)); // horners
          }
      });
  };
};

struct compute_spline_tridiagonals {
  template <typename T>
  void operator()(cudf::column_view const& t,
                  cudf::column_view const& y,
                  cudf::column_view const& prefixes,
                  cudf::mutable_column_view const& D,
                  cudf::mutable_column_view const& Dlu,
                  cudf::mutable_column_view const& u,
                  cudf::mutable_column_view const& h,
                  cudf::mutable_column_view const& i,
                  rmm::mr::device_memory_resource *mr,
                  cudaStream_t stream) {
      const T* T_ = t.data<T>();
      const T* Y_ = y.data<T>();
      const int32_t* PREFIXES = prefixes.data<int32_t>();
      T* D_ = D.data<T>();
      T* Dlu_ = Dlu.data<T>();
      T* U_ = u.data<T>();
      T* H_ = h.data<T>();
      T* I_ = i.data<T>();
      thrust::for_each(rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<int>(1),
        thrust::make_counting_iterator<int>(prefixes.size()),
        [T_, Y_, PREFIXES, D_, Dlu_, U_, H_, I_] __device__
        (int index) {
          int n = PREFIXES[index] - PREFIXES[index-1];
          int h = PREFIXES[index-1];
          int ci = 0;
          for(ci = 0 ; ci < n-1 ; ++ci) {
            H_[h + ci] = T_[h+ci+1] - T_[h+ci];
            I_[h + ci] = (Y_[h+ci+1] - Y_[h+ci]) / H_[h + ci];
          }
          for(ci = 0 ; ci < n-2 ; ++ci) {
            D_[h+ci+1] = (H_[h+ci+1]+H_[h+(n-2)-ci]) * 2;
            U_[h+ci+1] = (I_[h+ci+1] - I_[h+(n-2)-ci]) * 6;
          }
          for(ci = 0 ; ci < n-3 ; ++ci) {
            Dlu_[h+ci+1] = I_[h+ci+1];
          }
      });
  }
};

struct compute_splines {
//template<typename T, std::enable_if_t<std::is_floating_point<T>::value >* = nullptr>
  template<typename T>
    void operator()(cudf::column_view const& t, cudf::column_view const& y,
        cudf::column_view const& ids, cudf::column_view const& prefixes,
        cudf::mutable_column_view const& buffer,
        rmm::mr::device_memory_resource *mr,
        cudaStream_t stream)
      {
          T* BUFFER = buffer.data<T>();
          const T* TT = t.data<T>();
          const T* Y = y.data<T>();
          const int32_t* IDS = ids.data<int32_t>();
          const int32_t* PREFIXES = prefixes.data<int32_t>();
          int64_t size = y.size();
          CUDF_EXPECTS(buffer.size() == size* 4, "compute_splines bad input buffer");
          thrust::for_each(rmm::exec_policy(stream)->on(stream),
              thrust::make_counting_iterator<int>(1),
          thrust::make_counting_iterator<int>(static_cast<int>(prefixes.size())),
                           [TT, Y, IDS, PREFIXES, BUFFER] __device__
                           (int index) {
    int n = PREFIXES[index] - PREFIXES[index-1];
    int h = PREFIXES[index-1] * 4;
    int s_i = PREFIXES[index-1];
    int len_h = n-1;
    int i = h+len_h;
    int len_i = len_h;
    int v = i+len_i;
    int len_v = n-2;
    int u = v+len_v;
    int len_u = len_v;
    int ci = 0;
    for(ci = 0 ; ci < len_h ; ++ci) {
      BUFFER[h+ci] = TT[s_i+ci+1] - TT[s_i+ci];
      BUFFER[i+ci] = (Y[s_i+ci+1] - Y[s_i+ci]) / BUFFER[h+ci];
    }
    for(ci = 0 ; ci < len_v ; ++ci) {
      BUFFER[v+ci] = (BUFFER[h+ci+1]+BUFFER[h+len_h-1-ci]) * 2;
    }
    for(ci = 0 ; ci < len_u ; ++ci) {
      BUFFER[u+ci] = (BUFFER[i+ci+1] - BUFFER[i+len_i-1-ci]) * 6;
    } 
 });
      }
  /*
template<typename T, std::enable_if_t<!std::is_floating_point<T>::value >* = nullptr>
    void operator()(cudf::column_view const& t, cudf::column_view const& y,
        cudf::column_view const& ids, cudf::column_view const& prefixes,
        cudf::mutable_column_view const& buffer,
        rmm::mr::device_memory_resource *mr,
        cudaStream_t stream)
      {
          CUDF_FAIL("Non-floating point operation is not supported.");
      }
      */
};

} // anonymous namespace

namespace cuspatial
{

std::unique_ptr<cudf::column> cubicspline_interpolate(
    cudf::column_view points,
    cudf::column_view ids,
    cudf::table_view coefficients
)
{
    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();
    auto result = make_numeric_column(points.type(), points.size(), cudf::mask_state::UNALLOCATED, stream, mr);
    cudf::mutable_column_view temp = result->mutable_view();
    cudf::experimental::scalar_type_t<float> one;
    one.set_value(1.0);
    cudf::experimental::fill(temp, 0, result->size(), one);
    return result;
}

std::unique_ptr<cudf::experimental::table> cubicspline_full(
    cudf::column_view t,
    cudf::column_view y,
    cudf::column_view ids,
    cudf::column_view prefixes
)
{
    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();
    TPRINT(t, "t_");
    TPRINT(y, "y_");

    int64_t n = y.size();
    auto h_col = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
    auto i_col = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
    auto D_col = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
    auto Dlu_col = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
    auto Dll_col = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
    auto u_col = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
    cudf::mutable_column_view h_buffer = h_col->mutable_view();
    cudf::mutable_column_view i_buffer = i_col->mutable_view();
    cudf::mutable_column_view D_buffer = D_col->mutable_view();
    cudf::mutable_column_view Dlu_buffer = Dll_col->mutable_view();
    cudf::mutable_column_view Dll_buffer = Dll_col->mutable_view();
    cudf::mutable_column_view u_buffer = u_col->mutable_view();
    
    cudf::experimental::scalar_type_t<float> zero;
    zero.set_value(0.0);
    cudf::experimental::scalar_type_t<float> one;
    one.set_value(1.0);
    cudf::experimental::fill_in_place(h_buffer, 0, h_col->size(), zero);
    cudf::experimental::fill_in_place(i_buffer, 0, i_col->size(), zero);
    cudf::experimental::fill_in_place(D_buffer, 0, D_col->size(), one);
    cudf::experimental::fill_in_place(Dlu_buffer, 0, Dlu_col->size(), zero);
    cudf::experimental::fill_in_place(u_buffer, 0, u_col->size(), zero);
    
    TPRINT(h_buffer, "h_zero");
    TPRINT(D_buffer, "D_one");
    TPRINT(Dlu_buffer, "Dlu_zero");
    //cudf::experimental::type_dispatcher(y.type(), compute_spline_tridiagonals{}, t, y, prefixes, D_buffer, Dlu_buffer, h_buffer, i_buffer, cv_result, mr, stream);
    compute_spline_tridiagonals comp_tri;
    comp_tri.operator()<float>(t, y, prefixes, D_buffer, Dlu_buffer, u_buffer, h_buffer, i_buffer, mr, stream);

    TPRINT(h_buffer, "h_i");
    TPRINT(i_buffer, "i_i");
    TPRINT(D_buffer, "D_i");
    TPRINT(Dlu_buffer, "Dlu_i");
    TPRINT(u_buffer, "u_i");

    // cusparse solve n length m tridiagonal systems
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
    cusparseStatus = cusparseSgtsv2StridedBatch_bufferSizeExt(
        handle,
        y.size()/(prefixes.size()-1),
        Dll_buffer.data<float>(),
        D_buffer.data<float>(),
        Dlu_buffer.data<float>(),
        u_buffer.data<float>(),
        prefixes.size()-1,
        y.size()/(prefixes.size()-1),
        &pBufferSize
    );
    HANDLE_CUSPARSE_STATUS(cusparseStatus);
    rmm::device_vector<float> pBuffer(pBufferSize);
    cusparseStatus = cusparseSgtsv2StridedBatch(
        handle,
        y.size()/(prefixes.size()-1),
        Dll_buffer.data<float>(),
        D_buffer.data<float>(),
        Dlu_buffer.data<float>(),
        u_buffer.data<float>(),
        prefixes.size()-1,
        y.size()/(prefixes.size()-1),
        thrust::raw_pointer_cast(pBuffer.data())
    );
    HANDLE_CUSPARSE_STATUS(cusparseStatus);
    cusparseStatus = cusparseDestroy(handle);
    HANDLE_CUSPARSE_STATUS(cusparseStatus);
    TPRINT(u_buffer, "u_buffer");

    // Finally, compute coefficients via Horner's scheme
    auto d3_col = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
    auto d2_col = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
    auto d1_col = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
    auto d0_col = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
    cudf::mutable_column_view d3 = d3_col->mutable_view();
    cudf::mutable_column_view d2 = d2_col->mutable_view();
    cudf::mutable_column_view d1 = d1_col->mutable_view();
    cudf::mutable_column_view d0 = d0_col->mutable_view();
    cudf::experimental::fill(d3, 0, d3_col->size(), zero);
    cudf::experimental::fill(d2, 0, d2_col->size(), zero);
    cudf::experimental::fill(d1, 0, d1_col->size(), zero);
    cudf::experimental::fill(d0, 0, d0_col->size(), zero);

    coefficients_compute coefs;
    coefs.operator()<float>(t, y, prefixes, h_buffer, i_buffer, u_buffer, d3, d2, d1, d0, mr, stream);

    TPRINT(d3, "d3");
    TPRINT(d2, "d2");
    TPRINT(d1, "d1");
    TPRINT(d0, "d0");
    
    // Place d3..0 into a table and return
    std::vector<std::unique_ptr<cudf::column>> table;
    table.push_back(std::move(d3_col));
    table.push_back(std::move(d2_col));
    table.push_back(std::move(d1_col));
    table.push_back(std::move(d0_col));
    std::unique_ptr<cudf::experimental::table> result = std::make_unique<cudf::experimental::table>(move(table));
    return result;
}
}
