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

#include <cuspatial/cubic_spline.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include "cusparse.h"
#include <cuspatial/utility.hpp>

namespace { // anonymous

struct parallel_search {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, void>
  operator()(cudf::column_view const& t,
                  cudf::column_view const& curve_ids,
                  cudf::column_view const& prefixes,
                  cudf::column_view const& query_coords,
                  cudf::mutable_column_view const& result,
                  rmm::mr::device_memory_resource *mr,
                  cudaStream_t stream) {
      const T* T_ = t.data<T>();
      const int32_t* CURVE_IDS_ = curve_ids.data<int32_t>();
      const int32_t* PREFIXES_ = prefixes.data<int32_t>();
      const T* QUERY_COORDS_ = query_coords.data<T>();
      T* RESULT_ = result.data<T>();
      thrust::for_each(rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(query_coords.size()),
        [T_, CURVE_IDS_, PREFIXES_, QUERY_COORDS_, RESULT_] __device__
        (int index) {
          int curve = CURVE_IDS_[index];
          int len = PREFIXES_[curve+1] - PREFIXES_[curve];
          int h = PREFIXES_[curve];
          int dh = PREFIXES_[curve] - (curve);
          // O(n) search, can do log(n) easily
          for(int32_t i = 0 ; i < len ; ++i) {
            if((T_[h+i]+0.0001 - QUERY_COORDS_[index]) > 0.00001) {
              RESULT_[index] = dh+i-1;
              if(i == 0) RESULT_[index] = index-curve;
              return;
            }
          }
          // TODO: Important failure case:
          // This will use the final set of coefficients
          // for t_ values that are outside of the original
          // interpolation range.
          RESULT_[index] = h+len - 2;
      });
  };
  template <typename T>
  std::enable_if_t<not std::is_floating_point<T>::value, void>
  operator()(cudf::column_view const& t,
                  cudf::column_view const& curve_ids,
                  cudf::column_view const& prefixes,
                  cudf::column_view const& query_coords,
                  cudf::mutable_column_view const& result,
                  rmm::mr::device_memory_resource *mr,
                  cudaStream_t stream) {
      CUDF_FAIL("Non-floating point operation is not supported.");
  }
};

struct interpolate {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, void>
  operator()(cudf::column_view const& t,
                  cudf::column_view const& ids,
                  cudf::column_view const& coef_indexes,
                  cudf::table_view const& coefficients,
                  cudf::mutable_column_view const& result,
                  rmm::mr::device_memory_resource *mr,
                  cudaStream_t stream) {
      const T* T_ = t.data<T>();
      const int32_t* IDS_ = ids.data<int32_t>();
      const int32_t* COEF_INDEXES_ = coef_indexes.data<int32_t>();
      const T* D3_ = coefficients.column(3).data<T>();
      const T* D2_ = coefficients.column(2).data<T>();
      const T* D1_ = coefficients.column(1).data<T>();
      const T* D0_ = coefficients.column(0).data<T>();
      T* RESULT_ = result.data<T>();
      thrust::for_each(rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(t.size()),
        [T_, IDS_, COEF_INDEXES_, D3_, D2_, D1_, D0_, RESULT_] __device__
        (int index) {
          int h = RESULT_[index];
          RESULT_[index] = D3_[h] + T_[index] * (D2_[h] + T_[index] * (D1_[h] + (T_[index] * D0_[h])));
      });
  };
  template <typename T>
  std::enable_if_t<not std::is_floating_point<T>::value, void>
  operator()(cudf::column_view const& t,
                  cudf::column_view const& ids,
                  cudf::column_view const& coef_indexes,
                  cudf::table_view const& coefficients,
                  cudf::mutable_column_view const& result,
                  rmm::mr::device_memory_resource *mr,
                  cudaStream_t stream) {
      CUDF_FAIL("Non-floating point operation is not supported.");
  }
};

struct coefficients_compute {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, void>
  operator()(cudf::column_view const& t,
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
          int dh = PREFIXES[index-1] - (index-1);
          int ci = 0;
          for(ci = 0 ; ci < n-1 ; ++ci) {
            T a = Y_[h+ci];
            T b = I_[h+ci] - H_[h+ci] * (Z_[h+ci+1] + 2 * Z_[h+ci]) / 6;
            T c = Z_[h+ci] / 2.0;
            T d = (Z_[h+ci+1] - Z_[h+ci]) / 6 * H_[h+ci];
            T t = T_[h+ci];
            D3_[dh+ci] = d;
            D2_[dh+ci] = c - 3 * d * t;
            D1_[dh+ci] = b - t * (2*c - t * (3 * d));
            D0_[dh+ci] = a - t * (b - t * (c - t * d)); // horners
          }
      });
  };
  template<typename T>
  std::enable_if_t<not std::is_floating_point<T>::value, void>
  operator()(cudf::column_view const& t,
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
      CUDF_FAIL("Non-floating point operation is not supported.");
  }
};

struct compute_spline_tridiagonals {
  template<typename T>
  std::enable_if_t<std::is_floating_point<T>::value, void>
  operator()(cudf::column_view const& t,
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
  template<typename T>
  std::enable_if_t<not std::is_floating_point<T>::value, void>
  operator()(cudf::column_view const& t,
                  cudf::column_view const& y,
                  cudf::column_view const& prefixes,
                  cudf::mutable_column_view const& D,
                  cudf::mutable_column_view const& Dlu,
                  cudf::mutable_column_view const& u,
                  cudf::mutable_column_view const& h,
                  cudf::mutable_column_view const& i,
                  rmm::mr::device_memory_resource *mr,
                  cudaStream_t stream) {
      CUDF_FAIL("Non-floating point operation is not supported.");
  }
};

} // anonymous namespace

namespace cuspatial
{

std::unique_ptr<cudf::column> cubicspline_interpolate(
    cudf::column_view const& query_points,
    cudf::column_view const& curve_ids,
    cudf::column_view const& prefixes,
    cudf::column_view const& source_points,
    cudf::table_view const& coefficients,
    rmm::mr::device_memory_resource *mr,
    cudaStream_t stream
)
{
    auto result = make_numeric_column(query_points.type(), query_points.size(), cudf::mask_state::UNALLOCATED, stream, mr);
    cudf::mutable_column_view search_result = result->mutable_view();
    
    //auto search_result = cudf::experimental::type_dispatcher(query_points.type(), parallel_search{}, query_points, curve_ids, prefixes, source_points, mr, stream);
    cudf::experimental::type_dispatcher(query_points.type(), parallel_search{}, query_points, curve_ids, prefixes, source_points, search_result, mr, stream);
    TPRINT(search_result, "parallel_search_");
    TPRINT(query_points, "query_points_");
    TPRINT(curve_ids, "curve_ids_");
    TPRINT(prefixes, "prefixes_");

    cudf::experimental::type_dispatcher(query_points.type(), interpolate{}, query_points, curve_ids, prefixes, coefficients, search_result, mr, stream);
    TPRINT(query_points, "query_points_");
    TPRINT(curve_ids, "curve_ids_");
    TPRINT(prefixes, "prefixes_");
    TPRINT(search_result, "interpolate_");
    return result;
}

std::unique_ptr<cudf::column> cubicspline_interpolate_default(
    cudf::column_view const& query_points,
    cudf::column_view const& curve_ids,
    cudf::column_view const& prefixes,
    cudf::column_view const& source_points,
    cudf::table_view const& coefficients
)
{
  return cubicspline_interpolate(query_points, curve_ids, prefixes, source_points, coefficients, rmm::mr::get_default_resource(), 0);
}

std::unique_ptr<cudf::experimental::table> cubicspline_coefficients(
    cudf::column_view const& t,
    cudf::column_view const& y,
    cudf::column_view const& ids,
    cudf::column_view const& prefixes,
    rmm::mr::device_memory_resource *mr,
    cudaStream_t stream
)
{
    TPRINT(t, "t_");
    TPRINT(y, "y_");
    TPRINT(ids, "ids");
    TPRINT(prefixes, "prefixes");

    int64_t n = y.size();
    auto h_col = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
    auto i_col = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
    auto D_col = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
    auto Dlu_col = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
    auto Dll_col = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
    auto u_col = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
    auto h_buffer = h_col->mutable_view();
    auto i_buffer = i_col->mutable_view();
    auto D_buffer = D_col->mutable_view();
    auto Dlu_buffer = Dll_col->mutable_view();
    auto Dll_buffer = Dll_col->mutable_view();
    auto u_buffer = u_col->mutable_view();

    auto zero = cudf::numeric_scalar<float>(0.0);
    auto one = cudf::numeric_scalar<float>(1.0);
    cudf::experimental::fill_in_place(h_buffer, 0, h_col->size(), zero);
    cudf::experimental::fill_in_place(i_buffer, 0, i_col->size(), zero);
    cudf::experimental::fill_in_place(D_buffer, 0, D_col->size(), one);
    cudf::experimental::fill_in_place(Dlu_buffer, 0, Dlu_col->size(), zero);
    cudf::experimental::fill_in_place(u_buffer, 0, u_col->size(), zero);
    
    TPRINT(h_buffer, "h_zero");
    TPRINT(D_buffer, "D_one");
    TPRINT(Dlu_buffer, "Dlu_zero");
    cudf::experimental::type_dispatcher(y.type(), compute_spline_tridiagonals{}, t, y, prefixes, D_buffer, Dlu_buffer, u_buffer, h_buffer, i_buffer, mr, stream);

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
    detail::HANDLE_CUSPARSE_STATUS(cusparseStatus);
    size_t pBufferSize;
    int32_t batchStride = y.size() / (prefixes.size() - 1);
    int32_t batchSize = batchStride;
    cusparseStatus = cusparseSgtsv2StridedBatch_bufferSizeExt(
        handle,
        batchSize,
        Dll_buffer.data<float>(),
        D_buffer.data<float>(),
        Dlu_buffer.data<float>(),
        u_buffer.data<float>(),
        prefixes.size()-1,
        batchStride,
        &pBufferSize
    );
    detail::HANDLE_CUSPARSE_STATUS(cusparseStatus);
    rmm::device_vector<float> pBuffer(pBufferSize);
    cusparseStatus = cusparseSgtsv2StridedBatch(
        handle,
        batchSize,
        Dll_buffer.data<float>(),
        D_buffer.data<float>(),
        Dlu_buffer.data<float>(),
        u_buffer.data<float>(),
        prefixes.size()-1,
        batchStride,
        thrust::raw_pointer_cast(pBuffer.data().get())
    );
    detail::HANDLE_CUSPARSE_STATUS(cusparseStatus);
    cusparseStatus = cusparseDestroy(handle);
    detail::HANDLE_CUSPARSE_STATUS(cusparseStatus);

    int dn = n - (prefixes.size()-1);
    // Finally, compute coefficients via Horner's scheme
    auto d3_col = make_numeric_column(y.type(), dn, cudf::mask_state::UNALLOCATED, stream, mr);
    auto d2_col = make_numeric_column(y.type(), dn, cudf::mask_state::UNALLOCATED, stream, mr);
    auto d1_col = make_numeric_column(y.type(), dn, cudf::mask_state::UNALLOCATED, stream, mr);
    auto d0_col = make_numeric_column(y.type(), dn, cudf::mask_state::UNALLOCATED, stream, mr);
    auto d3 = d3_col->mutable_view();
    auto d2 = d2_col->mutable_view();
    auto d1 = d1_col->mutable_view();
    auto d0 = d0_col->mutable_view();

    cudf::experimental::type_dispatcher(y.type(), coefficients_compute{}, t, y, prefixes, h_buffer, i_buffer, u_buffer, d3, d2, d1, d0, mr, stream);

    TPRINT(h_buffer, "h_buffer_");
    TPRINT(i_buffer, "i_buffer_");
    TPRINT(u_buffer, "u_buffer_");

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
std::unique_ptr<cudf::experimental::table> cubicspline_coefficients_default(
    cudf::column_view const& t,
    cudf::column_view const& y,
    cudf::column_view const& ids,
    cudf::column_view const& prefixes
)
{
  return cubicspline_coefficients(t, y, ids, prefixes, rmm::mr::get_default_resource(), 0);
}
}
