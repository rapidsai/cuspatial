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

// This functor performs one linear search for each input point in query_coords
struct parallel_search {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>>
  operator()(cudf::column_view const& t,
             cudf::column_view const& curve_ids,
             cudf::column_view const& prefixes,
             cudf::column_view const& query_coords,
             rmm::mr::device_memory_resource *mr,
             cudaStream_t stream) {
      const T* p_t = t.data<T>();
      const int32_t* p_curve_ids = curve_ids.data<int32_t>();
      const int32_t* p_prefixes = prefixes.data<int32_t>();
      const T* p_query_coords = query_coords.data<T>();
      auto result = cudf::make_numeric_column(curve_ids.type(), t.size(),
              cudf::mask_state::UNALLOCATED, stream, mr);
      int32_t* p_result = result->mutable_view().data<int32_t>();
      thrust::for_each(rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(query_coords.size()),
        [p_t, p_curve_ids, p_prefixes, p_query_coords, p_result] __device__
        (int index) {
          int curve = p_curve_ids[index];
          int len = p_prefixes[curve+1] - p_prefixes[curve];
          int h = p_prefixes[curve];
          int dh = p_prefixes[curve] - (curve);
          // O(n) search, can do log(n) easily
          for(int32_t i = 0 ; i < len ; ++i) {
            if((p_t[h+i]+0.0001 - p_query_coords[index]) > 0.00001) {
              p_result[index] = dh+i-1;
              if(i == 0) p_result[index] = index-curve;
              return;
            }
          }
          // TODO: Important failure case:
          // This will use the final set of coefficients
          // for t_ values that are outside of the original
          // interpolation range.
          p_result[index] = h+len - 2;
      });
      return result;
  };
  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>>
  operator()(Args&&... args) {
      CUDF_FAIL("Non-floating point operation is not supported.");
  }
};

// This functor simply computes the interpolation of each coordinate `t[i]`
// using the coefficients from row `coef_indices[i]`.
struct interpolate {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>>
  operator()(cudf::column_view const& t,
             cudf::column_view const& ids,
             cudf::column_view const& coef_indices,
             cudf::table_view const& coefficients,
             rmm::mr::device_memory_resource *mr,
             cudaStream_t stream) {
      const T* p_t = t.data<T>();
      const int32_t* p_ids = ids.data<int32_t>();
      const int32_t* p_coef_indices = coef_indices.data<int32_t>();
      const T* p_d3 = coefficients.column(3).data<T>();
      const T* p_d2 = coefficients.column(2).data<T>();
      const T* p_d1 = coefficients.column(1).data<T>();
      const T* p_d0 = coefficients.column(0).data<T>();
      auto result = cudf::make_numeric_column(t.type(), t.size(),
              cudf::mask_state::UNALLOCATED, stream, mr);
      T* p_result = result->mutable_view().data<T>();
      thrust::for_each(rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(t.size()),
        [p_t, p_ids, p_coef_indices, p_d3, p_d2, p_d1, p_d0, p_result] __device__
        (int index) {
          int h = p_coef_indices[index];
          p_result[index] = p_d3[h] + p_t[index] * (p_d2[h] + p_t[index] * (p_d1[h] + (p_t[index] * p_d0[h])));
      });
      return result;
  };
  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>>
  operator()(Args&&... args) {
      CUDF_FAIL("Non-floating point operation is not supported.");
  }
};

// This functor computes the coefficients table for the cubic hermite spline
// specified by the inputs `t` and `y`.
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
      const T* p_t = t.data<T>();
      const T* p_y = y.data<T>();
      const int32_t* p_prefixes = prefixes.data<int32_t>();
      T* p_h = h.data<T>();
      T* p_i = i.data<T>();
      T* p_z = z.data<T>();
      T* p_d3 = d3.data<T>();
      T* p_d2 = d2.data<T>();
      T* p_d1 = d1.data<T>();
      T* p_d0 = d0.data<T>();
      thrust::for_each(rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<int>(1),
        thrust::make_counting_iterator<int>(prefixes.size()),
        [p_t, p_y, p_prefixes, p_h, p_i, p_z, p_d3, p_d2, p_d1, p_d0] __device__
        (int index) {
          int n = p_prefixes[index] - p_prefixes[index-1];
          int h = p_prefixes[index-1];
          int dh = p_prefixes[index-1] - (index-1);
          int ci = 0;
          for(ci = 0 ; ci < n-1 ; ++ci) {
            T a = p_y[h+ci];
            T b = p_i[h+ci] - p_h[h+ci] * (p_z[h+ci+1] + 2 * p_z[h+ci]) / 6;
            T c = p_z[h+ci] / 2.0;
            T d = (p_z[h+ci+1] - p_z[h+ci]) / 6 * p_h[h+ci];
            T t = p_t[h+ci];
            p_d3[dh+ci] = d;
            p_d2[dh+ci] = c - 3 * d * t;
            p_d1[dh+ci] = b - t * (2*c - t * (3 * d));
            p_d0[dh+ci] = a - t * (b - t * (c - t * d)); // horners
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

// Computes the diagonal `D` of a large sparse matrix, and also the upper and
// lower diagonals `Dlu`, which in this case are equal.
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
      const T* p_t = t.data<T>();
      const T* p_y = y.data<T>();
      const int32_t* p_prefixes = prefixes.data<int32_t>();
      T* p_d = D.data<T>();
      T* p_dlu = Dlu.data<T>();
      T* p_u = u.data<T>();
      T* p_h = h.data<T>();
      T* p_i = i.data<T>();
      thrust::for_each(rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<int>(1),
        thrust::make_counting_iterator<int>(prefixes.size()),
        [p_t, p_y, p_prefixes, p_d, p_dlu, p_u, p_h, p_i] __device__
        (int index) {
          int n = p_prefixes[index] - p_prefixes[index-1];
          int h = p_prefixes[index-1];
          int ci = 0;
          for(ci = 0 ; ci < n-1 ; ++ci) {
            p_h[h + ci] = p_t[h+ci+1] - p_t[h+ci];
            p_i[h + ci] = (p_y[h+ci+1] - p_y[h+ci]) / p_h[h + ci];
          }
          for(ci = 0 ; ci < n-2 ; ++ci) {
            p_d[h+ci+1] = (p_h[h+ci+1]+p_h[h+(n-2)-ci]) * 2;
            p_u[h+ci+1] = (p_i[h+ci+1] - p_i[h+(n-2)-ci]) * 6;
          }
          for(ci = 0 ; ci < n-3 ; ++ci) {
            p_dlu[h+ci+1] = p_i[h+ci+1];
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

namespace detail
{

// Computes f(query_points) by calculating which set of coefficients should be
// used and then interpolating them. See cubic_spline.hpp
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
    auto coefficient_indices = cudf::experimental::type_dispatcher(query_points.type(), parallel_search{}, query_points, curve_ids, prefixes, source_points, mr, stream);
    //TPRINT(coefficient_indices->mutable_view(), "parallel_search_");
    //TPRINT(query_points, "query_points_");
    //TPRINT(curve_ids, "curve_ids_");
    //TPRINT(prefixes, "prefixes_");

    auto result = cudf::experimental::type_dispatcher(query_points.type(), interpolate{}, query_points, curve_ids, coefficient_indices->view(), coefficients, mr, stream);
    //TPRINT(query_points, "query_points_");
    //TPRINT(curve_ids, "curve_ids_");
    //TPRINT(prefixes, "prefixes_");
    //cudf::column_view result_view = result->view();
    ////TPRINT(result_view, "interpolate_");
    return result;
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
    //rmm::device_vector<float>::iterator t_rd = rmm::device_vector<float>(t.data<float>());
    //TPRINT(t, "t_");
    //TPRINT(y, "y_");
    //TPRINT(ids, "ids");
    //TPRINT(prefixes, "prefixes");

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
    
    //TPRINT(h_buffer, "h_zero");
    //TPRINT(D_buffer, "D_one");
    //TPRINT(Dlu_buffer, "Dlu_zero");
    cudf::experimental::type_dispatcher(y.type(), compute_spline_tridiagonals{}, t, y, prefixes, D_buffer, Dlu_buffer, u_buffer, h_buffer, i_buffer, mr, stream);

    //TPRINT(h_buffer, "h_i");
    //TPRINT(i_buffer, "i_i");
    //TPRINT(D_buffer, "D_i");
    //TPRINT(Dlu_buffer, "Dlu_i");
    //TPRINT(u_buffer, "u_i");

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
        pBuffer.data().get()
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

    //TPRINT(h_buffer, "h_buffer_");
    //TPRINT(i_buffer, "i_buffer_");
    //TPRINT(u_buffer, "u_buffer_");

    //TPRINT(d3, "d3");
    //TPRINT(d2, "d2");
    //TPRINT(d1, "d1");
    //TPRINT(d0, "d0");
    
    // Place d3..0 into a table and return
    std::vector<std::unique_ptr<cudf::column>> table;
    table.push_back(std::move(d3_col));
    table.push_back(std::move(d2_col));
    table.push_back(std::move(d1_col));
    table.push_back(std::move(d0_col));
    std::unique_ptr<cudf::experimental::table> result = std::make_unique<cudf::experimental::table>(move(table));
    return result;
}

} // namespace detail

// Calls the interpolate function using default memory resources.
std::unique_ptr<cudf::column> cubicspline_interpolate(
    cudf::column_view const& query_points,
    cudf::column_view const& curve_ids,
    cudf::column_view const& prefixes,
    cudf::column_view const& source_points,
    cudf::table_view const& coefficients
)
{
  return cuspatial::detail::cubicspline_interpolate(query_points, curve_ids, prefixes, source_points, coefficients, rmm::mr::get_default_resource(), 0);
}

// Calls the coeffiecients  function using default memory resources.
std::unique_ptr<cudf::experimental::table> cubicspline_coefficients(
    cudf::column_view const& t,
    cudf::column_view const& y,
    cudf::column_view const& ids,
    cudf::column_view const& prefixes
)
{
  return cuspatial::detail::cubicspline_coefficients(t, y, ids, prefixes, rmm::mr::get_default_resource(), 0);
}

} // namespace cuspatial
