/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>

#include <cuspatial/cubic_spline.hpp>
#include <cuspatial/cusparse_error.hpp>
#include <cuspatial/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cusparse.h>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace {  // anonymous

// This functor performs one linear search for each input point in query_coords
struct parallel_search {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    cudf::column_view const& search_coords,
    cudf::column_view const& curve_ids,
    cudf::column_view const& prefixes,
    cudf::column_view const& query_coords,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    const T SEARCH_OFFSET{0.0001};
    const T QUERY_OFFSET{0.00001};

    const T* p_search_coords   = search_coords.data<T>();
    const int32_t* p_curve_ids = curve_ids.data<int32_t>();
    const int32_t* p_prefixes  = prefixes.data<int32_t>();
    const T* p_query_coords    = query_coords.data<T>();
    auto result                = cudf::make_numeric_column(
      curve_ids.type(), search_coords.size(), cudf::mask_state::UNALLOCATED, stream, mr);
    int32_t* p_result = result->mutable_view().data<int32_t>();
    thrust::for_each(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<int32_t>(0),
      thrust::make_counting_iterator<int32_t>(search_coords.size()),
      [=] __device__(int32_t index) {
        int32_t curve                    = p_curve_ids[index];
        int32_t len                      = p_prefixes[curve + 1] - p_prefixes[curve];
        int32_t query_coord_offset       = p_prefixes[curve];
        int32_t coefficient_table_offset = p_prefixes[curve] - curve;
        // O(n) search, can do log(n) easily
        const T search_coord = p_search_coords[index] + SEARCH_OFFSET;
        for (int32_t i = 1; i < len; ++i) {
          if ((search_coord < p_query_coords[query_coord_offset + i] + QUERY_OFFSET)) {
            p_result[index] = coefficient_table_offset + i - 1;
            return;
          }
        }
        // NOTE: Important failure case:
        // This will use the final set of coefficients
        // for t_ values that are outside of the original
        // interpolation range.
        p_result[index] = coefficient_table_offset + len - 2;
      });
    return result;
  }

  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Non-floating point operation is not supported.");
  }
};

// This functor simply computes the interpolation of each coordinate `t[i]`
// using the coefficients from row `coef_indices[i]`.
struct interpolate {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    cudf::column_view const& t,
    cudf::column_view const& coef_indices,
    cudf::table_view const& coefficients,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    const T* p_t                  = t.data<T>();
    const int32_t* p_coef_indices = coef_indices.data<int32_t>();
    const T* p_d3                 = coefficients.column(3).data<T>();
    const T* p_d2                 = coefficients.column(2).data<T>();
    const T* p_d1                 = coefficients.column(1).data<T>();
    const T* p_d0                 = coefficients.column(0).data<T>();
    auto result =
      cudf::make_numeric_column(t.type(), t.size(), cudf::mask_state::UNALLOCATED, stream, mr);
    T* p_result = result->mutable_view().data<T>();
    thrust::for_each(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<int32_t>(0),
                     thrust::make_counting_iterator<int32_t>(t.size()),
                     [=] __device__(int32_t index) {
                       int32_t h = p_coef_indices[index];
                       p_result[index] =
                         p_d3[h] +
                         p_t[index] * (p_d2[h] + p_t[index] * (p_d1[h] + (p_t[index] * p_d0[h])));
                     });
    return result;
  };
  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Non-floating point operation is not supported.");
  }
};

// This functor computes the coefficients table for the cubic hermite spline
// specified by the inputs `t` and `y`.
struct coefficients_compute {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, void> operator()(
    cudf::column_view const& t,
    cudf::column_view const& y,
    cudf::column_view const& prefixes,
    cudf::mutable_column_view const& h,
    cudf::mutable_column_view const& i,
    cudf::mutable_column_view const& z,
    cudf::mutable_column_view const& d3,
    cudf::mutable_column_view const& d2,
    cudf::mutable_column_view const& d1,
    cudf::mutable_column_view const& d0,
    rmm::cuda_stream_view stream)
  {
    const T* p_t              = t.data<T>();
    const T* p_y              = y.data<T>();
    const int32_t* p_prefixes = prefixes.data<int32_t>();
    T* p_h                    = h.data<T>();
    T* p_i                    = i.data<T>();
    T* p_z                    = z.data<T>();
    T* p_d3                   = d3.data<T>();
    T* p_d2                   = d2.data<T>();
    T* p_d1                   = d1.data<T>();
    T* p_d0                   = d0.data<T>();
    thrust::for_each(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<int32_t>(1),
      thrust::make_counting_iterator<int32_t>(prefixes.size()),
      [p_t, p_y, p_prefixes, p_h, p_i, p_z, p_d3, p_d2, p_d1, p_d0] __device__(int32_t index) {
        int32_t n  = p_prefixes[index] - p_prefixes[index - 1];
        int32_t h  = p_prefixes[index - 1];
        int32_t dh = p_prefixes[index - 1] - (index - 1);
        int32_t ci = 0;
        for (ci = 0; ci < n - 1; ++ci) {
          T a           = p_y[h + ci];
          T b           = p_i[h + ci] - p_h[h + ci] * (p_z[h + ci + 1] + 2 * p_z[h + ci]) / 6;
          T c           = p_z[h + ci] / 2.0;
          T d           = (p_z[h + ci + 1] - p_z[h + ci]) / 6 * p_h[h + ci];
          T t           = p_t[h + ci];
          p_d3[dh + ci] = d;
          p_d2[dh + ci] = c - 3 * d * t;
          p_d1[dh + ci] = b - t * (2 * c - t * (3 * d));
          p_d0[dh + ci] = a - t * (b - t * (c - t * d));  // horners
        }
      });
  }

  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, void> operator()(Args&&...)
  {
    CUSPATIAL_FAIL("Non-floating point operation is not supported.");
  }
};

// Computes the diagonal `D` of a large sparse matrix, and also the upper and
// lower diagonals `Dlu`, which in this case are equal.
struct compute_spline_tridiagonals {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, void> operator()(
    cudf::column_view const& t,
    cudf::column_view const& y,
    cudf::column_view const& prefixes,
    cudf::mutable_column_view const& D,
    cudf::mutable_column_view const& Dlu,
    cudf::mutable_column_view const& u,
    cudf::mutable_column_view const& h,
    cudf::mutable_column_view const& i,
    rmm::cuda_stream_view stream)
  {
    const T* p_t              = t.data<T>();
    const T* p_y              = y.data<T>();
    const int32_t* p_prefixes = prefixes.data<int32_t>();
    T* p_d                    = D.data<T>();
    T* p_dlu                  = Dlu.data<T>();
    T* p_u                    = u.data<T>();
    T* p_h                    = h.data<T>();
    T* p_i                    = i.data<T>();
    thrust::for_each(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<int32_t>(1),
                     thrust::make_counting_iterator<int32_t>(prefixes.size()),
                     [p_t, p_y, p_prefixes, p_d, p_dlu, p_u, p_h, p_i] __device__(int32_t index) {
                       int32_t n  = p_prefixes[index] - p_prefixes[index - 1];
                       int32_t h  = p_prefixes[index - 1];
                       int32_t ci = 0;
                       for (ci = 0; ci < n - 1; ++ci) {
                         p_h[h + ci] = p_t[h + ci + 1] - p_t[h + ci];
                         p_i[h + ci] = (p_y[h + ci + 1] - p_y[h + ci]) / p_h[h + ci];
                       }
                       for (ci = 0; ci < n - 2; ++ci) {
                         p_d[h + ci + 1] = (p_h[h + ci + 1] + p_h[h + (n - 2) - ci]) * 2;
                         p_u[h + ci + 1] = (p_i[h + ci + 1] - p_i[h + (n - 2) - ci]) * 6;
                       }
                       for (ci = 0; ci < n - 3; ++ci) {
                         p_dlu[h + ci + 1] = p_i[h + ci + 1];
                       }
                     });
  }

  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, void> operator()(Args&&...)
  {
    CUSPATIAL_FAIL("Non-floating point operation is not supported.");
  }
};

}  // anonymous namespace

namespace cuspatial {

namespace detail {

/**
 * @brief Finds the lower interpolant position of query_points from a set of
 * interpolation independent variables.
 *
 * @param[in] query_points column of coordinate values to be interpolated.
 * @param[in] spline_ids ids that identify the spline to interpolate each
 * coordinate into.
 * @param[in] offsets int32 column of offsets of the source_points.
 * This is used to calculate which values from the coefficients are
 * used for each interpolation.
 * @param[in] source_points column of the original `t` values used
 * to compute the coefficients matrix.  These source points are used to
 * identify which specific spline a given query_point is interpolated with.
 * cubicspline_coefficients.
 * @param[in] mr the optional caller specified RMM memory resource
 * @param[in] stream the optional caller specified cudaStream
 *
 * @return cudf::column of size equal to query points, one index position
 * of the first source_point mapped by offsets that is smaller than each
 * query point.
 **/
std::unique_ptr<cudf::column> find_coefficient_indices(cudf::column_view const& query_points,
                                                       cudf::column_view const& curve_ids,
                                                       cudf::column_view const& prefixes,
                                                       cudf::column_view const& source_points,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::mr::device_memory_resource* mr)
{
  auto coefficient_indices = cudf::type_dispatcher(query_points.type(),
                                                   parallel_search{},
                                                   query_points,
                                                   curve_ids,
                                                   prefixes,
                                                   source_points,
                                                   stream,
                                                   mr);
  return coefficient_indices;
}

/**
 * @brief Compute cubic interpolations of a set of points based on their
 * ids and a coefficient matrix.
 *
 * @param[in] query_points column of coordinate values to be interpolated.
 * @param[in] spline_ids ids that identift the spline to interpolate each
 * coordinate into.
 * @param[in] offsets int32 column of offset of the source_points.
 * This is used to calculate which values from the coefficients are
 * used for each interpolation.
 * @param[in] source_points column of the original `t` values used
 * to compute the coefficients matrix.  These source points are used to
 * identify which specific spline a given query_point is interpolated with.
 * @param[in] coefficients table of spline coefficients produced by
 * cubicspline_coefficients.
 * @param[in] mr the optional caller specified RMM memory resource
 * @param[in] stream the optional caller specified cudaStream
 *
 * @return cudf::column `y` coordinates interpolated from `x` and `coefs`.
 **/
std::unique_ptr<cudf::column> cubicspline_interpolate(cudf::column_view const& query_points,
                                                      cudf::column_view const& curve_ids,
                                                      cudf::column_view const& prefixes,
                                                      cudf::column_view const& source_points,
                                                      cudf::table_view const& coefficients,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::mr::device_memory_resource* mr)
{
  auto coefficient_indices = cudf::type_dispatcher(query_points.type(),
                                                   parallel_search{},
                                                   query_points,
                                                   curve_ids,
                                                   prefixes,
                                                   source_points,
                                                   stream,
                                                   mr);

  auto result = cudf::type_dispatcher(query_points.type(),
                                      interpolate{},
                                      query_points,
                                      coefficient_indices->view(),
                                      coefficients,
                                      stream,
                                      mr);

  return result;
}

/**
 * @brief Create a table of cubic spline coefficients from columns of coordinates.
 *
 * Computes coefficients for a natural cubic spline similar to the method
 * found on http://mathworld.wolfram.com/CubicSpline.html .
 *
 * The input data arrays `t` and `y` contain the vertices of many concatenated
 * splines.
 *
 * Currently, all input splines must be the same length. The minimum supported
 * length is 5.
 *
 * @note Ids should be prefixed with a 0, even when only a single spline
 * is fit, ids will be {0, 0}
 *
 * @param[in] t column_view of independent coordinates for fitting splines
 * @param[in] y column_view of dependent variables to be fit along t axis
 * @param[in] ids of incoming coordinate sets
 * @param[in] offsets the exclusive scan of the spline sizes, prefixed by
 * 0. For example, for 3 splines of 5 vertices each, the offsets input array
 * is {0, 5, 10, 15}.
 * @param[in] mr the optional caller specified RMM memory resource
 * @param[in] stream the optional caller specified cudaStream
 *
 * @return cudf::table_view of coefficients for spline interpolation. The size
 * of the table is ((M-n), 4) where M is `t.size()` and and n is
 * `ids.size()-1`.
 **/
std::unique_ptr<cudf::table> cubicspline_coefficients(cudf::column_view const& t,
                                                      cudf::column_view const& y,
                                                      cudf::column_view const& ids,
                                                      cudf::column_view const& prefixes,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::mr::device_memory_resource* mr)
{
  int64_t n       = y.size();
  auto h_col      = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto i_col      = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto D_col      = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto Dlu_col    = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto Dll_col    = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto u_col      = make_numeric_column(y.type(), n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto h_buffer   = h_col->mutable_view();
  auto i_buffer   = i_col->mutable_view();
  auto D_buffer   = D_col->mutable_view();
  auto Dlu_buffer = Dll_col->mutable_view();
  auto Dll_buffer = Dll_col->mutable_view();
  auto u_buffer   = u_col->mutable_view();

  auto zero = cudf::numeric_scalar<float>(0.0);
  auto one  = cudf::numeric_scalar<float>(1.0);
  cudf::fill_in_place(h_buffer, 0, h_col->size(), zero);
  cudf::fill_in_place(i_buffer, 0, i_col->size(), zero);
  cudf::fill_in_place(D_buffer, 0, D_col->size(), one);
  cudf::fill_in_place(Dlu_buffer, 0, Dlu_col->size(), zero);
  cudf::fill_in_place(u_buffer, 0, u_col->size(), zero);

  cudf::type_dispatcher(y.type(),
                        compute_spline_tridiagonals{},
                        t,
                        y,
                        prefixes,
                        D_buffer,
                        Dlu_buffer,
                        u_buffer,
                        h_buffer,
                        i_buffer,
                        stream);

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
  cusparseHandle_t handle;

  CUSPATIAL_CUDA_TRY(cudaMalloc(&handle, sizeof(cusparseHandle_t)));
  CUSPARSE_TRY(cusparseCreate(&handle));

  size_t pBufferSize;
  int32_t batchStride = y.size() / (prefixes.size() - 1);
  int32_t batchSize   = batchStride;

  CUSPARSE_TRY(cusparseSgtsv2StridedBatch_bufferSizeExt(handle,
                                                        batchSize,
                                                        Dll_buffer.data<float>(),
                                                        D_buffer.data<float>(),
                                                        Dlu_buffer.data<float>(),
                                                        u_buffer.data<float>(),
                                                        prefixes.size() - 1,
                                                        batchStride,
                                                        &pBufferSize));

  rmm::device_vector<float> pBuffer(pBufferSize);

  CUSPARSE_TRY(cusparseSgtsv2StridedBatch(handle,
                                          batchSize,
                                          Dll_buffer.data<float>(),
                                          D_buffer.data<float>(),
                                          Dlu_buffer.data<float>(),
                                          u_buffer.data<float>(),
                                          prefixes.size() - 1,
                                          batchStride,
                                          pBuffer.data().get()));

  CUSPARSE_TRY(cusparseDestroy(handle));

  int32_t dn = n - (prefixes.size() - 1);
  // Finally, compute coefficients via Horner's scheme
  auto d3_col = make_numeric_column(y.type(), dn, cudf::mask_state::UNALLOCATED, stream, mr);
  auto d2_col = make_numeric_column(y.type(), dn, cudf::mask_state::UNALLOCATED, stream, mr);
  auto d1_col = make_numeric_column(y.type(), dn, cudf::mask_state::UNALLOCATED, stream, mr);
  auto d0_col = make_numeric_column(y.type(), dn, cudf::mask_state::UNALLOCATED, stream, mr);
  auto d3     = d3_col->mutable_view();
  auto d2     = d2_col->mutable_view();
  auto d1     = d1_col->mutable_view();
  auto d0     = d0_col->mutable_view();

  cudf::type_dispatcher(y.type(),
                        coefficients_compute{},
                        t,
                        y,
                        prefixes,
                        h_buffer,
                        i_buffer,
                        u_buffer,
                        d3,
                        d2,
                        d1,
                        d0,
                        stream);

  // Place d3..0 into a table and return
  std::vector<std::unique_ptr<cudf::column>> table;
  table.push_back(std::move(d3_col));
  table.push_back(std::move(d2_col));
  table.push_back(std::move(d1_col));
  table.push_back(std::move(d0_col));
  std::unique_ptr<cudf::table> result = std::make_unique<cudf::table>(move(table));
  return result;
}

}  // namespace detail

// Calls the interpolate function using default memory resources.
std::unique_ptr<cudf::column> cubicspline_interpolate(cudf::column_view const& query_points,
                                                      cudf::column_view const& curve_ids,
                                                      cudf::column_view const& prefixes,
                                                      cudf::column_view const& source_points,
                                                      cudf::table_view const& coefficients,
                                                      rmm::mr::device_memory_resource* mr)
{
  return cuspatial::detail::cubicspline_interpolate(
    query_points, curve_ids, prefixes, source_points, coefficients, rmm::cuda_stream_default, mr);
}

// Calls the coeffiecients  function using default memory resources.
std::unique_ptr<cudf::table> cubicspline_coefficients(cudf::column_view const& t,
                                                      cudf::column_view const& y,
                                                      cudf::column_view const& ids,
                                                      cudf::column_view const& prefixes,
                                                      rmm::mr::device_memory_resource* mr)
{
  return cuspatial::detail::cubicspline_coefficients(
    t, y, ids, prefixes, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
