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

#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cuspatial/polyline_bounding_box.hpp>

#include <vector>

#include "utility/bbox_thrust.cuh"

namespace {

struct bounding_box_processor {
  template <typename T, std::enable_if_t<std::is_floating_point<T>::value> * = nullptr>
  std::unique_ptr<cudf::experimental::table> operator()(const cudf::column_view &spos,
                                                        const cudf::column_view &x,
                                                        const cudf::column_view &y,
                                                        double R,
                                                        rmm::mr::device_memory_resource *mr,
                                                        cudaStream_t stream)
  {
    uint32_t num_poly   = spos.size();
    uint32_t num_vertex = x.size();

    std::cout << "bounding_box_processor: num_poly=" << num_poly << ",num_vertex=" << num_vertex
              << std::endl;

    auto exec_policy = rmm::exec_policy(stream);

    const uint32_t *d_ply_spos = spos.data<uint32_t>();
    const T *d_ply_x           = x.data<T>();
    const T *d_ply_y           = y.data<T>();

    // compute bbox

    rmm::device_buffer *db_temp_spos =
      new rmm::device_buffer(num_poly * sizeof(uint32_t), stream, mr);
    CUDF_EXPECTS(db_temp_spos != nullptr,
                 "error allocating temporal memory for first ring position array");
    uint32_t *d_temp_spos = static_cast<uint32_t *>(db_temp_spos->data());

    rmm::device_buffer *db_vertex_pid =
      new rmm::device_buffer(num_vertex * sizeof(uint32_t), stream, mr);
    CUDF_EXPECTS(db_vertex_pid != nullptr, "error allocating temporal memory for vertex id array");
    uint32_t *d_vertex_pid = static_cast<uint32_t *>(db_vertex_pid->data());

    CUDA_TRY(cudaMemset(d_vertex_pid, 0, num_vertex * sizeof(uint32_t)));

    if (0) {
      printf("segment pos prefix sum\n");
      thrust::device_ptr<const uint32_t> d_spos_ptr = thrust::device_pointer_cast(d_ply_spos);
      thrust::copy(
        d_spos_ptr, d_spos_ptr + num_poly, std::ostream_iterator<uint32_t>(std::cout, " "));
      std::cout << std::endl;
    }

    thrust::adjacent_difference(
      exec_policy->on(stream), d_ply_spos, d_ply_spos + num_poly, d_temp_spos);
    thrust::exclusive_scan(
      exec_policy->on(stream), d_temp_spos, d_temp_spos + num_poly, d_temp_spos);
    thrust::scatter(exec_policy->on(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + num_poly,
                    d_temp_spos,
                    d_vertex_pid);
    thrust::inclusive_scan(exec_policy->on(stream),
                           d_vertex_pid,
                           d_vertex_pid + num_vertex,
                           d_vertex_pid,
                           thrust::maximum<int>());
    delete db_temp_spos;
    db_temp_spos = nullptr;

    if (0) {
      printf("d_vertex_pid\n");
      thrust::device_ptr<uint32_t> d_vertex_pid_ptr = thrust::device_pointer_cast(d_vertex_pid);
      thrust::copy(d_vertex_pid_ptr,
                   d_vertex_pid_ptr + num_vertex,
                   std::ostream_iterator<uint32_t>(std::cout, " "));
      std::cout << std::endl;
    }

    rmm::device_buffer *db_bbox = new rmm::device_buffer(num_poly * sizeof(SBBox<T>), stream, mr);
    CUDF_EXPECTS(db_bbox != nullptr, "error allocating memory for bboxes");
    SBBox<T> *d_p_bbox = static_cast<SBBox<T> *>(db_bbox->data());

    auto d_vertex_iter = thrust::make_zip_iterator(thrust::make_tuple(d_ply_x, d_ply_y));

    uint32_t num_bbox = thrust::reduce_by_key(
                          exec_policy->on(stream),
                          d_vertex_pid,
                          d_vertex_pid + num_vertex,
                          thrust::make_transform_iterator(d_vertex_iter, bbox_transformation<T>()),
                          d_vertex_pid,
                          d_p_bbox,
                          thrust::equal_to<uint32_t>(),
                          bbox_reduction<T>())
                          .first -
                        d_vertex_pid;
    std::cout << "num_poly=" << num_poly << ",num_bbox=" << num_bbox << std::endl;

    CUDF_EXPECTS(num_poly == num_bbox, "#of bbox after reduction should be the same as # of polys");
    delete db_vertex_pid;
    db_vertex_pid = nullptr;

    std::unique_ptr<cudf::column> x1_col =
      cudf::make_numeric_column(cudf::data_type{cudf::experimental::type_to_id<T>()},
                                num_poly,
                                cudf::mask_state::UNALLOCATED,
                                stream,
                                mr);
    T *x1 = cudf::mutable_column_device_view::create(x1_col->mutable_view(), stream)->data<T>();
    assert(x1 != nullptr);

    std::unique_ptr<cudf::column> y1_col =
      cudf::make_numeric_column(cudf::data_type{cudf::experimental::type_to_id<T>()},
                                num_poly,
                                cudf::mask_state::UNALLOCATED,
                                stream,
                                mr);
    T *y1 = cudf::mutable_column_device_view::create(y1_col->mutable_view(), stream)->data<T>();
    assert(y1 != nullptr);

    std::unique_ptr<cudf::column> x2_col =
      cudf::make_numeric_column(cudf::data_type{cudf::experimental::type_to_id<T>()},
                                num_poly,
                                cudf::mask_state::UNALLOCATED,
                                stream,
                                mr);
    T *x2 = cudf::mutable_column_device_view::create(x2_col->mutable_view(), stream)->data<T>();
    assert(x2 != nullptr);

    std::unique_ptr<cudf::column> y2_col =
      cudf::make_numeric_column(cudf::data_type{cudf::experimental::type_to_id<T>()},
                                num_poly,
                                cudf::mask_state::UNALLOCATED,
                                stream,
                                mr);
    T *y2 = cudf::mutable_column_device_view::create(y2_col->mutable_view(), stream)->data<T>();
    assert(y2 != nullptr);

    auto out_bbox_iter = thrust::make_zip_iterator(thrust::make_tuple(x1, y1, x2, y2));
    thrust::transform(
      exec_policy->on(stream), d_p_bbox, d_p_bbox + num_bbox, out_bbox_iter, bbox2tuple<T>(R));

    delete db_bbox;
    db_bbox = nullptr;

    std::vector<std::unique_ptr<cudf::column>> bbox_cols;
    bbox_cols.push_back(std::move(x1_col));
    bbox_cols.push_back(std::move(y1_col));
    bbox_cols.push_back(std::move(x2_col));
    bbox_cols.push_back(std::move(y2_col));
    std::unique_ptr<cudf::experimental::table> destination_table =
      std::make_unique<cudf::experimental::table>(std::move(bbox_cols));

    // std::cout<<"completing bounding_box_processor.................."<<std::endl;
    return destination_table;
  }

  template <typename T, std::enable_if_t<!std::is_floating_point<T>::value> * = nullptr>
  std::unique_ptr<cudf::experimental::table> operator()(const cudf::column_view &spos,
                                                        const cudf::column_view &x,
                                                        const cudf::column_view &y,
                                                        double R,
                                                        rmm::mr::device_memory_resource *mr,
                                                        cudaStream_t stream)
  {
    CUDF_FAIL("Non-floating point operation is not supported");
  }
};

}  // end anonymous namespace

namespace cuspatial {

std::unique_ptr<cudf::experimental::table> polyline_bbox(const cudf::column_view &spos,
                                                         const cudf::column_view &x,
                                                         const cudf::column_view &y,
                                                         double R)
{
  CUDF_EXPECTS(spos.size() > 0, "number of polylines must be greater than 0");
  CUDF_EXPECTS(x.size() == y.size(),
               "numbers of vertices must be the same for both x and y columns");
  CUDF_EXPECTS(x.size() >= 2 * spos.size(), "all polylines must have at least 2 vertices");
  CUDF_EXPECTS(R >= 0, "expansion radius must be greater or equal than 0");

  cudaStream_t stream                 = 0;
  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource();

  return cudf::experimental::type_dispatcher(
    x.type(), bounding_box_processor{}, spos, x, y, R, mr, stream);
}

}  // namespace cuspatial
