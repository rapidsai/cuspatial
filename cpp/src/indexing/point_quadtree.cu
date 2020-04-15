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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cuspatial/point_quadtree.hpp>
#include <utility/helper_thrust.cuh>
#include <utility/quadtree_thrust.cuh>
#include <vector>

typedef thrust::tuple<double, double, double, double, double, uint32_t,
                      uint32_t>
    quad_point_parameters;

namespace {  // anonymous

/*
 *quadtree indexing on points using the bottom-up algorithm described at ref.
 *http://www.adms-conf.org/2019-camera-ready/zhang_adms19.pdf
 *extra care on minmizing peak device memory usage by deallocating memory as
 *early as possible
 */

template <typename T>
std::vector<std::unique_ptr<cudf::column>> dowork(
    cudf::size_type point_len, T *d_pnt_x, T *d_pnt_y, SBBox<double> bbox,
    double scale, uint32_t num_level, uint32_t min_size,
    rmm::mr::device_memory_resource *mr, cudaStream_t stream)

{
  double x1 = thrust::get<0>(bbox.first);
  double y1 = thrust::get<1>(bbox.first);
  double x2 = thrust::get<0>(bbox.second);
  double y2 = thrust::get<1>(bbox.second);

  auto exec_policy = rmm::exec_policy(stream);

  auto d_pnt_iter =
      thrust::make_zip_iterator(thrust::make_tuple(d_pnt_x, d_pnt_y));

  rmm::device_buffer *db_pnt_pntkey =
      new rmm::device_buffer(point_len * sizeof(uint32_t), stream, mr);
  CUDF_EXPECTS(db_pnt_pntkey != nullptr,
               "error allocating memory for array of Morton codes of points");
  uint32_t *d_pnt_pntkey = static_cast<uint32_t *>(db_pnt_pntkey->data());

  // computing Morton code (Z-order)
  thrust::transform(exec_policy->on(stream), d_pnt_iter, d_pnt_iter + point_len,
                    d_pnt_pntkey, xytoz<T>(bbox, num_level, scale));

  thrust::sort_by_key(exec_policy->on(stream), d_pnt_pntkey,
                      d_pnt_pntkey + point_len, d_pnt_iter);

  rmm::device_buffer *db_pnt_runkey =
      new rmm::device_buffer(point_len * sizeof(uint32_t), stream, mr);
  CUDF_EXPECTS(
      db_pnt_runkey != nullptr,
      "error allocating memory for intermediate array of quadrant keys");
  uint32_t *d_pnt_runkey = static_cast<uint32_t *>(db_pnt_runkey->data());

  rmm::device_buffer *db_pnt_runlen =
      new rmm::device_buffer(point_len * sizeof(uint32_t), stream, mr);
  CUDF_EXPECTS(db_pnt_runlen != nullptr,
               "error allocating memory for intermediate array of numbers of "
               "points in quadrants");
  uint32_t *d_pnt_runlen = static_cast<uint32_t *>(db_pnt_runlen->data());

  uint32_t num_top_quads =
      thrust::reduce_by_key(
          exec_policy->on(stream), d_pnt_pntkey, d_pnt_pntkey + point_len,
          thrust::constant_iterator<int>(1), d_pnt_runkey, d_pnt_runlen)
          .first -
      d_pnt_runkey;
  std::cout << "num_top_quads=" << num_top_quads << std::endl;

  // allocate sufficient GPU memory for "full quadrants" (Secection 4.1 of ref.)
  // assuming num_level*num_top_quads is far less than num_pnt, allocating
  // d_pnt_parentkey/d_pnt_pntlen/d_pnt_numchild is accetable as
  // d_pnt_pntkey/d_pnt_parentkey/d_pnt_runlen are freed before the allocations
  // and peak memory usage will not increase

  rmm::device_buffer *db_pnt_parentkey = new rmm::device_buffer(
      num_level * num_top_quads * sizeof(uint32_t), stream, mr);
  CUDF_EXPECTS(db_pnt_parentkey != nullptr,
               " error allocating memory for full array of quadrant keys");
  uint32_t *d_pnt_parentkey = static_cast<uint32_t *>(db_pnt_parentkey->data());
  HANDLE_CUDA_ERROR(cudaMemcpy((void *)d_pnt_parentkey, (void *)d_pnt_runkey,
                               num_top_quads * sizeof(uint32_t),
                               cudaMemcpyDeviceToDevice));

  delete db_pnt_runkey;
  db_pnt_runkey = nullptr;

  rmm::device_buffer *db_pnt_pntlen = new rmm::device_buffer(
      num_level * num_top_quads * sizeof(uint32_t), stream, mr);
  CUDF_EXPECTS(db_pnt_pntlen != nullptr,
               " error allocating memory for full array of numbers of points "
               "in quadrants");
  uint32_t *d_pnt_pntlen = static_cast<uint32_t *>(db_pnt_pntlen->data());
  HANDLE_CUDA_ERROR(cudaMemcpy((void *)d_pnt_pntlen, (void *)d_pnt_runlen,
                               num_top_quads * sizeof(uint32_t),
                               cudaMemcpyDeviceToDevice));

  delete db_pnt_runlen;
  db_pnt_runlen = nullptr;

  rmm::device_buffer *db_pnt_numchild = new rmm::device_buffer(
      num_level * num_top_quads * sizeof(uint32_t), stream, mr);
  CUDF_EXPECTS(db_pnt_numchild != nullptr,
               " error allocating memory for full array of numbers of child "
               "nodes in quadrants");
  uint32_t *d_pnt_numchild = static_cast<uint32_t *>(db_pnt_numchild->data());
  HANDLE_CUDA_ERROR(
      cudaMemset(d_pnt_numchild, 0, num_top_quads * sizeof(uint32_t)));

  // generating keys of paraent quadrants and numbers of child quadrants of
  // "full quadrants" based on the second of paragraph of Section 4.2 of ref.
  // keeping track of the number of quadrants, their begining/ending positions
  // for each level

  int lev_num[num_level], lev_bpos[num_level], lev_epos[num_level];
  lev_num[num_level - 1] = num_top_quads;
  uint32_t begin_pos = 0, end_pos = num_top_quads;
  for (int k = num_level - 1; k >= 0; k--) {
    uint32_t nk = thrust::reduce_by_key(
                      exec_policy->on(stream),
                      thrust::make_transform_iterator(
                          d_pnt_parentkey + begin_pos, get_parent(2)),
                      thrust::make_transform_iterator(d_pnt_parentkey + end_pos,
                                                      get_parent(2)),
                      thrust::constant_iterator<int>(1),
                      d_pnt_parentkey + end_pos, d_pnt_numchild + end_pos)
                      .first -
                  (d_pnt_parentkey + end_pos);

    uint32_t nn =
        thrust::reduce_by_key(exec_policy->on(stream),
                              thrust::make_transform_iterator(
                                  d_pnt_parentkey + begin_pos, get_parent(2)),
                              thrust::make_transform_iterator(
                                  d_pnt_parentkey + end_pos, get_parent(2)),
                              d_pnt_pntlen + begin_pos,
                              d_pnt_parentkey + end_pos, d_pnt_pntlen + end_pos)
            .first -
        (d_pnt_parentkey + end_pos);

    lev_num[k] = nk;
    lev_bpos[k] = begin_pos;
    lev_epos[k] = end_pos;

    std::cout << "lev=" << k << " begin_pos=" << begin_pos
              << " end_pos=" << end_pos << " nk=" << nk << " nn=" << nn
              << std::endl;
    begin_pos = end_pos;
    end_pos += nk;
  }

  /*
   * allocate three temporal arrays for parent key,number of children,
   * and the number of points in each quadrant, respectively
   * d_pnt_fullkey will be copied to the data array of the key column after
   * revmoing invlaid quadtree ndoes d_pnt_qtclen and d_pnt_qtnlen will be
   * combined to generate the final length array see fig.1 of ref.
   */
  rmm::device_buffer *db_pnt_fullkey =
      new rmm::device_buffer(end_pos * sizeof(uint32_t), stream, mr);
  CUDF_EXPECTS(db_pnt_fullkey != nullptr,
               " error allocating memory for compacted array of quadrant keys");
  uint32_t *d_pnt_fullkey = static_cast<uint32_t *>(db_pnt_fullkey->data());

  rmm::device_buffer *db_pnt_qtclen =
      new rmm::device_buffer(end_pos * sizeof(uint32_t), stream, mr);
  CUDF_EXPECTS(db_pnt_qtclen != nullptr,
               " error allocating memory for compacted array of numbers of "
               "quadrant child nodes");
  uint32_t *d_pnt_qtclen = static_cast<uint32_t *>(db_pnt_qtclen->data());

  rmm::device_buffer *db_pnt_qtnlen =
      new rmm::device_buffer(end_pos * sizeof(uint32_t), stream, mr);
  CUDF_EXPECTS(db_pnt_qtnlen != nullptr,
               " error allocating memory for compacted array of numbers of "
               "points in quadrants");
  uint32_t *d_pnt_qtnlen = static_cast<uint32_t *>(db_pnt_qtnlen->data());

  rmm::device_buffer *db_pnt_fulllev =
      new rmm::device_buffer(end_pos * sizeof(uint8_t), stream, mr);
  CUDF_EXPECTS(
      db_pnt_fulllev != nullptr,
      " error allocating memory for compacted array of levels of quadrants");
  uint8_t *d_pnt_fulllev = static_cast<uint8_t *>(db_pnt_fulllev->data());

  // reverse the order of quadtree nodes for easier manipulation; skip the root
  // node
  uint32_t num_count_nodes = 0;
  for (uint32_t k = 0; k < num_level; k++) {
    thrust::fill(thrust::device, d_pnt_fulllev + num_count_nodes,
                 d_pnt_fulllev + num_count_nodes + (lev_epos[k] - lev_bpos[k]),
                 k);

    uint32_t num_lev_nodes =
        thrust::copy(exec_policy->on(stream), d_pnt_parentkey + lev_bpos[k],
                     d_pnt_parentkey + lev_epos[k],
                     d_pnt_fullkey + num_count_nodes) -
        (d_pnt_fullkey + num_count_nodes);

    thrust::copy(exec_policy->on(stream), d_pnt_numchild + lev_bpos[k],
                 d_pnt_numchild + lev_epos[k], d_pnt_qtclen + num_count_nodes);

    thrust::copy(exec_policy->on(stream), d_pnt_pntlen + lev_bpos[k],
                 d_pnt_pntlen + lev_epos[k], d_pnt_qtnlen + num_count_nodes);

    thrust::reduce(exec_policy->on(stream), d_pnt_pntlen + lev_bpos[k],
                   d_pnt_pntlen + lev_epos[k]);

    num_count_nodes += num_lev_nodes;
  }
  // Note: root node (size is 1) not counted in num_count_nodes
  CUDF_EXPECTS(num_count_nodes == begin_pos,
               "number of quadtree nodes veryifcation failed");
  std::cout << "num_count_nodes=" << num_count_nodes << std::endl;

  /*
   *delete oversized nodes for memroy efficiency
   *num_count_nodes should be typically much smaller than
   *num_level*num_top_quads
   */
  delete db_pnt_parentkey;
  db_pnt_parentkey = nullptr;
  delete db_pnt_numchild;
  db_pnt_numchild = nullptr;
  delete db_pnt_pntlen;
  db_pnt_pntlen = nullptr;

  int num_parent_nodes = 0;
  for (uint32_t k = 1; k < num_level; k++) num_parent_nodes += lev_num[k];
  std::cout << "num_parent_nodes=" << num_parent_nodes << std::endl;

  // five columns in the quadtree structure
  std::unique_ptr<cudf::column> key_col, lev_col, sign_col, length_col,
      fpos_col;

  // if the top level nodes are already all leaf nodes, special care is needed
  if (num_parent_nodes > 0) {
    // temporal device memory for vector expansion
    rmm::device_buffer *db_pnt_tmppos =
        new rmm::device_buffer(num_parent_nodes * sizeof(uint32_t), stream, mr);
    CUDF_EXPECTS(
        db_pnt_tmppos != nullptr,
        " error allocating memory for temporal array for vector expansion");
    uint32_t *d_pnt_tmppos = static_cast<uint32_t *>(db_pnt_tmppos->data());

    // line 1 of algorithm in Fig. 5 in ref.
    thrust::exclusive_scan(exec_policy->on(stream), d_pnt_qtclen,
                           d_pnt_qtclen + num_parent_nodes, d_pnt_tmppos);
    size_t num_child_nodes = thrust::reduce(
        exec_policy->on(stream), d_pnt_qtclen, d_pnt_qtclen + num_parent_nodes);
    std::cout << "num_child_nodes=" << num_child_nodes << std::endl;

    rmm::device_buffer *db_pnt_parentpos =
        new rmm::device_buffer(num_child_nodes * sizeof(uint32_t), stream, mr);
    CUDF_EXPECTS(db_pnt_parentpos != nullptr,
                 " error allocating memory for array of parent node positions");
    uint32_t *d_pnt_parentpos =
        static_cast<uint32_t *>(db_pnt_parentpos->data());
    HANDLE_CUDA_ERROR(
        cudaMemset(d_pnt_parentpos, 0, num_child_nodes * sizeof(uint32_t)));

    // line 2 of algorithm in Fig. 5 in ref.
    thrust::scatter(exec_policy->on(stream), thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + num_parent_nodes,
                    d_pnt_tmppos, d_pnt_parentpos);

    delete db_pnt_tmppos;
    db_pnt_tmppos = nullptr;

    // line 3 of algorithm in Fig. 5 in ref.
    thrust::inclusive_scan(exec_policy->on(stream), d_pnt_parentpos,
                           d_pnt_parentpos + num_child_nodes, d_pnt_parentpos,
                           thrust::maximum<int>());

    /*
     *counting the number of nodes whose children have numbers of points no less
     *than min_size; note that we start at level 2 as level nodes (whose parents
     *are the root node -level 0) need to be kept
     */
    auto iter_in = thrust::make_zip_iterator(thrust::make_tuple(
        d_pnt_fullkey + lev_num[1], d_pnt_fulllev + lev_num[1],
        d_pnt_qtclen + lev_num[1], d_pnt_qtnlen + lev_num[1], d_pnt_parentpos));

    int num_invalid_parent_nodes =
        thrust::count_if(exec_policy->on(stream), iter_in,
                         iter_in + (num_parent_nodes - lev_num[1]),
                         remove_discard(d_pnt_qtnlen, min_size));

    CUDF_EXPECTS(num_invalid_parent_nodes <= num_parent_nodes,
                 "check on number of invalid parent nodes no more than number "
                 "of parent nodes failed");

    num_parent_nodes -= num_invalid_parent_nodes;

    std::cout << "after:num_parent_nodes=" << num_parent_nodes << std::endl;

    // line 4 of algorithm in Fig. 5 in ref.
    rmm::device_buffer *db_pnt_templen =
        new rmm::device_buffer(end_pos * sizeof(uint32_t), stream, mr);
    CUDF_EXPECTS(db_pnt_templen != nullptr,
                 " error allocating memory for the copy array of numbers of "
                 "points in quadrants");
    uint32_t *d_pnt_templen = static_cast<uint32_t *>(db_pnt_templen->data());
    HANDLE_CUDA_ERROR(cudaMemcpy((void *)d_pnt_templen, (void *)d_pnt_qtnlen,
                                 end_pos * sizeof(uint32_t),
                                 cudaMemcpyDeviceToDevice));

    // line 5 of algorithm in Fig. 5 in ref.
    int num_valid_nodes =
        thrust::remove_if(exec_policy->on(stream), iter_in,
                          iter_in + num_child_nodes,
                          remove_discard(d_pnt_templen, min_size)) -
        iter_in;
    std::cout << "num_valid_nodes=" << num_valid_nodes << std::endl;

    delete db_pnt_templen;
    db_pnt_templen = nullptr;
    delete db_pnt_parentpos;
    db_pnt_parentpos = nullptr;

    // add back level 1 nodes
    num_valid_nodes += lev_num[1];
    std::cout << "num_invalid_parent_nodes=" << num_invalid_parent_nodes
              << std::endl;
    std::cout << "num_valid_nodes=" << num_valid_nodes << std::endl;

    /*
     *preparing the key column for output
     *Note: only the first num_valid_nodes elements should in the output array
     */
    key_col = cudf::make_numeric_column(
        cudf::data_type(cudf::type_id::INT32), num_valid_nodes,
        cudf::mask_state::UNALLOCATED, stream, mr);
    uint32_t *d_pnt_qtkey = cudf::mutable_column_device_view::create(
                                key_col->mutable_view(), stream)
                                ->data<uint32_t>();
    CUDF_EXPECTS(d_pnt_qtkey != nullptr,
                 " error allocating storage memory for key column");

    thrust::copy(exec_policy->on(stream), d_pnt_fullkey,
                 d_pnt_fullkey + num_valid_nodes, d_pnt_qtkey);

    delete db_pnt_fullkey;
    db_pnt_fullkey = nullptr;

    lev_col = cudf::make_numeric_column(
        cudf::data_type(cudf::type_id::INT8), num_valid_nodes,
        cudf::mask_state::UNALLOCATED, stream, mr);
    uint8_t *d_pnt_qtlev = cudf::mutable_column_device_view::create(
                               lev_col->mutable_view(), stream)
                               ->data<uint8_t>();
    CUDF_EXPECTS(d_pnt_qtlev != nullptr,
                 " error allocating storage memory for level column");

    thrust::copy(exec_policy->on(stream), d_pnt_fulllev,
                 d_pnt_fulllev + num_valid_nodes, d_pnt_qtlev);

    delete db_pnt_fulllev;
    db_pnt_fulllev = nullptr;

    // preparing the sign/indicator array for output
    sign_col = cudf::make_numeric_column(
        cudf::data_type(cudf::type_id::BOOL8), num_valid_nodes,
        cudf::mask_state::UNALLOCATED, stream, mr);

    bool *d_pnt_qtsign = cudf::mutable_column_device_view::create(
                             sign_col->mutable_view(), stream)
                             ->data<bool>();
    CUDF_EXPECTS(d_pnt_qtsign != nullptr,
                 " error allocating storage memory for sign column");
    HANDLE_CUDA_ERROR(
        cudaMemset(d_pnt_qtsign, 0, num_valid_nodes * sizeof(bool)));

    // line 6 of algorithm in Fig. 5 in ref.
    thrust::transform(exec_policy->on(stream), d_pnt_qtnlen,
                      d_pnt_qtnlen + num_parent_nodes, d_pnt_qtsign,
                      thrust::placeholders::_1 > min_size);

    // line 7 of algorithm in Fig. 5 in ref.
    thrust::replace_if(exec_policy->on(stream), d_pnt_qtnlen,
                       d_pnt_qtnlen + num_parent_nodes, d_pnt_qtsign,
                       thrust::placeholders::_1, 0);

    // allocating two temporal array:the first child position array and first
    // point position array,respectively later they will be used to generate the
    // final position array

    rmm::device_buffer *db_pnt_qtnpos =
        new rmm::device_buffer(num_valid_nodes * sizeof(uint32_t), stream, mr);
    CUDF_EXPECTS(
        db_pnt_qtnpos != nullptr,
        "  error allocating memory for array of first point positions");
    uint32_t *d_pnt_qtnpos = static_cast<uint32_t *>(db_pnt_qtnpos->data());

    rmm::device_buffer *db_pnt_qtcpos =
        new rmm::device_buffer(num_valid_nodes * sizeof(uint32_t), stream, mr);
    CUDF_EXPECTS(
        db_pnt_qtcpos != nullptr,
        "  error allocating memory for array of first quadtree node positions");
    uint32_t *d_pnt_qtcpos = static_cast<uint32_t *>(db_pnt_qtcpos->data());

    /*
     *revision to line 8 of algorithm in Fig. 5 in ref.
     *ajust nlen and npos based on last-level z-order code
     */

    rmm::device_buffer *db_pnt_tmp_key =
        new rmm::device_buffer(num_valid_nodes * sizeof(uint32_t), stream, mr);
    CUDF_EXPECTS(db_pnt_tmp_key != nullptr,
                 "  error allocating memory for array of temporal keys in "
                 "reordering first point positions");
    uint32_t *d_pnt_tmp_key = static_cast<uint32_t *>(db_pnt_tmp_key->data());
    HANDLE_CUDA_ERROR(cudaMemcpy((void *)d_pnt_tmp_key, (void *)d_pnt_qtkey,
                                 num_valid_nodes * sizeof(uint32_t),
                                 cudaMemcpyDeviceToDevice));

    rmm::device_buffer *db_pnt_tmp_lpos =
        new rmm::device_buffer(num_valid_nodes * sizeof(uint32_t), stream, mr);
    CUDF_EXPECTS(db_pnt_tmp_lpos != nullptr,
                 "  error allocating memory for array of temporal leaf node "
                 "positions in reordering first point positions");
    uint32_t *d_pnt_tmp_lpos = static_cast<uint32_t *>(db_pnt_tmp_lpos->data());

    auto key_lev_iter = thrust::make_zip_iterator(
        thrust::make_tuple(d_pnt_qtkey, d_pnt_qtlev, d_pnt_qtsign));
    thrust::transform(exec_policy->on(stream), key_lev_iter,
                      key_lev_iter + num_valid_nodes, d_pnt_tmp_key,
                      flatten_z_code(num_level));

    uint32_t num_leaf_nodes =
        thrust::copy_if(
            exec_policy->on(stream), thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(0) + num_valid_nodes, d_pnt_qtsign,
            d_pnt_tmp_lpos, !thrust::placeholders::_1) -
        d_pnt_tmp_lpos;

    std::cout << "num_leaf_nodes=" << num_leaf_nodes << std::endl;

    rmm::device_buffer *db_pnt_tmp_seq =
        new rmm::device_buffer(num_valid_nodes * sizeof(uint32_t), stream, mr);
    CUDF_EXPECTS(db_pnt_tmp_seq != nullptr,
                 "  error allocating memory for array of sequential numbers in "
                 "reordering first point positions");
    uint32_t *d_pnt_tmp_seq = static_cast<uint32_t *>(db_pnt_tmp_seq->data());

    rmm::device_buffer *db_pnt_tmp_nlen =
        new rmm::device_buffer(num_valid_nodes * sizeof(uint32_t), stream, mr);
    CUDF_EXPECTS(db_pnt_tmp_nlen != nullptr,
                 "  error allocating memory for array of sequential numbers in "
                 "reordering first point positions");
    uint32_t *d_pnt_tmp_nlen = static_cast<uint32_t *>(db_pnt_tmp_nlen->data());

    rmm::device_buffer *db_pnt_tmp_npos =
        new rmm::device_buffer(num_valid_nodes * sizeof(uint32_t), stream, mr);
    CUDF_EXPECTS(db_pnt_tmp_npos != nullptr,
                 "  error allocating memory for array of sequential numbers in "
                 "reordering first point positions");
    uint32_t *d_pnt_tmp_npos = static_cast<uint32_t *>(db_pnt_tmp_npos->data());

    thrust::sequence(exec_policy->on(stream), d_pnt_tmp_seq,
                     d_pnt_tmp_seq + num_valid_nodes);

    thrust::copy(exec_policy->on(stream), d_pnt_qtnlen,
                 d_pnt_qtnlen + num_valid_nodes, d_pnt_tmp_nlen);

    auto seq_len_pos = thrust::make_zip_iterator(
        thrust::make_tuple(d_pnt_tmp_seq, d_pnt_tmp_nlen));

    thrust::stable_sort_by_key(exec_policy->on(stream), d_pnt_tmp_key,
                               d_pnt_tmp_key + num_valid_nodes, seq_len_pos);

    thrust::remove_if(exec_policy->on(stream), d_pnt_tmp_nlen,
                      d_pnt_tmp_nlen + num_valid_nodes, d_pnt_tmp_nlen,
                      thrust::placeholders::_1 == 0);

    // only the first num_leaf_nodes are needed after the above removal (copy_if
    // and remove_if should return the same numbers
    thrust::exclusive_scan(exec_policy->on(stream), d_pnt_tmp_nlen,
                           d_pnt_tmp_nlen + num_leaf_nodes, d_pnt_tmp_npos);

    auto len_pos_iter = thrust::make_zip_iterator(
        thrust::make_tuple(d_pnt_tmp_nlen, d_pnt_tmp_npos));

    thrust::stable_sort_by_key(thrust::device, d_pnt_tmp_seq,
                               d_pnt_tmp_seq + num_leaf_nodes, len_pos_iter);

    delete db_pnt_tmp_seq;
    db_pnt_tmp_seq = nullptr;

    HANDLE_CUDA_ERROR(
        cudaMemset(d_pnt_qtnlen, 0, num_valid_nodes * sizeof(uint32_t)));
    HANDLE_CUDA_ERROR(
        cudaMemset(d_pnt_qtnpos, 0, num_valid_nodes * sizeof(uint32_t)));

    auto in_len_pos_iter = thrust::make_zip_iterator(
        thrust::make_tuple(d_pnt_tmp_nlen, d_pnt_tmp_npos));
    auto out_len_pos_iter = thrust::make_zip_iterator(
        thrust::make_tuple(d_pnt_qtnlen, d_pnt_qtnpos));
    thrust::scatter(thrust::device, in_len_pos_iter,
                    in_len_pos_iter + num_leaf_nodes, d_pnt_tmp_lpos,
                    out_len_pos_iter);

    delete db_pnt_tmp_lpos;
    db_pnt_tmp_lpos = nullptr;
    delete db_pnt_tmp_nlen;
    db_pnt_tmp_nlen = nullptr;
    delete db_pnt_tmp_npos;
    db_pnt_tmp_npos = nullptr;

    // line 9 of algorithm in Fig. 5 in ref.
    thrust::replace_if(exec_policy->on(stream), d_pnt_qtclen,
                       d_pnt_qtclen + num_valid_nodes, d_pnt_qtsign,
                       !thrust::placeholders::_1, 0);

    // line 10 of algorithm in Fig. 5 in ref.
    thrust::exclusive_scan(exec_policy->on(stream), d_pnt_qtclen,
                           d_pnt_qtclen + num_valid_nodes, d_pnt_qtcpos,
                           lev_num[1]);

    // preparing the length and fpos array for output

    length_col = cudf::make_numeric_column(
        cudf::data_type(cudf::type_id::INT32), num_valid_nodes,
        cudf::mask_state::UNALLOCATED, stream, mr);
    uint32_t *d_pnt_qtlength = cudf::mutable_column_device_view::create(
                                   length_col->mutable_view(), stream)
                                   ->data<uint32_t>();
    CUDF_EXPECTS(d_pnt_qtlength != nullptr,
                 " error allocating memory storage for length column");

    fpos_col = cudf::make_numeric_column(
        cudf::data_type(cudf::type_id::INT32), num_valid_nodes,
        cudf::mask_state::UNALLOCATED, stream, mr);
    uint32_t *d_pnt_qtfpos = cudf::mutable_column_device_view::create(
                                 fpos_col->mutable_view(), stream)
                                 ->data<uint32_t>();
    CUDF_EXPECTS(d_pnt_qtfpos != nullptr,
                 " error allocating memory storage for fpos column");

    // line 11 of algorithm in Fig. 5 in ref.
    auto iter_len_in = thrust::make_zip_iterator(
        thrust::make_tuple(d_pnt_qtclen, d_pnt_qtnlen, d_pnt_qtsign));
    auto iter_pos_in = thrust::make_zip_iterator(
        thrust::make_tuple(d_pnt_qtcpos, d_pnt_qtnpos, d_pnt_qtsign));
    thrust::transform(exec_policy->on(stream), iter_len_in,
                      iter_len_in + num_valid_nodes, d_pnt_qtlength,
                      what2output());
    thrust::transform(exec_policy->on(stream), iter_pos_in,
                      iter_pos_in + num_valid_nodes, d_pnt_qtfpos,
                      what2output());

    delete db_pnt_qtnpos;
    db_pnt_qtnpos = nullptr;
    delete db_pnt_qtcpos;
    db_pnt_qtcpos = nullptr;
    delete db_pnt_qtnlen;
    db_pnt_qtnlen = nullptr;
    delete db_pnt_qtclen;
    db_pnt_qtclen = nullptr;
  } else {
    uint32_t num_valid_nodes = num_top_quads;
    std::cout << "quadtree:num_valid_nodes=" << num_valid_nodes << std::endl;
    key_col = cudf::make_numeric_column(
        cudf::data_type(cudf::type_id::INT32), num_valid_nodes,
        cudf::mask_state::UNALLOCATED, stream, mr);
    uint32_t *d_pnt_qtkey = cudf::mutable_column_device_view::create(
                                key_col->mutable_view(), stream)
                                ->data<uint32_t>();
    CUDF_EXPECTS(d_pnt_qtkey != nullptr,
                 " error allocating storage memory for key column");
    thrust::copy(exec_policy->on(stream), d_pnt_fullkey,
                 d_pnt_fullkey + num_valid_nodes, d_pnt_qtkey);
    delete db_pnt_fullkey;
    db_pnt_fullkey = nullptr;

    lev_col = cudf::make_numeric_column(
        cudf::data_type(cudf::type_id::INT8), num_valid_nodes,
        cudf::mask_state::UNALLOCATED, stream, mr);
    uint8_t *d_pnt_qtlev = cudf::mutable_column_device_view::create(
                               lev_col->mutable_view(), stream)
                               ->data<uint8_t>();
    CUDF_EXPECTS(d_pnt_qtlev != nullptr,
                 " error allocating storage memory for level column");
    thrust::copy(exec_policy->on(stream), d_pnt_fulllev,
                 d_pnt_fulllev + num_valid_nodes, d_pnt_qtlev);

    if (0) {
      rmm::device_vector<uint8_t> d_temp_lev(num_valid_nodes);
      thrust::fill(d_temp_lev.begin(), d_temp_lev.end(), 0);
      bool lev_res =
          thrust::equal(exec_policy->on(stream), d_pnt_qtlev,
                        d_pnt_qtlev + num_valid_nodes, d_temp_lev.begin());
      CUDF_EXPECTS(lev_res, " top level quadrants must have lev=0");
    }
    delete db_pnt_fulllev;
    db_pnt_fulllev = nullptr;

    sign_col = cudf::make_numeric_column(
        cudf::data_type(cudf::type_id::BOOL8), num_valid_nodes,
        cudf::mask_state::UNALLOCATED, stream, mr);
    bool *d_pnt_qtsign = cudf::mutable_column_device_view::create(
                             sign_col->mutable_view(), stream)
                             ->data<bool>();
    rmm::device_vector<bool> d_temp_sign(num_valid_nodes);
    thrust::fill(d_temp_sign.begin(), d_temp_sign.end(), 0);
    thrust::copy(exec_policy->on(stream), d_temp_sign.begin(),
                 d_temp_sign.end(), d_pnt_qtsign);

    length_col = cudf::make_numeric_column(
        cudf::data_type(cudf::type_id::INT32), num_valid_nodes,
        cudf::mask_state::UNALLOCATED, stream, mr);
    uint32_t *d_pnt_qtlength = cudf::mutable_column_device_view::create(
                                   length_col->mutable_view(), stream)
                                   ->data<uint32_t>();
    CUDF_EXPECTS(d_pnt_qtlength != nullptr,
                 " error allocating memory storage for length column");
    thrust::copy(exec_policy->on(stream), d_pnt_qtnlen,
                 d_pnt_qtnlen + num_valid_nodes, d_pnt_qtlength);
    delete db_pnt_qtnlen;
    db_pnt_qtnlen = nullptr;

    fpos_col = cudf::make_numeric_column(
        cudf::data_type(cudf::type_id::INT32), num_valid_nodes,
        cudf::mask_state::UNALLOCATED, stream, mr);
    uint32_t *d_pnt_qtfpos = cudf::mutable_column_device_view::create(
                                 fpos_col->mutable_view(), stream)
                                 ->data<uint32_t>();
    CUDF_EXPECTS(d_pnt_qtfpos != nullptr,
                 " error allocating memory storage for fpos column");
    thrust::exclusive_scan(exec_policy->on(stream), d_pnt_qtlength,
                           d_pnt_qtlength + num_valid_nodes, d_pnt_qtfpos);

    delete db_pnt_qtclen;
    db_pnt_qtclen = nullptr;
  }

  std::vector<std::unique_ptr<cudf::column>> quad_cols;
  quad_cols.push_back(std::move(key_col));
  quad_cols.push_back(std::move(lev_col));
  quad_cols.push_back(std::move(sign_col));
  quad_cols.push_back(std::move(length_col));
  quad_cols.push_back(std::move(fpos_col));
  return quad_cols;
}

struct quadtree_point_processor {
  template <typename T,
            std::enable_if_t<std::is_floating_point<T>::value> * = nullptr>
  std::unique_ptr<cudf::experimental::table> operator()(
      cudf::mutable_column_view &x, cudf::mutable_column_view &y,
      quad_point_parameters qpi, rmm::mr::device_memory_resource *mr,
      cudaStream_t stream) {
    T *d_pnt_x = cudf::mutable_column_device_view::create(x, stream)->data<T>();
    T *d_pnt_y = cudf::mutable_column_device_view::create(y, stream)->data<T>();
    double x1 = thrust::get<0>(qpi);
    double y1 = thrust::get<1>(qpi);
    double x2 = thrust::get<2>(qpi);
    double y2 = thrust::get<3>(qpi);
    SBBox<double> bbox(thrust::make_tuple(x1, y1), thrust::make_tuple(x2, y2));
    double scale = thrust::get<4>(qpi);
    uint32_t num_level = thrust::get<5>(qpi);
    uint32_t min_size = thrust::get<6>(qpi);

    std::vector<std::unique_ptr<cudf::column>> quad_cols =
        dowork<T>(x.size(), d_pnt_x, d_pnt_y, bbox, scale, num_level, min_size,
                  mr, stream);

    std::unique_ptr<cudf::experimental::table> destination_table =
        std::make_unique<cudf::experimental::table>(std::move(quad_cols));

    return destination_table;
  }

  template <typename T,
            std::enable_if_t<!std::is_floating_point<T>::value> * = nullptr>
  std::unique_ptr<cudf::experimental::table> operator()(
      cudf::mutable_column_view &x, cudf::mutable_column_view &y,
      quad_point_parameters qpi, rmm::mr::device_memory_resource *mr,
      cudaStream_t stream) {
    CUDF_FAIL("Non-floating point operation is not supported");
  }
};

}  // end anonymous namespace

namespace cuspatial {

std::unique_ptr<cudf::experimental::table> quadtree_on_points(
    cudf::mutable_column_view &x, cudf::mutable_column_view &y, double x1,
    double y1, double x2, double y2, double scale, int num_level,
    int min_size) {
  CUDF_EXPECTS(x.size() == y.size(),
               "x and y columns might have the same lenght");
  CUDF_EXPECTS(x.size() > 0, "point dataset can not be empty");
  CUDF_EXPECTS(x1 < x2 && y1 < y2, "invalid bounding box (x1,y1,x2,y2)");
  CUDF_EXPECTS(scale > 0, "scale must be positive");
  CUDF_EXPECTS(num_level >= 0 && num_level < 16,
               "maximum of levels might be in [0,16)");
  CUDF_EXPECTS(
      min_size > 0,
      "minimum number of points for a non-leaf node must be larger than zero");

  cudaStream_t stream = 0;
  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource();

  quad_point_parameters qpi =
      thrust::make_tuple(x1, y1, x2, y2, scale, num_level, min_size);
  return cudf::experimental::type_dispatcher(
      x.type(), quadtree_point_processor{}, x, y, qpi, mr, stream);
}

}  // namespace cuspatial
