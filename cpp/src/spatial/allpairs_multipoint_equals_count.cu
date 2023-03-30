/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuspatial/error.hpp>
#include <cuspatial/experimental/allpairs_multipoint_equals_count.cuh>
#include <cuspatial/experimental/geometry_collection/multipoint_ref.cuh>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/vec_2d.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <type_traits>
#include <utility>

namespace cuspatial {
namespace detail {
namespace {

struct dispatch_allpairs_multipoint_equals_count {
  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Non-floating point operation is not supported");
  }

  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    cudf::column_view const& lhs,
    cudf::column_view const& rhs,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto size = lhs.size();
    auto type = cudf::data_type(cudf::type_to_id<uint32_t>());

    auto result =
      cudf::make_fixed_width_column(type, size, cudf::mask_state::UNALLOCATED, stream, mr);

    auto lhs_iterator = make_vec_2d_iterator(lhs.begin<T>());
    auto rhs_iterator = make_vec_2d_iterator(rhs.begin<T>());
    auto lhs_ref      = multipoint_ref(lhs_iterator, lhs_iterator + lhs.size() / 2);
    auto rhs_ref      = multipoint_ref(rhs_iterator, rhs_iterator + rhs.size() / 2);

    cuspatial::allpairs_multipoint_equals_count(
      lhs_ref, rhs_ref, result->mutable_view().begin<uint32_t>(), stream);

    return result;
  }
};

}  // namespace

std::unique_ptr<cudf::column> allpairs_multipoint_equals_count(cudf::column_view const& lhs,
                                                               cudf::column_view const& rhs,
                                                               rmm::cuda_stream_view stream,
                                                               rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(lhs.type() == rhs.type(), "Column type mismatch");

  return cudf::type_dispatcher(
    lhs.type(), dispatch_allpairs_multipoint_equals_count(), lhs, rhs, stream, mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> allpairs_multipoint_equals_count(cudf::column_view const& lhs,
                                                               cudf::column_view const& rhs,
                                                               rmm::mr::device_memory_resource* mr)
{
  return detail::allpairs_multipoint_equals_count(lhs, rhs, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
