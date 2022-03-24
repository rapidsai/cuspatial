#include "distances_utilities.cuh"

#include <cuspatial/error.hpp>
#include <cuspatial/polygon_distance.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <limits>
#include <memory>
#include <type_traits>

namespace {

struct polygon_distance_functor {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    cudf::column_view const& xs,
    cudf::column_view const& ys,
    cudf::device_span<cudf::size_type> const& space_offsets,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto const num_points = static_cast<uint32_t>(xs.size());
    auto const num_spaces = static_cast<uint32_t>(space_offsets.size());

    CUSPATIAL_EXPECTS(num_spaces < (1 << 15), "Total number of spaces must be less than 2^16");

    auto const num_results = num_spaces * num_spaces;

    auto result =
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id{cudf::type_to_id<T>()}},
                                    num_results,
                                    cudf::mask_state::UNALLOCATED,
                                    stream,
                                    mr);

    auto result_mview = result->mutable_view();
    thrust::fill(rmm::exec_policy(stream),
                 result_mview.begin<T>(),
                 result_mview.end<T>(),
                 std::numeric_limits<T>::max());

    auto const threads_per_block = 64;
    auto const num_blocks        = (num_points + threads_per_block - 1) / threads_per_block;

    auto kernel = cuspatial::detail::distances_kernel<T, cuspatial::DISTANCE_KIND::SHORTEST>;
    kernel<<<num_blocks, threads_per_block>>>(num_points,
                                              xs.begin<T>(),
                                              ys.begin<T>(),
                                              space_offsets.size(),
                                              space_offsets.begin(),
                                              result_mview.begin<T>());

    return result;
  }

  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Non-floating point operation is not supported");
  }
};

}  // namespace

namespace cuspatial {

std::unique_ptr<cudf::column> polygon_distance(
  cudf::column_view const& xs,
  cudf::column_view const& ys,
  cudf::device_span<cudf::size_type> const& space_offsets,
  rmm::mr::device_memory_resource* mr)
{
  using device_span_size_type = cudf::device_span<cudf::size_type>::size_type;

  CUSPATIAL_EXPECTS(xs.type() == ys.type(), "Inputs `xs` and `ys` must have same type.");
  CUSPATIAL_EXPECTS(xs.size() == ys.size(), "Inputs `xs` and `ys` must have same length.");

  CUSPATIAL_EXPECTS(not xs.has_nulls() and not ys.has_nulls(), "Inputs must not have nulls.");

  CUSPATIAL_EXPECTS(static_cast<device_span_size_type>(xs.size()) >= space_offsets.size(),
                    "At least one point is required for each space");

  return cudf::type_dispatcher(
    xs.type(), polygon_distance_functor{}, xs, ys, space_offsets, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial