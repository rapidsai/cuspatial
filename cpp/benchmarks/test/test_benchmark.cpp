#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <memory>
#include <tests/utilities/column_wrapper.hpp>

static void BM_dummy(benchmark::State& state)
{
//   cudf::size_type size   = state.range(0);
//   cudf::size_type offset = size * (static_cast<double>(shift_factor) / 100.0);
//   auto idx_begin         = thrust::make_counting_iterator<cudf::size_type>(0);
//   auto idx_end           = thrust::make_counting_iterator<cudf::size_type>(size);

//   auto input = use_validity
//                  ? cudf::test::fixed_width_column_wrapper<int>(
//                      idx_begin,
//                      idx_end,
//                      thrust::make_transform_iterator(idx_begin, [](auto idx) { return true; }))
//                  : cudf::test::fixed_width_column_wrapper<int>(idx_begin, idx_end);

//   auto fill = use_validity ? make_scalar<int>() : make_scalar<int>(777);

//   for (auto _ : state) {
//     cuda_event_timer raii(state, true);
//     auto output = cudf::experimental::shift(input, offset, *fill);
//   }
}

class Dummy : public cudf::benchmark {
};

#define DUMMY_BM_BENCHMARK_DEFINE(name) \
  BENCHMARK_DEFINE_F(Dummy, name)(::benchmark::State & state)       \
  {                                                                 \
    BM_dummy(state);                    \
  }                                                                 \
  BENCHMARK_REGISTER_F(Dummy, name)                                 \
    ->RangeMultiplier(32)                                           \
    ->Range(1 << 10, 1 << 30)                                       \
    ->UseManualTime()                                               \
    ->Unit(benchmark::kMillisecond);

DUMMY_BM_BENCHMARK_DEFINE(dummy);
