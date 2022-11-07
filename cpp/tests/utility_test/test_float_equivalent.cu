#include <cuspatial/detail/utility/floating_point.cuh>

#include <limits>
#include <rmm/device_vector.hpp>

#include <gtest/gtest.h>

using namespace cuspatial;

template <typename T>
struct ULPFloatingPointEquivalenceTest : public ::testing::Test {
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(ULPFloatingPointEquivalenceTest, TestTypes);

template <typename Float>
struct float_eq_comp {
  bool __device__ operator()(Float lhs, Float rhs) { return detail::float_equal(lhs, rhs); }
};

template <typename T>
T increment(T f, unsigned step)
{
  if (!step) return f;
  return increment(std::nextafter(f, std::numeric_limits<T>::max()), step - 1);
}

template <typename T>
T decrement(T f, unsigned step)
{
  if (!step) return f;
  return decrement(std::nextafter(f, std::numeric_limits<T>::min()), step - 1);
}

template <typename T>
void run_test(T base)
{
  T FourULPGreater = increment(base, 4);
  T FiveULPGreater = increment(base, 5);
  T FourULPLess    = decrement(base, 4);
  T FiveULPLess    = decrement(base, 5);

  std::vector<T> first{base, base, base, base};
  std::vector<T> second{FourULPGreater, FiveULPGreater, FourULPLess, FiveULPLess};

  rmm::device_vector<T> d_first(first);
  rmm::device_vector<T> d_second(second);

  std::vector<bool> expected{true, false, true, false};
  rmm::device_vector<bool> got(4);

  thrust::transform(
    d_first.begin(), d_first.end(), d_second.begin(), got.begin(), float_eq_comp<T>{});

  EXPECT_EQ(expected, got);
}

TYPED_TEST(ULPFloatingPointEquivalenceTest, BiasedFromPositiveZero)
{
  using T = TypeParam;
  run_test(T{0.0});
}

TYPED_TEST(ULPFloatingPointEquivalenceTest, BiasedFromNegativeZero)
{
  using T = TypeParam;
  run_test(T{-0.0});
}

TYPED_TEST(ULPFloatingPointEquivalenceTest, TestVeryNearZeroPositive)
{
  using T                     = TypeParam;
  T very_small_positive_float = increment(T{0.0}, 1);
  run_test(very_small_positive_float);
}

TYPED_TEST(ULPFloatingPointEquivalenceTest, TestVeryNearZeroNegative)
{
  using T                     = TypeParam;
  T very_small_negative_float = decrement(T{0.0}, 1);
  run_test(very_small_negative_float);
}

TYPED_TEST(ULPFloatingPointEquivalenceTest, BiasedFromSmallPostiveFloat)
{
  using T = TypeParam;
  run_test(T{0.1});
}

TYPED_TEST(ULPFloatingPointEquivalenceTest, BiasedFromSmallNegativeFloat)
{
  using T = TypeParam;
  run_test(T{-0.1});
}

TYPED_TEST(ULPFloatingPointEquivalenceTest, BiasedFromPostiveFloat)
{
  using T = TypeParam;
  run_test(T{1234.0});
}

TYPED_TEST(ULPFloatingPointEquivalenceTest, BiasedFromNegativeFloat)
{
  using T = TypeParam;
  run_test(T{-5678.0});
}

TYPED_TEST(ULPFloatingPointEquivalenceTest, BiasedFromVeryLargePositiveFloat)
{
  using T                     = TypeParam;
  T very_large_positive_float = decrement(std::numeric_limits<T>::max(), 10);
  run_test(very_large_positive_float);
}

TYPED_TEST(ULPFloatingPointEquivalenceTest, BiasedFromVeryLargeNegativeFloat)
{
  using T                     = TypeParam;
  T very_large_negative_float = increment(std::numeric_limits<T>::min(), 10);
  run_test(very_large_negative_float);
}
