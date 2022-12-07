/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuspatial/cuda_utils.hpp>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace cuspatial {
namespace detail {

constexpr unsigned default_max_ulp = 4;

template <int size, typename = void>
struct uint_selector;

template <int size>
struct uint_selector<size, std::enable_if_t<size == 2>> {
  using type = uint16_t;
};

template <int size>
struct uint_selector<size, std::enable_if_t<size == 4>> {
  using type = uint32_t;
};

template <int size>
struct uint_selector<size, std::enable_if_t<size == 8>> {
  using type = uint64_t;
};

template <typename Bits>
Bits constexpr sign_bit_mask()
{
  return Bits{1} << 8 * sizeof(Bits) - 1;
}

template <typename T>
union FloatingPointBits {
  using Bits = typename uint_selector<sizeof(T)>::type;
  CUSPATIAL_HOST_DEVICE FloatingPointBits(T float_number) : _f(float_number) {}
  T _f;
  Bits _b;
};

/**
 * @internal
 * @brief Converts integer of sign-magnitude representation to biased representation.
 *
 * Biased representation has 1 representation of zero while sign-magnitude has 2.
 * This conversion will collapse the two representations into 1. This is in line with
 * our expectation that a positive number 1 differ from a negative number -1 by 2 hops
 * instead of 3 in biased representation.
 *
 * Example:
 * Assume `N` bits in the type `Bits`. In total 2^(N-1) representable numbers.
 * (N=4):
 *              |--------------|  |-----------------|
 * decimal    -2^3+1          -0 +0                2^3-1
 * SaM         1111          1000 0000             0111
 *
 * In SaM, 0 is represented twice. In biased representation we need to collapse
 * them to single representation, resulting in 1 more representable number in
 * biased form.
 *
 * Naturally, lowest bit should map to the smallest number representable in the range.
 * With 1 more representable number in biased form, we discard the lowest bit and start
 * at the next lowest bit.
 *              |--------------|-----------------|
 * decimal    -2^3+1           0                2^3-1
 * biased      0001           0111              1110
 *
 * The following implements the mapping independently in negative and positive range.
 *
 * Read http://en.wikipedia.org/wiki/Signed_number_representations for more
 * details on signed number representations.
 *
 * @tparam Bits Unsigned type to store the bits
 * @param sam Sign and magnitude representation
 * @return Biased representation
 */
template <typename Bits>
std::enable_if_t<std::is_unsigned_v<Bits>, Bits> CUSPATIAL_HOST_DEVICE
signmagnitude_to_biased(Bits const& sam)
{
  return sam & sign_bit_mask<Bits>() ? ~sam + 1 : sam | sign_bit_mask<Bits>();
}

/**
 * @brief Floating-point equivalence comparator based on ULP (Unit in the last place).
 *
 * @note to compare if two floating points `flhs` and `frhs` are equivalent,
 * use float_equal(flhs, frhs), instead of `float_equal(flhs-frhs, 0)`.
 * See "Infernal Zero" section of
 * https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
 *
 * @tparam T Type of floating point
 * @tparam max_ulp Maximum tolerable unit in the last place
 * @param flhs First floating point to compare
 * @param frhs Second floating point to compare
 * @return `true` if two floating points differ by less or equal to `ulp`.
 */
template <typename T, unsigned max_ulp = default_max_ulp>
bool CUSPATIAL_HOST_DEVICE float_equal(T const& flhs, T const& frhs)
{
  FloatingPointBits<T> lhs{flhs};
  FloatingPointBits<T> rhs{frhs};
  if (std::isnan(lhs._f) || std::isnan(rhs._f)) return false;
  auto lhsbiased = signmagnitude_to_biased(lhs._b);
  auto rhsbiased = signmagnitude_to_biased(rhs._b);

  return lhsbiased >= rhsbiased ? (lhsbiased - rhsbiased) <= max_ulp
                                : (rhsbiased - lhsbiased) <= max_ulp;
}

}  // namespace detail
}  // namespace cuspatial
