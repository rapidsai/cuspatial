#include <type_traits>

namespace cuspatial {
namespace detail {

/**
 * @internal
 * @brief returns true if all types are the same.
 */
template <typename T, typename... Ts>
constexpr bool is_same()
{
  return std::conjunction_v<std::is_same<T, Ts>...>;
}

/**
 * @internal
 * @brief returns true if all types are floating point types.
 */
template <typename... Ts>
constexpr bool is_floating_point()
{
  return std::conjunction_v<std::is_floating_point<Ts>...>;
}

/**
 * @internal
 * @brief returns true if all types are floating point types.
 */
template <typename... Ts>
constexpr bool is_integral()
{
  return std::conjunction_v<std::is_integral<Ts>...>;
}

/**
 * @internal
 * @brief returns true if T and all types Ts... are the same floating point type.
 */
template <typename T, typename... Ts>
constexpr bool is_same_floating_point()
{
  return std::conjunction_v<std::is_same<T, Ts>...> and
         std::conjunction_v<std::is_floating_point<Ts>...>;
}

/**
 * @internal
 * @brief Get the value type of @p Iterator type
 *
 * @tparam Iterator The iterator type to get from
 */
template <typename Iterator>
using iterator_value_type = typename std::iterator_traits<Iterator>::value_type;

/**
 * @internal
 * @brief Get the value type of the underlying vec_2d type from @p Iterator type
 *
 * @tparam Iterator The value type to get from, must point to a cuspatial::vec_2d
 */
template <typename Iterator>
using iterator_vec_base_type = typename iterator_value_type<Iterator>::value_type;

}  // namespace detail
}  // namespace cuspatial
