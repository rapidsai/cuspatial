#include <type_traits>

namespace cuspatial {
namespace detail {

/**
 * @internal
 * @brief returns true if all types Ts... are the same as T.
 */
template <typename T, typename... Ts>
constexpr bool is_same()
{
  return std::conjunction_v<std::is_same<T, Ts>...>;
}

/**
 * @internal
 * @brief returns true if all types Ts... are convertible to U.
 */
template <typename U, typename... Ts>
constexpr bool is_convertible_to()
{
  return std::conjunction_v<std::is_convertible<Ts, U>...>;
}

/**
 * @internal
 * @brief returns true if all types Ts... are floating point types.
 */
template <typename... Ts>
constexpr bool is_floating_point()
{
  return std::conjunction_v<std::is_floating_point<Ts>...>;
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

}  // namespace detail
}  // namespace cuspatial
