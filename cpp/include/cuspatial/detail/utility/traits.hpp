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

}  // namespace detail
}  // namespace cuspatial
