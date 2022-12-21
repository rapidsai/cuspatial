#pragma once

#include <cstdint>

namespace cuspatial {

/**
 * @brief The underlying geometry type of a geometry_column_view.
 */
enum class geometry_type_id : uint8_t { POINT, LINESTRING, POLYGON };

/**
 * @brief The underlying collection type of a geometry_column_view.
 */
enum class collection_type_id : uint8_t { SINGLE, MULTI };

}  // namespace cuspatial
