#pragma once

#include <cstdint>

namespace cuspatial {

enum class geometry_type_id : uint8_t { POINT, LINESTRING, POLYGON };

enum class collection_type_id : uint8_t { SINGLE, MULTI };

}  // namespace cuspatial
