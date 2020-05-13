#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cudf/utilities/error.hpp>
#include <rmm/rmm.h>
#include <cudf/types.h>
#include <cudf/column/column.hpp>
#include <cuspatial/soa_readers.hpp>
#include "cudf/utilities/type_dispatcher.hpp"
#include "rmm/mr/device/device_memory_resource.hpp"
#include "rmm/thrust_rmm_allocator.h"
#include <utility/legacy/utility.hpp>

namespace cuspatial {
namespace experimental {
    /**
	* @brief read timestamp (ts: Time type) data from file as column
	 
    * see soa_readers.hpp
    */

    // TODO: define timestamp to cuspatial timestamp kernel here
    // Reason: No more its_timestamp - its_timestamp is always converted to libcudf++
    // timestamp.

    std::unique_ptr<cudf::column> read_timestamp_soa(std::string const& filename, rmm::mr::device_memory_resource *mr)
    {
        std::vector<its_timestamp> timestamp = cuspatial::detail::read_field_to_vec<its_timestamp>(filename.c_str());

        auto tid = cudf::experimental::type_to_id<int64_t>();
        auto type = cudf::data_type{ tid };
        rmm::device_buffer dbuff(timestamp.data(), timestamp.size() * sizeof(its_timestamp));
        auto ts = std::make_unique<cudf::column>(
            type, timestamp.size(), dbuff);
        return ts;
    }

}//experimental
}//cuspatial
