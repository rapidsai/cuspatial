#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cudf/utilities/error.hpp>
#include <rmm/rmm.h>
#include <cudf/types.h>
#include <cudf/column/column.hpp>
#include <cuspatial/soa_readers.hpp>
#include <utility/utility.hpp>
#include "cudf/utilities/type_dispatcher.hpp"
#include "rmm/thrust_rmm_allocator.h"

namespace cuspatial
{
    /**
	* @brief read timestamp (ts: Time type) data from file as column
	 
    * see soa_readers.hpp
    */

    // TODO: define timestamp to cuspatial timestamp kernel here
    // Reason: No more its_timestamp - its_timestamp is always converted to libcudf++
    // timestamp.

    std::unique_ptr<cudf::column> read_timestamp_soa(const char *filename)
    {
        std::vector<its_timestamp> timestamp = read_field_to_vec<its_timestamp>(filename);

        auto tid = cudf::experimental::type_to_id<int64_t>();
        auto type = cudf::data_type{ tid };
        rmm::device_buffer dbuff(timestamp.data(), timestamp.size() * sizeof(its_timestamp));
        auto ts = std::make_unique<cudf::column>(
            type, timestamp.size(), dbuff);
        return ts;
    }

}//cuspatial
