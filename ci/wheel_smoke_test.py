# Copyright (c) 2023, NVIDIA CORPORATION.

import numpy as np
import cudf
import cuspatial
import pyarrow as pa
from shapely.geometry import Point

if __name__ == '__main__':
    order, quadtree = cuspatial.quadtree_on_points(
        cuspatial.GeoSeries([Point(0.5, 0.5), Point(1.5, 1.5)]),
        *(0, 2, 0, 2),  # bbox
        1,  # scale
        1,  # max_depth
        1,  # min_size
    )
    cudf.testing.assert_frame_equal(
        quadtree,
        cudf.DataFrame(
            {
                "key": cudf.Series(pa.array([0, 3], type=pa.uint32())),
                "level": cudf.Series(pa.array([0, 0], type=pa.uint8())),
                "is_internal_node": cudf.Series(pa.array([False, False], type=pa.bool_())),
                "length": cudf.Series(pa.array([1, 1], type=pa.uint32())),
                "offset": cudf.Series(pa.array([0, 1], type=pa.uint32())),
            }
        ),
    )
