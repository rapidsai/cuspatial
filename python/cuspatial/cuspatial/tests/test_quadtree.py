# Copyright (c) 2019, NVIDIA CORPORATION.

import cudf
from cuspatial.indexing.quadtree import quadtree_on_points


col1 = cudf.Series([0, 13, 3, 9])._column
col2 = cudf.Series([5, 3, 2, 7])._column

res= quadtree_on_points (col1,col2)
