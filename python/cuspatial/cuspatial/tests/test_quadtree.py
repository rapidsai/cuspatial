# Copyright (c) 2019, NVIDIA CORPORATION.

import cudf
import cuspatial.indexing.quadtree as qt


col1 = cudf.Series([0, 13, 3, 9])._column
col2 = cudf.Series([5, 3, 2, 7])._column

#res_tbl= qt.quadtree_on_points (col1,col2)
#print(len(res_tbl._data)) #expect 4, got 4 

res_col= qt.nested_column_test (col1,col2)
print(len(res_col.children)) #expect 4, got 0