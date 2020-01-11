# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
# distutils: include_dirs = cuspatial/bindings/

from cudf._lib.cudf cimport gdf_column

cdef extern from "shapefile_readers.hpp" namespace "cuspatial" nogil:
    cdef gdf_column read_polygon_shapefile(
        const char *pnt_fn,
        gdf_column *f_pos,
        gdf_column *r_pos,
        gdf_column *poly_x,
        gdf_column *poly_y
    ) except +
