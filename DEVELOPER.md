[//]: <> (This document was written on a 94 line width terminal. Every line is spaced)
[//]: <> (precisely using the markdown "double-space" carriage return rule.)
[//]: <> (All links begin on the next line, which improves document readability in text)
[//]: <> (format and allows lining up the parenthesis on the next line for good width.)


# cuspatial python Developer's Guide

cuspatial lets developers take advantage of extremely high performance spatial algorithms  
by leveraging CUDA-enabled GPUs.

## Installing cuspatial

Read the [RAPIDS Quickstart Guide](
https://rapids.ai/start.html     ) to learn how to install all RAPIDS libraries,  
including cuspatial. It is best to install cuspatial using the [RAPIDS Release Selector](
https://rapids.ai/start.html#get-rapids).  

Install a local docker image:

```bash
docker pull rapidsai/rapidsai-core:22.06-cuda11.5-runtime-ubuntu20.04-py3.9
docker run --gpus all --rm -it \
    --shm-size=1g --ulimit memlock=-1 \
    -p 8888:8888 -p 8787:8787 -p 8786:8786 \
    rapidsai/rapidsai-core:22.06-cuda11.5-runtime-ubuntu20.04-py3.9
```

or use conda:

```bash
conda create -n rapids-22.06 -c rapidsai -c nvidia -c conda-forge  \
    cuspatial=22.06 python=3.9 cudatoolkit=11.5
```

If you wish to contribute to cuspatial, you should create a source build using the  
excellent [rapids-compose](
https://github.com/trxcllnt/rapids-compose).  


## GPU Accelerated Memory Layout

cuspatial uses `GeoArrow` buffers, a GPU-accelerated buffer type format that makes  
the performance you get with cuspatial possible. See [I/O](#io) on the fastest methods to  
get your data into cuspatial. GeoArrow uses [PyArrow](
https://arrow.apache.org/docs/python/index.html       ) bindings and types  
for storing data. GeoArrow supports [ListArrays](
https://arrow.apache.org/docs/python/data.html#arrays) for `Points`, `MultiPoints`,  
`LineStrings`, `MultiLineStrings`, `Polygons`, and `MultiPolygons`. Using an Arrow [DenseArray](
https://arrow.apache.org/docs/python/data.html#union-arrays),  
GeoArrow stores heterogeneous orderings of Features, similar to the [GeoSeries](
https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.html     ) used  
in [GeoPandas](
https://geopandas.org/en/stable/index.html).

## I/O

cuspatial supports two modes of loading Feature coordinates. [cuspatial.read_polygon_shapefile](
https://docs.rapids.ai/api/cuspatial/stable/api_docs/io.html#cuspatial.read_polygon_shapefile)  
loads a `Polygon`-only shapefile from disk. It uses GPU acceleration and can read hundreds  
of megabytes of `Polygon` information in milliseconds.  

```python
host_dataframe = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
host_dataframe["geometry"].to_file("naturalearth_lowres")
gpu_arrow = cuspatial.read_polygon_shapefile("naturalearth_lowres.shp")
print(gpu_arrow)

(0        0
1        3
2        4
3        5
4       35
      ... 
172    284
173    285
174    286
175    287
176    288
Name: f_pos, Length: 177, dtype: int32, 0          0
1          8
2         17
3         22
4         74
       ...  
284    10496
285    10544
286    10562
287    10583
288    10591
Name: r_pos, Length: 289, dtype: int32,                 x          y
0      180.000000 -16.067133
1      179.413509 -16.379054
2      179.096609 -16.433984
3      178.596839 -16.639150
4      178.725059 -17.012042
...           ...        ...
10649   28.696678   4.455077
10650   29.159078   4.389267
10651   29.715995   4.600805
10652   29.953500   4.173699
10653   30.833852   3.509172

[10654 rows x 2 columns])
```

The result of `cuspatial.read_polygon_shapefile` is a `Tuple` of GeoArrow buffers that can
be converted into a `cuspatial.GeoSeries` or used directly with other interface methods.  

```python
# TODO: The above GeoSeries constructor is in-progress in the branch  
# referenced in https://github.com/rapidsai/cuspatial/issues/612

gpu_series = cuspatial.GeoSeries(gpu_arrow)
```


If you need other geometry types, the easiest way to get data into cuspatial is via  
our GeoPandas interface:

```python
host_dataframe = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
gpu_dataframe = cuspatial.from_geopandas(host_dataframe)
print(gpu_dataframe)

         pop_est      continent                      name iso_a3  gdp_md_est  \
0       889953.0        Oceania                      Fiji    FJI        5496   
1     58005463.0         Africa                  Tanzania    TZA       63177   
2       603253.0         Africa                 W. Sahara    ESH         907   
3     37589262.0  North America                    Canada    CAN     1736425   
4    328239523.0  North America  United States of America    USA    21433226   
..           ...            ...                       ...    ...         ...   
172    6944975.0         Europe                    Serbia    SRB       51475   
173     622137.0         Europe                Montenegro    MNE        5542   
174    1794248.0         Europe                    Kosovo    KOS        7926   
175    1394973.0  North America       Trinidad and Tobago    TTO       24269   
176   11062113.0         Africa                  S. Sudan    SSD       11998   

                                              geometry  
0    MULTIPOLYGON (((180.00000 -16.06713, 180.00000...  
1    POLYGON ((33.90371 -0.95000, 34.07262 -1.05982...  
2    POLYGON ((-8.66559 27.65643, -8.66512 27.58948...  
3    MULTIPOLYGON (((-122.84000 49.00000, -122.9742...  
4    MULTIPOLYGON (((-122.84000 49.00000, -120.0000...  
..                                                 ...  
172  POLYGON ((18.82982 45.90887, 18.82984 45.90888...  
173  POLYGON ((20.07070 42.58863, 19.80161 42.50009...  
174  POLYGON ((20.59025 41.85541, 20.52295 42.21787...  
175  POLYGON ((-61.68000 10.76000, -61.10500 10.890...  
176  POLYGON ((30.83385 3.50917, 29.95350 4.17370, ...  

[177 rows x 6 columns]
(GPU)
```

While [from_geopandas](
https://docs.rapids.ai/api/cuspatial/stable/api_docs/io.html#cuspatial.read_polygon_shapefile                       ) should support any GeoPandas dataframe, it is limited because you  
must first load your data using the CPU, into host memory, before copying the data to gpu  
with `from_geopandas`.

## Geopandas and cudf integration

A cuspatial [GeoDataFrame](
https://docs.rapids.ai/api/cuspatial/stable/api_docs/geopandas_compatibility.html#cuspatial.GeoDataFrame               ) is a collection of [cudf](
https://docs.rapids.ai/api/cudf/stable/              ) [Series](
https://docs.rapids.ai/api/cudf/stable/api_docs/series.html    ) data types and  
[cuspatial.GeoSeries](
https://docs.rapids.ai/api/cuspatial/stable/api_docs/geopandas_compatibility.html#cuspatial.GeoSeries             ) `"geometry"` dtypes. Both columns are stored on the GPU, with  
`GeoSeries` being represented under the hood with `GeoArrow`. 

One of the most important features of cuspatial is that it is highly integrated with `cudf`.  
You can use any `cudf` operation on cuspatial metadata columns, and most row 1:1 operations  will work with a `geometry` column.

```python
host_dataframe = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
gpu_dataframe = cuspatial.from_geopandas(host_dataframe)
continents_dataframe = gpu_dataframe.sort_values("continent")
print(continents_dataframe)

         pop_est      continent                      name iso_a3  gdp_md_est  \
0       889953.0        Oceania                      Fiji    FJI        5496   
1     58005463.0         Africa                  Tanzania    TZA       63177   
2       603253.0         Africa                 W. Sahara    ESH         907   
3     37589262.0  North America                    Canada    CAN     1736425   
4    328239523.0  North America  United States of America    USA    21433226   
..           ...            ...                       ...    ...         ...   
172    6944975.0         Europe                    Serbia    SRB       51475   
173     622137.0         Europe                Montenegro    MNE        5542   
174    1794248.0         Europe                    Kosovo    KOS        7926   
175    1394973.0  North America       Trinidad and Tobago    TTO       24269   
176   11062113.0         Africa                  S. Sudan    SSD       11998   

                                              geometry  
0    MULTIPOLYGON (((180.00000 -16.06713, 180.00000...  
1    POLYGON ((33.90371 -0.95000, 34.07262 -1.05982...  
2    POLYGON ((-8.66559 27.65643, -8.66512 27.58948...  
3    MULTIPOLYGON (((-122.84000 49.00000, -122.9742...  
4    MULTIPOLYGON (((-122.84000 49.00000, -120.0000...  
..                                                 ...  
172  POLYGON ((18.82982 45.90887, 18.82984 45.90888...  
173  POLYGON ((20.07070 42.58863, 19.80161 42.50009...  
174  POLYGON ((20.59025 41.85541, 20.52295 42.21787...  
175  POLYGON ((-61.68000 10.76000, -61.10500 10.890...  
176  POLYGON ((30.83385 3.50917, 29.95350 4.17370, ...  

[177 rows x 6 columns]
(GPU)
```

```
# TODO: Another example
```

Though the operations will run on your CPU and lose GPU efficiency, you can also convert  
between GPU-backed `cuspatial.GeoDataFrame` and host-backed `geopandas.GeoDataFrame` with  
`from_geopandas` and `to_geopandas`, enabling you to take advantage of any native GeoPandas  
operation. The following example displays the `Polygon` associated with the first item in  
the dataframe sorted alphabetically by name.

```python
host_dataframe = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
gpu_dataframe = cuspatial.from_geopandas(host_dataframe)
gpu_dataframe.sort_value("name")
sorted_dataframe = gpu_dataframe.to_geopandas()
geopandas[0]
```
![afghanistan.png](img/afghanistan/png)

## Spatial joins

cuspatial supports high-performance spatial joins. The API surface for spatial joins does  
not yet map to GeoPandas, but with knowledge of our underlying data formats you can call  
`cuspatial.point_in_polygon` for large numbers of points on 32 polygons or less, or call  
`cuspatial.point_in_polygon_quadtree` for large numbers of points and polygons.

```python
host_dataframe = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres))
gpu_dataframe = cuspatial.from_geopandas(host_dataframe)
```


## Trajectory fits

## Spatial utilities

## GeoArrow data format
