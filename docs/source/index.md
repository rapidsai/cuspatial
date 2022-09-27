# Welcome to cuSpatial's documentation!

cuSpatial is a general, vector-based,
GPU accelerated GIS library that provides functionalities to spatial computation,
indexing, joins and trajectory computations.
Example functions include:
- Spatial indexing and joins supported by GPU accelerated point-in-polygon
- Trajectory identification and reconstruction
- Haversine distance and grid projection

cuSpatial integrate neatly with [GeoPandas](https://geopandas.org/en/stable/)
and [cuDF](https://docs.rapids.ai/api/cuspatial/stable/).
This enables you to accelerate performance critical sections in your `GeoPandas` workflow using and `cuSpatial` and `cuDF`.


```{toctree}
:maxdepth: 2
:caption: Contents

user_guide/index
api_docs/index
developer_guide/index
```

# Indices and tables

- {ref}`genindex`
- {ref}`search`
