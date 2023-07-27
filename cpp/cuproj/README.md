# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;cuProj: GPU-Accelerated Coordinate Projection</div>

## Resources
- [cuProj User's Guide](https://docs.rapids.ai/api/cuspatial/cuproj/stable/): Python API reference and guides
- [cuProj Developer Documentation](https://docs.rapids.ai/api/cuspatial/cuproj/stable/): Understand cuProj's architecture
- [Getting Started](https://docs.rapids.ai/install): Instructions for installing cuSpatial/cuProj

## Overview

cuProj is a GPU-accelerated generic coordinate transformation library that transforms geospatial coordinates from one coordinate reference system (CRS) to another. cuProj is inspired by [PROJ](https://proj.org/en/9.2/), and aims to be compatible with it.

## Supported Transformations

cuProj supports WGS84 (crs: 4326) <-> any of the 60 UTM zone transformations (crs: 32601-32660, 32701-32760).

## Installation
To install cuProj, follow the instructions in the [cuSpatial README](https://github.com/rapidsai/cuspatial/blob/main/README.md)

## Sample Usage Comparison

```python
import numpy as np
import cupy as cp

from pyproj import Transformer
from cuproj import Transformer as cuTransformer

# Define a transformer from WGS84 to UTM
transformer = Transformer.from_crs("epsg:4326", "epsg:32610")
cu_transformer = cuTransformer.from_crs("epsg:4326", "epsg:32610")

# Use a meshgrid to create a 2D array of bounded by San Francisco in WGS84
x = cp.linspace(-122.5, -121.5, 10000)
y = cp.linspace(37.5, 38.5, 10000)
xx, yy = cp.meshgrid(x, y)

cu_points = cp.stack([xx, yy], axis=-1).reshape(-1, 2)
points = cu_points.asnumpy()

# transform the points
transformed_points = transformer.transform(points[:, 0], points[:, 1])
cu_transformed_points = cu_transformer.transform(cu_points[:, 0], cu_points[:, 1])
```
