# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;cuSpatial - GPU-Accelerated Vector Geospatial Data Analysis</div>

> **Note** 
> 
> cuSpatial depends on [cuDF](https://github.com/rapidsai/cudf) and [RMM](https://github.com/rapidsai/rmm) from [RAPIDS](https://rapids.ai/).

## Resources

- [cuSpatial User's Guide](https://docs.rapids.ai/api/cuspatial/stable/user_guide/cuspatial_api_examples.html): Python API reference and guides
- [cuSpatial Developer Documentation](https://docs.rapids.ai/api/cuspatial/stable/developer_guide/index.html): Understand cuSpatial's architecture
- [Getting Started](https://rapids.ai/start.html): Instructions for installing cuSpatial
- [cuSpatial Community](https://github.com/rapidsai/cuspatial/discussions): Get help, collaborate, and ask the team questions
- [cuSpatial Issues](https://github.com/rapidsai/cuspatial/issues/new/choose): Request a feature/documentation or report a bug
- [cuSpatial Roadmap](https://github.com/orgs/rapidsai/projects/41/views/5): Report issues or request features.

## Overview
cuSpatial accelerates vector geospatial operations through GPU parallelization. As part of the RAPIDS libraries, cuSpatial is inherently connected to [cuDF](https://github.com/rapidsai/cudf), [cuML](https://github.com/rapidsai/cuml), and [cuGraph](https://github.com/rapidsai/cugraph), enabling GPU acceleration across entire workflows. 

cuSpatial represents data in [GeoArrow](https://github.com/geoarrow/geoarrow) format, which enables compatibility with the [Apache Arrow](https://arrow.apache.org) ecosystem.

cuSpatial's Python API is closely matched to GeoPandas and data can seamlessly transition between the two:
```python
import geopandas
from shapely.geometry import Polygon
import cuspatial

p1 = Polygon([(0, 0), (1, 0), (1, 1)])
p2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
geoseries = geopandas.GeoSeries([p1, p2])

cuspatial_geoseries = cuspatial.from_geopandas(geoseries)
print(cuspatial_geoseries)
```
Output:
```
0    POLYGON ((0 0, 1 0, 1 1, 0 0))
1    POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))
```

For additional examples, browse the complete [API documentation](https://docs.rapids.ai/api/cuspatial/stable/), or check out more detailed [notebooks](https://github.com/rapidsai/notebooks-contrib). the [NYC Taxi][1] and [Weather][2] notebooks make use of cuSpatial.

## Supported Geospatial Operations

cuSpatial is constantly working on new features! Check out the [epics](https://github.com/orgs/rapidsai/projects/41/views/4) for a high-level view of our development, or the [roadmap](https://github.com/orgs/rapidsai/projects/41/views/5) for the details!

### Core Spatial Functions
- Spatial relationship queries (DE-9IM) [Follow Development!](https://github.com/rapidsai/cuspatial/milestone/5)
  - [Contains Properly](https://docs.rapids.ai/api/cuspatial/stable/api_docs/geopandas_compatibility.html#cuspatial.GeoSeries.contains_properly)
  - [Linestring-Linestring Intersections](https://docs.rapids.ai/api/cuspatial/stable/user_guide/cuspatial_api_examples.html#Linestring-Intersections)
- Distance computations (ST_Distance) [Follow Development!](https://github.com/rapidsai/cuspatial/issues/767)
  - [Pairwise Linestring Distance](https://docs.rapids.ai/api/cuspatial/stable/user_guide/cuspatial_api_examples.html#cuspatial.pairwise_linestring_distance)
  - [Pairwise Point-Linestring Distance](https://docs.rapids.ai/api/cuspatial/stable/api_docs/spatial.html#cuspatial.pairwise_point_linestring_distance)
- [Haversine distance](https://docs.rapids.ai/api/cuspatial/stable/user_guide/cuspatial_api_examples.html#cuspatial.haversine_distance)
- [Hausdorff distance](https://docs.rapids.ai/api/cuspatial/stable/user_guide/cuspatial_api_examples.html#cuspatial.directed_hausdorff_distance)
- [Spatial window filtering](https://docs.rapids.ai/api/cuspatial/stable/user_guide/cuspatial_api_examples.html#cuspatial.points_in_spatial_window)

### Indexing and Join Functions
- [Quadtree indexing](https://docs.rapids.ai/api/cuspatial/nightly/user_guide/cuspatial_api_examples.html#Quadtree-Indexing)
- [Spatial joins](https://docs.rapids.ai/api/cuspatial/nightly/user_guide/cuspatial_api_examples.html#Indexed-Spatial-Joins)
- [Quadtree-based point-in-polygon](https://docs.rapids.ai/api/cuspatial/stable/user_guide/cuspatial_api_examples.html#cuspatial.quadtree_point_in_polygon)
- [Quadtree-based point-to-nearest-linestring](https://docs.rapids.ai/api/cuspatial/stable/user_guide/cuspatial_api_examples.html#cuspatial.quadtree_point_to_nearest_linestring)

### Trajectory Functions
- [Deriving trajectories from point location data](https://docs.rapids.ai/api/cuspatial/stable/user_guide/cuspatial_api_examples.html#cuspatial.derive_trajectories)
- [Computing distance/speed of trajectories](https://docs.rapids.ai/api/cuspatial/stable/user_guide/cuspatial_api_examples.html#cuspatial.trajectory_distances_and_speeds)
- [Computing spatial bounding boxes of trajectories](https://docs.rapids.ai/api/cuspatial/stable/user_guide/cuspatial_api_examples.html#cuspatial.trajectory_bounding_boxes)

### What if operations I need aren't supported?
Thanks to the `from_geopandas` and `to_geopandas` functions you can accelerate what cuSpatial supports, and leave the rest of the workflow in place.

```mermaid
---
title: Integrating into Existing Workflows
---
%%{init: { 'logLevel': 'debug', 'theme': 'base', 'gitGraph': {'showBranches': false},
            'themeVariables': {'commitLabelColor': '#000000',
            'commitLabelBackground': '#ffffff',
            'commitLabelFontSize': '14px'}} }%%
gitGraph
   commit id: "Existing Workflow Start"
   commit id: "GeoPandas IO"
   commit id: "Geospatial Analytics"
   branch a
   checkout a
   commit id: "from_geopandas"
   commit id: "cuSpatial GPU Acceleration"
   branch b
   checkout b
   commit id: "cuDF"
   commit id: "cuML"
   commit id: "cuGraph"
   checkout a
   merge b
   commit id: "to_geopandas"
   checkout main
   merge a
   commit id: "Continue Work"
```


## Using cuSpatial
**CUDA/GPU requirements**
```
CUDA 11.2+
NVIDIA driver 450.80.02+
Pascal architecture or better (Compute Capability >=6.0)
```

### Quick start: Docker
Use the [RAPIDS Release Selector](https://rapids.ai/start.html#get-rapids), selecting `Docker` as the installation method. All RAPIDS Docker images contain cuSpatial.

An example command from the Release Selector:
```shell
docker pull nvcr.io/nvidia/rapidsai/rapidsai-core:23.02-cuda11.8-runtime-ubuntu22.04-py3.10
docker run --gpus all --rm -it \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8888:8888 -p 8787:8787 -p 8786:8786 \
    nvcr.io/nvidia/rapidsai/rapidsai-core:23.02-cuda11.8-runtime-ubuntu22.04-py3.10
```

### Install from Conda

To install via conda:
> **Note** cuSpatial is supported only on Linux or [through WSL](https://rapids.ai/wsl2.html), and with Python versions 3.8 and later

cuSpatial can be installed with conda (miniconda, or the full Anaconda distribution) from the rapidsai channel:

```shell
conda install -c rapidsai -c conda-forge -c nvidia \
    cuspatial=23.04 python=3.10 cudatoolkit=11.8
```
We also provide nightly Conda packages built from the HEAD of our latest development branch.

See the [RAPIDS release selector](https://rapids.ai/start.html#get-rapids) for more OS and version info.

### Install from Source

To build and install cuSpatial from source please see the [build documentation](https://docs.rapids.ai/api/cuspatial/stable/developer_guide/build.html).


[1]:https://github.com/rapidsai-community/notebooks-contrib/blob/main/community_tutorials_and_guides/taxi/NYCTaxi-E2E.ipynb
[2]:https://github.com/rapidsai-community/notebooks-contrib/blob/main/community_tutorials_and_guides/weather.ipynb
