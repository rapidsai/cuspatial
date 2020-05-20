# cuSpatial Notebooks
## Intro
These notebooks provide examples of how to use cuSpatial.  Some of these notebooks are designed to be self-contained with the `runtime` version of the [RAPIDS Docker Container](https://hub.docker.com/r/rapidsai/rapidsai/) and [RAPIDS Nightly Docker Containers](https://hub.docker.com/r/rapidsai/rapidsai-nightly) and can run on air-gapped systems, while others require an additional download.  You can quickly get this container using the install guide from the [RAPIDS.ai Getting Started page](https://rapids.ai/start.html#get-rapids)

## Getting Started
For a good overview of how cuSpatial works, 
- Read our docs: [our precompiled docs (external link)](https://docs.rapids.ai/api/cuspatial/stable/api.html) or [build the docs them locally yourself](../docs/source/) in the
documentation tree, 
- Read [our introductory blog (external link)](https://medium.com/rapids-ai/releasing-cuspatial-to-accelerate-geospatial-and-spatiotemporal-processing-b686d8b32a9)
- Run [our python demos](../python/cuspatial/demos)


## Notebook Information
Notebook Title | Data set(s) | Notebook Description | External Download (Size)
--- | --- | --- | ---
[NYC Taxi Years Correlation](nyc_taxi_years_correlation.ipynb) | [NYC Taxi Yellow 01/2016, 01/2017, taxi zone data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | Demonstrates using Point in Polygon to correlate the NYC Taxi datasets pre-2017 `lat/lon` locations with the post-2017 `LocationID` for cross format comparisons. | Yes (~3GB)

## For more details
Many more examples can be found in the [RAPIDS Notebooks
Contrib](https://github.com/rapidsai/notebooks-contrib) repository,
which contains community-maintained notebooks.
