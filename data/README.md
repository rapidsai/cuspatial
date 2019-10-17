# Data pre-processing for C++/Python test code
## Data Sources

The schema data derived from a traffic surveillance camera dataset named
schema_HWY_20_AND_LOCUST-filtered.json can be
[downloaded here](https://drive.google.com/file/d/1GKTB5SV2RK7lEOIWz8tWab5MGWDtMWWW/view?usp=sharing).

Regions of Interest (ROIs) covered by cameras in ESRI shapefile format (named
its_4326_roi.*) can be
[downloaded here](https://nvidia-my.sharepoint.com/:u:/p/jiantingz/ESvNHXtWgSxDtf2xXTcVN1IByp5HKoUWLhuPTr_bS2ecSw?e=gf4VUu).

The camera parameter file (for 27 ROIs) can be
[downloaded here](https://nvidia-my.sharepoint.com/:x:/p/jiantingz/EZPkLpJPrUtOmwmBPSlNNxwBgeh8UAYlEyrRuT5QLkvj7Q?e=thLUQS)  

For application background [see here](https://www.nvidia.com/en-us/deep-learning-ai/industries/ai-cities/)

## Instructions
Download these three data files to {cudf_home}/data and compile/run two data
preprocessing C++ programs in the folder to prepare the data files for the
C++/Python test code. In addition to its_4326_roi.* and its_camera_2.csv,
four derived SoA data files are needed for the tests: vehicle identification
(`.objectid`), timestamp (`.time`), lon/lat location
(`.location`) and polygon (`.ply`). The instructions to compile and run
`json2soa.cpp` and `poly2soa.cpp` are provided at the beginning of the two
programs.

### json2soa
To compile, download cJSON.c and cJSON.h from the
[cJson website](https://github.com/DaveGamble/cJSON) and put them in the
current directory.

```
g++ json2soa.cpp cJSON.c -o json2soa -O3
```

To run:

```
./json2soa schema_HWY_20_AND_LOCUST-filtered.json locust -1
```

The three parameters for the program are: input json file name
(schema_HWY_20_AND_LOCUST-filtered.json, must follow the specific schema),
the output root file name and the number of records to be processed. A total of
five files with `.time`, `.objectid`, `.bbox`, `.location`, `.coordinate`
extensions will be generated and three will be used: `.time`, `.objectid` and
`.location`. The last parameter is for the desired number of locations to be
processed; -1 indicates all records but the value can be a smaller number for
easy inspection.   

### poly2soa
To compile, install a recent version of [GDAL](https://gdal.org/download.html)
under `/usr/local`.

```
g++ -I /usr/local/include -L /usr/local/lib poly2soa.cpp -lgdal -o poly2soa
```

To run:

```
 ./poly2soa its.cat itsroi.ply
```

The first parameter is the catalog file of all Shapefiles from which to extract
polygons. Currently, the provided `its.cat` has only one line which is the path
(relative or full) of the provided ROI polygon file (its_4326_roi.shp). If you
have multiple ROI shapefiles, you can list them in `its.cat` file, one `.shp`
file name per line.

## Additional Notes
The design supports multiple polygons from multiple shapefiles and polygons in
each file is considered to be a group. However, the group information is not
exposed in the current implementation but this can be changed in the future if
needed.
