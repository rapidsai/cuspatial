<h1>Data pre-processing for C++/Python test code <h1> 
<h2> Data Sources</h2>
The schema data derived from a traffic surveillance camera dataset named 
schema_HWY_20_AND_LOCUST-filtered.json can be downloaded from [here](https://drive.google.com/file/d/1GKTB5SV2RK7lEOIWz8tWab5MGWDtMWWW/view?usp=sharing) <br>
The Region of Interests (ROIs) covered by cameras in ESRI shapefile format (named its_4326_roi.*) can be downloaded from [here](https://nvidia-my.sharepoint.com/:u:/p/jiantingz/ESvNHXtWgSxDtf2xXTcVN1IByp5HKoUWLhuPTr_bS2ecSw?e=gf4VUu)<br>
The camera parameter file (for 27 ROIs) can be downloaded from [here](https://nvidia-my.sharepoint.com/:x:/p/jiantingz/EZPkLpJPrUtOmwmBPSlNNxwBgeh8UAYlEyrRuT5QLkvj7Q?e=thLUQS)  
For application background see [here](https://www.nvidia.com/en-us/deep-learning-ai/industries/ai-cities/)

<h2>Instructions </h2> 
Download these three data files to {cudf_home}/data and compile/run two data pre-processing C++ programs in the folder to prepare for the data files to be used in C++/Python test code.<br>  
In addition to its_4326_roi.* and its_camera_2.csv, four devirved SoA data files are needed for the tests: <br>
vehical identification (.objectid), timestamp (.time), lon/lat location (.location) and polygon (.ply) <br>
The instructions to compile and run json2soa.cpp and poly2soa.cpp are provided at the beginning of the two programs. </br>

<h3>json2soa</h3>
To compile, download cJSON.c and cJSON.h from [cJson](https://github.com/DaveGamble/cJSON) website and put them under the current directory. <br> 
g++ json2soa.cpp cJSON.c -o json2soa -O3 <br>
To run:  <br>
./json2soa schema_HWY_20_AND_LOCUST-filtered.json locust -1 <br>
The three parameters for the program are: the input json file name (schema_HWY_20_AND_LOCUST-filtered.json, must follow the specific schema, e.g., ), the output root file name and the #of records to be processed. <br>
A total of five files with ".time",".objectid",".bbox",".location",".coordinate" extensions will be generated where three will be used: ".time",".objectid" and ".location". 
The last parameter is for the desired number of locations to be processed; -1 indicates all records but the value can be a smaller number for easy inspection.   

<h3> poly2soa</h3> 
To compile, install a recent version of [GDAL](https://gdal.org/download.html) and install it under /usr/local<br>
g++ -I /usr/local/include -L /usr/local/lib poly2soa.cpp -lgdal -o poly2soa <br>
To run: <br>
 ./poly2soa its.cat itsroi.ply <br>
The first parameter is the catalog file of all Shapefiles that we want to extract polygons from.  <br>
Currently, the provided its.cat has only one line which is the path (relative or full) of the provided ROI polygon file (its_4326_roi.shp). <br>
If you have multiple ROI shapefiles, you can list them in its.cat file, one .shp file name per line. <br>

<h2> Additional Notes </h2>
The design supports mutliple polygons from multiple shapefiles and polygons in each file is considered to be a group. <br>
However, the group information is not exposed in the current implementation but this can be changed in the future, should there is a need. <br>
