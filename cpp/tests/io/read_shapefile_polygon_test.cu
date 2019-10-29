/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <time.h>
#include <sys/time.h>
#include <string>

#include <gtest/gtest.h>
#include <utilities/legacy/error_utils.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <cuspatial/shapefile_readers.hpp>
#include <utility/utility.hpp> 

struct ReadShapefilePolygonTest : public GdfTest 
{
};   

#if  1 // disable until data files are checked in
TEST_F(ReadShapefilePolygonTest, readshapefilepolygontest)
{
    gdf_column f_pos,r_pos,poly_x,poly_y;
    std::string shape_filename=std::string("/home/jianting/cuspatial_data/its_4326_roi.shp"); 
    struct timeval t0,t1;
    gettimeofday(&t0, nullptr);
    cuspatial::read_polygon_shapefile(shape_filename.c_str(),&f_pos,&r_pos,&poly_x,&poly_y);
    std::cout<<"# of polygons= "<<f_pos.size<<std::endl;
    std::cout<<"# of rings= "<<r_pos.size<<std::endl;
    std::cout<<"# of vertices= "<<poly_x.size<<std::endl;
    
    gettimeofday(&t1, nullptr);
    float gpu_pip_time1=cuspatial::calc_time("read shapefile time......",t0,t1);
}
#endif
