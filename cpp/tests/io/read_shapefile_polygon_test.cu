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
    bool check_polygon(const cuspatial::polygons<double>&  h_polygon,const gdf_column& f_pos,
        const gdf_column& r_pos,const gdf_column& poly_x,const gdf_column& poly_y)
   {
        CUDF_EXPECTS(h_polygon.num_feature==(uint32_t)f_pos.size,"number of features/polygons mismatches expected");
        CUDF_EXPECTS(h_polygon.num_ring==(uint32_t)r_pos.size,"number of rings mismatches expected");
        CUDF_EXPECTS(h_polygon.num_vertex==(uint32_t)poly_x.size,"number of vertices mismatches expected");
        CUDF_EXPECTS(poly_x.size==poly_y.size,"numbers of vertices in x and y vectors mismatch");

        std::vector<uint32_t> h_f_pos(h_polygon.num_feature);
        EXPECT_EQ(cudaMemcpy(h_f_pos.data(),f_pos.data,
                             h_polygon.num_feature* sizeof(uint32_t),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for(size_t i = 0; i<h_polygon.num_feature;i++)
            EXPECT_EQ(h_polygon.feature_position[i],h_f_pos[i]);

        std::vector<uint32_t> h_r_pos(h_polygon.num_ring);
        EXPECT_EQ(cudaMemcpy(h_r_pos.data(),r_pos.data,
                             h_polygon.num_ring* sizeof(uint32_t),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for(size_t i = 0; i<h_polygon.num_ring;i++)
            EXPECT_EQ(h_polygon.ring_position[i],h_r_pos[i]);

        std::vector<double> h_x(h_polygon.num_vertex);
        EXPECT_EQ(cudaMemcpy(h_x.data(),poly_x.data,
                             h_polygon.num_vertex* sizeof(double),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for(size_t i = 0; i<h_polygon.num_vertex;i++)
           EXPECT_NEAR(h_polygon.x[i],h_x[i],1e-9);

        std::vector<double> h_y(h_polygon.num_vertex);
        EXPECT_EQ(cudaMemcpy(h_y.data(),poly_y.data,
                             h_polygon.num_vertex* sizeof(double),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for(size_t i = 0; i<h_polygon.num_vertex;i++)
            EXPECT_NEAR(h_polygon.y[i],h_y[i],1e-9);
        return true;
    }
};

#if  1 // disable until data files are checked in

TEST_F(ReadShapefilePolygonTest, testNonExist)
{
    const char* env_p = std::getenv("CUSPATIAL_HOME");
    CUDF_EXPECTS(env_p!=NULL,"CUSPATIAL_HOME environmental variable must be set");
    std::string shape_filename=std::string(env_p)+std::string("/test_fixtures/shapefiles/non_exist.shp"); 
    std::cout<<"Using shapefile "<<shape_filename<<std::endl;
    
    gdf_column f_pos,r_pos,poly_x,poly_y;
    EXPECT_THROW(cuspatial::read_polygon_shapefile(shape_filename.c_str(),&f_pos,&r_pos,&poly_x,&poly_y),cudf::logic_error);
}

TEST_F(ReadShapefilePolygonTest, testZero)
{

    const char* env_p = std::getenv("CUSPATIAL_HOME");
    CUDF_EXPECTS(env_p!=NULL,"CUSPATIAL_HOME environmental variable must be set");
    std::string shape_filename=std::string(env_p)+std::string("/test_fixtures/shapefiles/empty_poly.shp"); 
    std::cout<<"Using shapefile "<<shape_filename<<std::endl;

    gdf_column f_pos,r_pos,poly_x,poly_y;
    EXPECT_THROW(cuspatial::read_polygon_shapefile(shape_filename.c_str(),&f_pos,&r_pos,&poly_x,&poly_y),cudf::logic_error);
}

TEST_F(ReadShapefilePolygonTest, testOne)
{
    const char* env_p = std::getenv("CUSPATIAL_HOME");
    CUDF_EXPECTS(env_p!=NULL,"CUSPATIAL_HOME environmental variable must be set");
    std::string shape_filename=std::string(env_p)+std::string("/test_fixtures/shapefiles/one_poly.shp"); 
    std::cout<<"Using shapefile "<<shape_filename<<std::endl;
    
    cuspatial::polygons<double> h_polygon;
    h_polygon.num_group=1;
    h_polygon.num_feature=1;
    h_polygon.num_ring=1;
    h_polygon.num_vertex=5;
    h_polygon.feature_position=new uint32_t[h_polygon.num_feature]{1};
    h_polygon.ring_position=new uint32_t[h_polygon.num_ring]{5};
    h_polygon.x=new double[h_polygon.num_vertex]{-10,   5, 5, -10, -10};
    h_polygon.y=new double[h_polygon.num_vertex]{-10, -10, 5,   5, -10};

    gdf_column f_pos,r_pos,poly_x,poly_y;
    cuspatial::read_polygon_shapefile(shape_filename.c_str(),&f_pos,&r_pos,&poly_x,&poly_y);
    CUDF_EXPECTS(this->check_polygon(h_polygon,f_pos,r_pos,poly_x,poly_y),"polygon readout mismatches expected");
}

TEST_F(ReadShapefilePolygonTest, testTwo)
{
    const char* env_p = std::getenv("CUSPATIAL_HOME");
    CUDF_EXPECTS(env_p!=NULL,"CUSPATIAL_HOME environmental variable must be set");
    std::string shape_filename=std::string(env_p)+std::string("/test_fixtures/shapefiles/two_polys.shp"); 
    std::cout<<"Using shapefile "<<shape_filename<<std::endl;

    cuspatial::polygons<double> h_polygon;
    h_polygon.num_group=1;
    h_polygon.num_feature=2;
    h_polygon.num_ring=2;
    h_polygon.num_vertex=10;
    h_polygon.feature_position=new uint32_t[h_polygon.num_feature]{1,2};
    h_polygon.ring_position=new uint32_t[h_polygon.num_ring]{5,10};
    h_polygon.x=new double[h_polygon.num_vertex]{-10,   5, 5, -10, -10,0, 10, 10,  0, 0};
    h_polygon.y=new double[h_polygon.num_vertex]{-10, -10, 5,   5, -10,0,  0, 10, 10, 0};

    gdf_column f_pos,r_pos,poly_x,poly_y;
    cuspatial::read_polygon_shapefile(shape_filename.c_str(),&f_pos,&r_pos,&poly_x,&poly_y);

    CUDF_EXPECTS(this->check_polygon(h_polygon,f_pos,r_pos,poly_x,poly_y),"polygon readout mismatches expected");
}

#endif
