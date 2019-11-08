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

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <rmm/thrust_rmm_allocator.h>
#include <cudf/types.h>
#include <cudf/legacy/column.hpp>
#include <utilities/legacy/error_utils.hpp>
#include <cuspatial/shapefile_readers.hpp>
#include <utility/utility.hpp>

#include <ogrsf_frmts.h>

namespace
{
 
    /*
     * Read a LinearRing into x/y/size vectors
    */
 
    void VertexFromLinearRing(OGRLinearRing const& poRing, std::vector<double> &aPointX, 
        std::vector<double> &aPointY,std::vector<int> &aPartSize )
    {
        int nCount = poRing.getNumPoints();
        int nNewCount = aPointX.size() + nCount;
 
        aPointX.reserve( nNewCount );
        aPointY.reserve( nNewCount );
        for (int i = nCount - 1; i >= 0; i-- )
        {
            aPointX.push_back( poRing.getX(i));
            aPointY.push_back( poRing.getY(i));
        }
        aPartSize.push_back( nCount );	
    }
 
     /*
      * Read a Polygon (could be with multiple rings) into x/y/size vectors
     */

    void LinearRingFromPolygon(OGRPolygon const & poPolygon, std::vector<double> &aPointX, 
        std::vector<double> &aPointY,std::vector<int> &aPartSize )
    {
        
        VertexFromLinearRing( *(poPolygon.getExteriorRing()),
                                        aPointX, aPointY, aPartSize );

        for(int i = 0; i < poPolygon.getNumInteriorRings(); i++ )
            VertexFromLinearRing( *(poPolygon.getInteriorRing(i)),
                                            aPointX, aPointY, aPartSize );
    }

     /*
      * Read a Geometry (could be MultiPolygon/GeometryCollection) into x/y/size vectors
     */

    void PolygonFromGeometry(OGRGeometry const *poShape, std::vector<double> &aPointX, 
        std::vector<double> &aPointY,std::vector<int> &aPartSize )
    {
        OGRwkbGeometryType eFlatType = wkbFlatten(poShape->getGeometryType());

        if (eFlatType == wkbMultiPolygon || eFlatType == wkbGeometryCollection )             
        {      
            OGRGeometryCollection *poGC = (OGRGeometryCollection *) poShape;
            for(int i = 0; i < poGC->getNumGeometries(); i++ )
            {
                OGRGeometry *poGeom=poGC->getGeometryRef(i);
                PolygonFromGeometry(poGeom,aPointX, aPointY, aPartSize );
            }
        }
        else if (eFlatType == wkbPolygon)
            LinearRingFromPolygon(*((OGRPolygon *) poShape),aPointX, aPointY, aPartSize );
        else
           CUDF_EXPECTS(0, "must be polygonal geometry." );    
    }

     /*
     * Read a GDALDataset layer (corresponding to a shapefile) into five vectors
     *
     * layer: OGRLayer layer holding polygon data
     * g_len_v: vector of group lengths, i.e., numbers of features/polygons (should be 1 for a single layer/g_len_v)
     * f_len_v: vector of feature lengths, i.e., numbers of rings in features/polygons
     * r_len_v: vector of ring lengths, i.e., numbers of vertices in rings
     * x_v: vector of x coordiantes of vertices
     * y_v: vector of y coordiantes of vertices
     * returns number of features/polygons
     
    */
    
    int ReadLayer(const OGRLayerH layer,std::vector<int>& g_len_v,std::vector<int>&f_len_v,
         std::vector<int>& r_len_v,std::vector<double>& x_v, std::vector<double>& y_v)         
    {
        int num_feature=0;
        OGR_L_ResetReading(layer );
        OGRFeatureH hFeat;
        while( (hFeat = OGR_L_GetNextFeature( layer )) != NULL )
        {
            OGRGeometry *poShape=(OGRGeometry *)OGR_F_GetGeometryRef( hFeat );
            CUDF_EXPECTS(poShape!=NULL,"Invalid Shape");
            
            std::vector<double> aPointX;
            std::vector<double> aPointY;
            std::vector<int> aPartSize;
            PolygonFromGeometry( poShape, aPointX, aPointY, aPartSize );

            x_v.insert(x_v.end(),aPointX.begin(),aPointX.end());
            y_v.insert(y_v.end(),aPointY.begin(),aPointY.end());
            r_len_v.insert(r_len_v.end(),aPartSize.begin(),aPartSize.end());
            f_len_v.push_back(aPartSize.size());
            OGR_F_Destroy( hFeat );
            num_feature++;
        }
        g_len_v.push_back(num_feature);
        return num_feature;
    }
}

namespace cuspatial
{
    /*
    * Read a polygon shapefile and fill in a polygons structure
    * ToDo: read associated relational data into a CUDF Table 
    *
    * filename: ESRI shapefile name (wtih .shp extension
    * pm: structure polygons (fixed to double type) to hold polygon data
    
    * Note: only the first layer is read - shapefiles have only one layer in GDALDataset model    
    */

    void polygon_from_shapefile(const char *filename, polygons<double>& pm)
    {
        std::vector<int> g_len_v,f_len_v,r_len_v;
        std::vector<double> x_v, y_v;
        GDALAllRegister();

        GDALDatasetH hDS = GDALOpenEx( filename, GDAL_OF_VECTOR, NULL, NULL, NULL );
        CUDF_EXPECTS(hDS!=NULL,"Failed to open ESRI Shapefile dataset");		    
        OGRLayerH hLayer = GDALDatasetGetLayer( hDS,0 );
        CUDF_EXPECTS(hLayer!=NULL,"Failed to open the first layer");
        int num_f=ReadLayer(hLayer,g_len_v,f_len_v,r_len_v,x_v,y_v);
        CUDF_EXPECTS(num_f>0,"Shapefile must have at lest one polygon");
        
        pm.num_group=g_len_v.size();
        pm.num_feature=f_len_v.size();
        pm.num_ring=r_len_v.size();
        pm.num_vertex=x_v.size();
        pm.group_length=new uint32_t[ pm.num_group];
        pm.feature_length=new uint32_t[ pm.num_feature];
        pm.ring_length=new uint32_t[ pm.num_ring];
        pm.x=new double [pm.num_vertex];
        pm.y=new double [pm.num_vertex];
        CUDF_EXPECTS(pm.group_length !=nullptr, "NULL group_length pointer");
        CUDF_EXPECTS(pm.feature_length != nullptr, "NULL feature_length pointer");
        CUDF_EXPECTS(pm.ring_length != nullptr, "NULL ring_length pointer");
        CUDF_EXPECTS(pm.x != nullptr && pm.y != nullptr, "NULL polygon x/y data pointer");

        std::copy_n(g_len_v.begin(),pm.num_group,pm.group_length);
        std::copy_n(f_len_v.begin(),pm.num_feature,pm.feature_length);
        std::copy_n(r_len_v.begin(),pm.num_ring,pm.ring_length);
        std::copy_n(x_v.begin(),pm.num_vertex,pm.x);
        std::copy_n(y_v.begin(),pm.num_vertex,pm.y);
    }

    /*
    * read polygon data from file in ESRI Shapefile format; data type of vertices is fixed to double (GDF_FLOAT64)
    * see shp_readers.hpp
    */

    void read_polygon_shapefile(const char *filename,
                      gdf_column* ply_fpos, gdf_column* ply_rpos,
                      gdf_column* ply_x, gdf_column* ply_y)
    {
        memset(ply_fpos,0,sizeof(gdf_column));
        memset(ply_rpos,0,sizeof(gdf_column));
        memset(ply_x,0,sizeof(gdf_column));
        memset(ply_y,0,sizeof(gdf_column));

        polygons<double> pm;
        memset(&pm,0,sizeof(pm));
        polygon_from_shapefile(filename,pm);
        if (pm.num_feature <=0) return;

        cudaStream_t stream{0};
        auto exec_policy = rmm::exec_policy(stream)->on(stream);

        int32_t* temp{nullptr};
        RMM_TRY( RMM_ALLOC(&temp, pm.num_feature * sizeof(int32_t), stream) );
        CUDA_TRY( cudaMemcpyAsync(temp, pm.feature_length,
                              pm.num_feature * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream) );
        //prefix-sum: len to pos
        thrust::inclusive_scan(exec_policy, temp, temp + pm.num_feature, temp);
        gdf_column_view_augmented(ply_fpos, temp, nullptr, pm.num_feature,
                              GDF_INT32, 0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE}, "f_pos");

        RMM_TRY( RMM_ALLOC(&temp, pm.num_ring * sizeof(int32_t), stream) );
        CUDA_TRY( cudaMemcpyAsync(temp, pm.ring_length,
                              pm.num_ring * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream) );
        thrust::inclusive_scan(exec_policy, temp, temp + pm.num_feature, temp);
        gdf_column_view_augmented(ply_rpos, temp, nullptr, pm.num_ring,
                              GDF_INT32, 0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE}, "r_pos");

        RMM_TRY( RMM_ALLOC(&temp, pm.num_vertex * sizeof(double), stream) );
        CUDA_TRY( cudaMemcpyAsync(temp, pm.x,
                              pm.num_vertex * sizeof(double),
                              cudaMemcpyHostToDevice, stream) );
        gdf_column_view_augmented(ply_x, temp, nullptr, pm.num_vertex,
                              GDF_FLOAT64, 0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE}, "x");

        RMM_TRY( RMM_ALLOC(&temp, pm.num_vertex * sizeof(double), stream) );
        CUDA_TRY( cudaMemcpyAsync(temp, pm.y,
                              pm.num_vertex * sizeof(double),
                              cudaMemcpyHostToDevice, stream) );
        gdf_column_view_augmented(ply_y, temp, nullptr, pm.num_vertex,
                              GDF_FLOAT64, 0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE}, "y");

        delete[] pm.feature_length;
        delete[] pm.ring_length;
        delete[] pm.x;
        delete[] pm.y;
        delete[] pm.group_length;
    }

}// namespace cuspatial
