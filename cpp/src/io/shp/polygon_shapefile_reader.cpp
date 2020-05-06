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
#include <algorithm>
#include <cuspatial/error.hpp>
#include <cudf/types.hpp>
#include <utility/utility.hpp>


#include <ogrsf_frmts.h>

namespace
{
    /*
     * Read a LinearRing into x/y/size vectors
    */
    int ReadRing(OGRLinearRing const& ring,
                 std::vector<double>& xs,
                 std::vector<double>& ys)
    {
        int num_vertices = ring.getNumPoints();

        // append points in reverse order
        for (int i = num_vertices - 1; i >= 0; i--) 
        {
            xs.push_back(ring.getX(i));
            ys.push_back(ring.getY(i));
        }

        return num_vertices;
    }

    /*
    * Read a Polygon (could be with multiple rings) into x/y/size vectors
    */
    int ReadPolygon(OGRPolygon const& polygon,
                    std::vector<int>& ring_lengths,
                    std::vector<double>& xs,
                    std::vector<double>& ys)
    {
        auto num_vertices = ReadRing(*(polygon.getExteriorRing()), xs, ys);
        ring_lengths.push_back(num_vertices);

        int num_interior_rings = polygon.getNumInteriorRings();

        for (int i = 0; i < num_interior_rings; i++)
        {
            auto num_vertices = ReadRing(*(polygon.getInteriorRing(i)), xs, ys);
            ring_lengths.push_back(num_vertices);
        }

        return 1 + num_interior_rings;
    }

    /*
    * Read a Geometry (could be MultiPolygon / GeometryCollection) into x / y / size vectors
    */
    int ReadGeometryFeature(OGRGeometry const* geometry,
                            std::vector<int>& ring_lengths,
                            std::vector<double>& xs,
                            std::vector<double>& ys)
    {
        OGRwkbGeometryType geometry_type = wkbFlatten(geometry->getGeometryType());

        if (geometry_type == wkbPolygon)
        {
            return ReadPolygon(*((OGRPolygon *) geometry), ring_lengths, xs, ys);
        }

        if (geometry_type == wkbMultiPolygon || geometry_type == wkbGeometryCollection)
        {
            OGRGeometryCollection *geometry_collection = (OGRGeometryCollection *) geometry;

            int num_rings = 0;

            for (int i = 0; i < geometry_collection->getNumGeometries(); i++)
            {
                num_rings += ReadGeometryFeature(geometry_collection->getGeometryRef(i),
                                                 ring_lengths,
                                                 xs,
                                                 ys);
            }

            return num_rings;
        }
        
        CUSPATIAL_FAIL("must be polygonal geometry.");
    }

    /*
    * Read a GDALDatasetH layer (corresponding to a shapefile) into five vectors
    *
    * layer: OGRLayerH layer holding polygon data
    * group_lengths: vector of group lengths,   i.e. numbers of features/polygons (should be 1 for a single layer / group lengths)
    * feature_lengths: vector of feature lengths, i.e. numbers of rings in features/polygons
    * ring_lengths: vector of ring lengths,    i.e. numbers of vertices in rings
    * xs:     The x component of vertices
    * ys:     The y component of vertices
    * returns number of features/polygons
    */
    int ReadLayer(const OGRLayerH layer,
                  std::vector<int>& feature_lengths,
                  std::vector<int>& ring_lengths,
                  std::vector<double>& xs,
                  std::vector<double>& ys)
    {
        int num_features = 0;

        OGR_L_ResetReading(layer);

        OGRFeatureH feature;

        while ((feature = OGR_L_GetNextFeature(layer)) != nullptr)
        {
            auto geometry = (OGRGeometry*) OGR_F_GetGeometryRef(feature);

            CUSPATIAL_EXPECTS(geometry != nullptr, "Invalid Shape");

            auto num_rings = ReadGeometryFeature(geometry, ring_lengths, xs, ys);

            OGR_F_Destroy(feature);

            feature_lengths.push_back(num_rings);

            num_features++;
        }

        return num_features;
    }
}

namespace cuspatial
{
namespace detail
{
/*
* Read a polygon shapefile and fill in a polygons structure
* ToDo: read associated relational data into a CUDF Table
*
* filename: ESRI shapefile name (wtih .shp extension
* pm: structure polygons (fixed to double type) to hold polygon data

* Note: only the first layer is read - shapefiles have only one layer in GDALDatasetH model
*/

polygons read_polygon_shapefile(const char *filename)
{
    GDALAllRegister();

    GDALDatasetH dataset = GDALOpenEx(filename, GDAL_OF_VECTOR, nullptr, nullptr, nullptr);

    CUSPATIAL_EXPECTS(dataset != nullptr, "Failed to open ESRI Shapefile dataset");

    OGRLayerH dataset_layer = GDALDatasetGetLayer(dataset, 0);

    CUSPATIAL_EXPECTS(dataset_layer != nullptr, "Failed to open the first layer");

    auto poly = polygons();

    int num_features = ReadLayer(dataset_layer,
                                 poly.feature_lengths,
                                 poly.ring_lengths,
                                 poly.xs,
                                 poly.ys);

    CUSPATIAL_EXPECTS(num_features > 0, "Shapefile must have at lest one polygon");
    
    poly.group_lengths.push_back(num_features);
    poly.shrink_to_fit();

    return poly;
}

} // namespace detail

} // namespace cuspatial
