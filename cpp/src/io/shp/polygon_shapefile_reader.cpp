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
#include <utility/utility.hpp>

#include <ogrsf_frmts.h>

namespace
{
    /*
     * Read a LinearRing into x/y/size vectors
    */
    void VertexFromLinearRing(OGRLinearRing const& poRing,
                              std::vector<double>& aPointX,
                              std::vector<double>& aPointY,
                              std::vector<int>& aPartSize)
    {
        int nCount    = poRing.getNumPoints();
        int nNewCount = aPointX.size() + nCount;

        aPointX.reserve(nNewCount);
        aPointY.reserve(nNewCount);

        for (int i = nCount-1; i >= 0; i--)
        {
            aPointX.push_back(poRing.getX(i));
            aPointY.push_back(poRing.getY(i));
        }

        aPartSize.push_back(nCount);
    }

    /*
    * Read a Polygon (could be with multiple rings) into x/y/size vectors
    */
    void LinearRingFromPolygon(OGRPolygon const& poPolygon,
                               std::vector<double>& aPointX,
                               std::vector<double>& aPointY,
                               std::vector<int>& aPartSize )
    {

        VertexFromLinearRing(*(poPolygon.getExteriorRing()),
                             aPointX,
                             aPointY,
                             aPartSize);

        for (int i = 0; i < poPolygon.getNumInteriorRings(); i++)
        {
            VertexFromLinearRing(*(poPolygon.getInteriorRing(i)),
                                 aPointX,
                                 aPointY,
                                 aPartSize);
        }
    }

    /*
    * Read a Geometry (could be MultiPolygon / GeometryCollection) into x / y / size vectors
    */
    void PolygonFromGeometry(OGRGeometry const* poShape,
                             std::vector<double>& aPointX,
                             std::vector<double>& aPointY,
                             std::vector<int>& aPartSize )
    {
        OGRwkbGeometryType eFlatType = wkbFlatten(poShape->getGeometryType());

        if (eFlatType == wkbMultiPolygon || eFlatType == wkbGeometryCollection)
        {
            OGRGeometryCollection *poGC = (OGRGeometryCollection *) poShape;

            for (int i = 0; i < poGC->getNumGeometries(); i++)
            {
                PolygonFromGeometry(poGC->getGeometryRef(i),
                                    aPointX,
                                    aPointY,
                                    aPartSize);
            }
        }
        else if (eFlatType == wkbPolygon)
        {
            LinearRingFromPolygon(*((OGRPolygon *) poShape),
                                  aPointX,
                                  aPointY,
                                  aPartSize);

        }
        else
        {
            CUDF_FAIL("must be polygonal geometry.");
        }
    }

    /*
    * Read a GDALDatasetH layer (corresponding to a shapefile) into five vectors
    *
    * layer: OGRLayerH layer holding polygon data
    * g_len_v: vector of group lengths,   i.e. numbers of features/polygons (should be 1 for a single layer / g_len_v)
    * feature_lengths: vector of feature lengths, i.e. numbers of rings in features/polygons
    * ring_lengths: vector of ring lengths,    i.e. numbers of vertices in rings
    * xs:     The x component of vertices
    * ys:     The y component of vertices
    * returns number of features/polygons
    */
    int ReadLayer(const OGRLayerH layer,std::vector<int>& g_len_v,
                  std::vector<int>& feature_lengths,
                  std::vector<int>& ring_lengths,
                  std::vector<double>& xs,
                  std::vector<double>& ys)
    {
        int num_feature = 0;

        OGR_L_ResetReading(layer);

        OGRFeatureH feature;

        while ((feature = OGR_L_GetNextFeature(layer)) != NULL)
        {
            auto shape = (OGRGeometry*) OGR_F_GetGeometryRef(feature);

            CUDF_EXPECTS(shape != NULL, "Invalid Shape");

            std::vector<double> xs_temp;
            std::vector<double> ys_temp;
            std::vector<int> ring_lengths_temp;

            PolygonFromGeometry(shape, xs_temp, ys_temp, ring_lengths_temp);

            xs.insert(xs.end(), xs_temp.begin(), xs_temp.end());
            ys.insert(ys.end(), ys_temp.begin(), ys_temp.end());

            ring_lengths.insert(ring_lengths.end(),
                                ring_lengths_temp.begin(),
                                ring_lengths_temp.end());

            feature_lengths.push_back(ring_lengths_temp.size());

            OGR_F_Destroy(feature);

            num_feature++;
        }

        g_len_v.push_back(num_feature);

        return num_feature;
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

    polygons<double> polygon_from_shapefile(const char *filename)
    {
        std::vector<int> g_len_v;
        std::vector<int> feature_lengths;
        std::vector<int> ring_lengths;
        std::vector<double> xs, ys;

        GDALAllRegister();

        GDALDatasetH dataset = GDALOpenEx(filename, GDAL_OF_VECTOR, NULL, NULL, NULL);

        CUDF_EXPECTS(dataset != NULL, "Failed to open ESRI Shapefile dataset");

        OGRLayerH dataset_layer = GDALDatasetGetLayer(dataset, 0);

        CUDF_EXPECTS(dataset_layer != NULL, "Failed to open the first layer");

        int num_f = ReadLayer(dataset_layer,
                              g_len_v,
                              feature_lengths,
                              ring_lengths,
                              xs,
                              ys);

        CUDF_EXPECTS(num_f > 0, "Shapefile must have at lest one polygon");

        polygons<double> result = {
            static_cast<uint32_t>(g_len_v.size()),
            static_cast<uint32_t>(feature_lengths.size()),
            static_cast<uint32_t>(ring_lengths.size()),
            static_cast<uint32_t>(xs.size()),
            new uint32_t[g_len_v.size()],
            new uint32_t[feature_lengths.size()],
            new uint32_t[ring_lengths.size()],
            nullptr,
            nullptr,
            nullptr,
            new double[static_cast<uint32_t>(xs.size())],
            new double[static_cast<uint32_t>(ys.size())]
        };

        std::copy_n(g_len_v.begin(),         result.num_group,   result.group_length);
        std::copy_n(feature_lengths.begin(), result.num_feature, result.feature_length);
        std::copy_n(ring_lengths.begin(),    result.num_ring,    result.ring_length);
        std::copy_n(xs.begin(),              result.num_vertex,  result.x);
        std::copy_n(ys.begin(),              result.num_vertex,  result.y);

        return result;
    }
} // namespace detail
} // namespace cuspatial
