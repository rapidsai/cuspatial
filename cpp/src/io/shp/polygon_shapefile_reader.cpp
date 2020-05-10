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

#include <utility/utility.hpp>
#include <cuspatial/error.hpp>
#include <ogrsf_frmts.h>

#include <string>

namespace
{

int read_ring(OGRLinearRing const& ring,
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

int read_polygon(OGRPolygon const& polygon,
                 std::vector<int>& ring_lengths,
                 std::vector<double>& xs,
                 std::vector<double>& ys)
{
    auto num_vertices = read_ring(*(polygon.getExteriorRing()), xs, ys);
    ring_lengths.push_back(num_vertices);

    int num_interior_rings = polygon.getNumInteriorRings();

    for (int i = 0; i < num_interior_rings; i++)
    {
        auto num_vertices = read_ring(*(polygon.getInteriorRing(i)), xs, ys);
        ring_lengths.push_back(num_vertices);
    }

    return 1 + num_interior_rings;
}

int read_geometry_feature(OGRGeometry const* geometry,
                          std::vector<int>& ring_lengths,
                          std::vector<double>& xs,
                          std::vector<double>& ys)
{
    OGRwkbGeometryType geometry_type = wkbFlatten(geometry->getGeometryType());

    if (geometry_type == wkbPolygon)
    {
        return read_polygon(*((OGRPolygon *) geometry), ring_lengths, xs, ys);
    }

    if (geometry_type == wkbMultiPolygon || geometry_type == wkbGeometryCollection)
    {
        OGRGeometryCollection *geometry_collection = (OGRGeometryCollection *) geometry;

        int num_rings = 0;

        for (int i = 0; i < geometry_collection->getNumGeometries(); i++)
        {
            num_rings += read_geometry_feature(geometry_collection->getGeometryRef(i),
                                                ring_lengths,
                                                xs,
                                                ys);
        }

        return num_rings;
    }

    CUSPATIAL_FAIL("must be polygonal geometry.");
}

int read_layer(const OGRLayerH layer,
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

        auto num_rings = read_geometry_feature(geometry, ring_lengths, xs, ys);

        OGR_F_Destroy(feature);

        feature_lengths.push_back(num_rings);

        num_features++;
    }

    return num_features;
}

} // namespace

namespace cuspatial {
namespace detail {

polygon_vectors read_polygon_shapefile(std::string const& filename)
{
    GDALAllRegister();

    GDALDatasetH dataset = GDALOpenEx(filename.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);

    CUSPATIAL_EXPECTS(dataset != nullptr, "Failed to open ESRI Shapefile dataset");

    OGRLayerH dataset_layer = GDALDatasetGetLayer(dataset, 0);

    CUSPATIAL_EXPECTS(dataset_layer != nullptr, "Failed to open the first layer");

    auto poly = polygon_vectors();

    int num_features = read_layer(dataset_layer,
                                  poly.feature_lengths,
                                  poly.ring_lengths,
                                  poly.xs,
                                  poly.ys);

    poly.shrink_to_fit();

    return poly;
}

} // namespace detail
} // namespace cuspatial