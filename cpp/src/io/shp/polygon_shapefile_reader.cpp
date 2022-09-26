/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/shapefile_reader.hpp>

#include <ogrsf_frmts.h>

#include <cudf/types.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <memory>
#include <string>
#include <tuple>

namespace {

// TODO: a better approach is to precompute the total size needed by xs and ys
// as a preprocess and preallocate xs, ys. `read_ring` should simply accept
// an iterator range and return the pass-the-end iterator.
cudf::size_type read_ring(OGRLinearRing const& ring,
                          std::vector<double>& xs,
                          std::vector<double>& ys,
                          const cuspatial::winding_order ring_order)
{
  auto num_vertices      = ring.getNumPoints();
  auto prev_num_vertices = xs.size();
  xs.resize(xs.size() + num_vertices);
  ys.resize(ys.size() + num_vertices);

  auto output_it = thrust::make_zip_iterator(
    thrust::make_tuple(xs.begin() + prev_num_vertices, ys.begin() + prev_num_vertices));
  if (ring_order == cuspatial::winding_order::COUNTER_CLOCKWISE) {
    auto ring_it = cuspatial::detail::make_counting_transform_iterator(
      0, [&ring](auto i) { return thrust::make_tuple(ring.getX(i), ring.getY(i)); });

    std::copy_n(ring_it, num_vertices, output_it);
  } else {
    auto ring_it =
      cuspatial::detail::make_counting_transform_iterator(0, [&ring, &num_vertices](auto i) {
        return thrust::make_tuple(ring.getX(num_vertices - i - 1), ring.getY(num_vertices - i - 1));
      });
    std::copy_n(ring_it, num_vertices, output_it);
  }

  return num_vertices;
}

cudf::size_type read_polygon(OGRPolygon const& polygon,
                             std::vector<int>& ring_lengths,
                             std::vector<double>& xs,
                             std::vector<double>& ys,
                             const cuspatial::winding_order ring_order)
{
  auto num_vertices = read_ring(*(polygon.getExteriorRing()), xs, ys, ring_order);
  ring_lengths.push_back(num_vertices);

  cudf::size_type num_interior_rings = polygon.getNumInteriorRings();

  for (cudf::size_type i = 0; i < num_interior_rings; i++) {
    auto num_vertices = read_ring(*(polygon.getInteriorRing(i)), xs, ys, ring_order);
    ring_lengths.push_back(num_vertices);
  }

  return 1 + num_interior_rings;
}

cudf::size_type read_geometry_feature(OGRGeometry const* geometry,
                                      std::vector<int>& ring_lengths,
                                      std::vector<double>& xs,
                                      std::vector<double>& ys,
                                      const cuspatial::winding_order ring_order)
{
  OGRwkbGeometryType geometry_type = wkbFlatten(geometry->getGeometryType());

  if (geometry_type == wkbPolygon) {
    auto polygon = dynamic_cast<const OGRPolygon*>(geometry);
    if (polygon == nullptr) { CUSPATIAL_FAIL("Can't cast `wkbPolygon` to `OGRPolygon&`"); }
    return read_polygon(*polygon, ring_lengths, xs, ys, ring_order);
  }

  if (geometry_type == wkbMultiPolygon || geometry_type == wkbGeometryCollection) {
    auto* geometry_collection = (OGRGeometryCollection*)geometry;

    int num_rings = 0;

    for (int i = 0; i < geometry_collection->getNumGeometries(); i++) {
      num_rings += read_geometry_feature(
        geometry_collection->getGeometryRef(i), ring_lengths, xs, ys, ring_order);
    }

    return num_rings;
  }

  CUSPATIAL_FAIL("Shapefile reader supports polygon geometry only");
}

cudf::size_type read_layer(const OGRLayerH layer,
                           std::vector<cudf::size_type>& feature_lengths,
                           std::vector<cudf::size_type>& ring_lengths,
                           std::vector<double>& xs,
                           std::vector<double>& ys,
                           const cuspatial::winding_order ring_order)
{
  cudf::size_type num_features = 0;

  OGR_L_ResetReading(layer);

  OGRFeatureH feature;

  while ((feature = OGR_L_GetNextFeature(layer)) != nullptr) {
    auto geometry = static_cast<OGRGeometry*>(OGR_F_GetGeometryRef(feature));

    CUSPATIAL_EXPECTS(geometry != nullptr, "Invalid Shape");

    auto num_rings = read_geometry_feature(geometry, ring_lengths, xs, ys, ring_order);

    feature_lengths.push_back(num_rings);

    OGR_F_Destroy(feature);

    num_features++;
  }

  return num_features;
}

}  // namespace

namespace cuspatial {
namespace detail {

std::tuple<std::vector<cudf::size_type>,
           std::vector<cudf::size_type>,
           std::vector<double>,
           std::vector<double>>
read_polygon_shapefile(std::string const& filename, cuspatial::winding_order outer_ring_winding)
{
  GDALAllRegister();

  GDALDatasetH dataset = GDALOpenEx(filename.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);

  CUSPATIAL_EXPECTS(dataset != nullptr, "ESRI Shapefile: Failed to open file");

  OGRLayerH dataset_layer = GDALDatasetGetLayer(dataset, 0);

  CUSPATIAL_EXPECTS(dataset_layer != nullptr, "ESRI Shapefile: Failed to read first layer");

  std::vector<cudf::size_type> feature_lengths;
  std::vector<cudf::size_type> ring_lengths;
  std::vector<double> xs;
  std::vector<double> ys;

  read_layer(dataset_layer, feature_lengths, ring_lengths, xs, ys, outer_ring_winding);
  feature_lengths.shrink_to_fit();
  ring_lengths.shrink_to_fit();
  xs.shrink_to_fit();
  ys.shrink_to_fit();

  return std::make_tuple(
    std::move(feature_lengths), std::move(ring_lengths), std::move(xs), std::move(ys));
}

}  // namespace detail
}  // namespace cuspatial
