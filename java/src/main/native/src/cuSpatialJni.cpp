/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/table/table.hpp>

#include "cuspatial/coordinate_transform.hpp"
#include "cuspatial/hausdorff.hpp"
#include "cuspatial/haversine.hpp"
#include "cuspatial/point_in_polygon.hpp"
#include "cuspatial/spatial_window.hpp"
#include "cuspatial/trajectory.hpp"

#include "jni_utils.hpp"

constexpr char const *CUSPATIAL_ERROR_CLASS =
    "ai/rapids/cuspatial/CuSpatialException"; // java class package path
#define CATCH_STD_CUSPATIAL(env, ret_val) CATCH_STD_CLASS(env, CUSPATIAL_ERROR_CLASS, ret_val)

/**
 * Take a table returned by some operation and turn it into an array of column* so we can track them
 * ourselves in java instead of having their life tied to the table.
 * @param table_result the table to convert for return
 * @param extra_columns columns not in the table that will be added to the result at the end.
 */
static jlongArray
convert_table_for_return(JNIEnv *env, std::unique_ptr<cudf::table> &table_result,
                         std::vector<std::unique_ptr<cudf::column>> &extra_columns) {
  std::vector<std::unique_ptr<cudf::column>> ret = table_result->release();
  int table_cols = ret.size();
  int num_columns = table_cols + extra_columns.size();
  cudf::jni::native_jlongArray outcol_handles(env, num_columns);
  for (int i = 0; i < table_cols; i++) {
    outcol_handles[i] = reinterpret_cast<jlong>(ret[i].release());
  }
  for (int i = 0; i < extra_columns.size(); i++) {
    outcol_handles[i + table_cols] = reinterpret_cast<jlong>(extra_columns[i].release());
  }
  return outcol_handles.get_jArray();
}

namespace {

jlongArray convert_table_for_return(JNIEnv *env, std::unique_ptr<cudf::table> &table_result) {
  std::vector<std::unique_ptr<cudf::column>> extra;
  return convert_table_for_return(env, table_result, extra);
}

jlongArray convert_columns_for_return(JNIEnv *env,
                                      std::vector<std::unique_ptr<cudf::column>> &columns) {
  int num_columns = columns.size();
  cudf::jni::native_jlongArray outcol_handles(env, num_columns);
  for (int i = 0; i < num_columns; i++) {
    outcol_handles[i] = reinterpret_cast<jlong>(columns[i].release());
  }
  return outcol_handles.get_jArray();
}

} // anonymous namespace

extern "C" {

////////
// Native methods for cuspatial/haversine.hpp
////////

JNIEXPORT jlong JNICALL Java_ai_rapids_cuspatial_CuSpatial_haversineDistanceImpl(
    JNIEnv *env, jclass clazz, jlong a_lon_view_handle, jlong a_lat_view_handle,
    jlong b_lon_view_handle, jlong b_lat_view_handle) {
  JNI_NULL_CHECK(env, a_lon_view_handle, "input column_view a_lon is null", 0);
  JNI_NULL_CHECK(env, a_lat_view_handle, "input column_view a_lat is null", 0);
  JNI_NULL_CHECK(env, b_lon_view_handle, "input column_view b_lon is null", 0);
  JNI_NULL_CHECK(env, b_lat_view_handle, "input column_view b_lat is null", 0);

  using cudf::column;
  using cudf::column_view;

  try {
    column_view *a_lon_column_view = reinterpret_cast<column_view *>(a_lon_view_handle);
    column_view *a_lat_column_view = reinterpret_cast<column_view *>(a_lat_view_handle);
    column_view *b_lon_column_view = reinterpret_cast<column_view *>(b_lon_view_handle);
    column_view *b_lat_column_view = reinterpret_cast<column_view *>(b_lat_view_handle);
    std::unique_ptr<column> result = cuspatial::haversine_distance(
        *a_lon_column_view, *a_lat_column_view, *b_lon_column_view, *b_lat_column_view);

    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD_CUSPATIAL(env, 0);
}

////////
// Native methods for cuspatial/hausdorff.hpp
////////

JNIEXPORT jlong JNICALL Java_ai_rapids_cuspatial_CuSpatial_directedHausdorffDistanceImpl(
    JNIEnv *env, jclass clazz, jlong xs_view_handle, jlong ys_view_handle,
    jlong points_per_space_view_handle) {
  JNI_NULL_CHECK(env, xs_view_handle, "input column_view xs is null", 0);
  JNI_NULL_CHECK(env, ys_view_handle, "input column_view ys is null", 0);
  JNI_NULL_CHECK(env, points_per_space_view_handle, "input column_view points_per_space is null",
                 0);

  using cudf::column;
  using cudf::column_view;

  try {
    column_view *xs_column_view = reinterpret_cast<column_view *>(xs_view_handle);
    column_view *ys_column_view = reinterpret_cast<column_view *>(ys_view_handle);
    column_view *points_per_space_column_view =
        reinterpret_cast<column_view *>(points_per_space_view_handle);
    std::unique_ptr<column> result = cuspatial::directed_hausdorff_distance(
        *xs_column_view, *ys_column_view, *points_per_space_column_view);

    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD_CUSPATIAL(env, 0);
}

////////
// Native methods for cuspatial/spatial_window.hpp
////////

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cuspatial_CuSpatial_pointsInSpatialWindowImpl(
    JNIEnv *env, jclass clazz, jdouble window_min_x, jdouble window_max_x, jdouble window_min_y,
    jdouble window_max_y, jlong x_view_handle, jlong y_view_handle) {
  JNI_NULL_CHECK(env, x_view_handle, "input column_view points_x is null", 0);
  JNI_NULL_CHECK(env, y_view_handle, "input column_view points_y is null", 0);

  using cudf::column;
  using cudf::column_view;
  using cudf::table;

  try {
    column_view *x_column_view = reinterpret_cast<column_view *>(x_view_handle);
    column_view *y_column_view = reinterpret_cast<column_view *>(y_view_handle);
    std::unique_ptr<table> result = cuspatial::points_in_spatial_window(
        window_min_x, window_max_x, window_min_y, window_max_y, *x_column_view, *y_column_view);
    return convert_table_for_return(env, result);
  }
  CATCH_STD_CUSPATIAL(env, NULL);
}

////////
// Native methods for cuspatial/trajectory.hpp
////////

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cuspatial_CuSpatial_deriveTrajectoriesImpl(
    JNIEnv *env, jclass clazz, jlong object_id_view_handle, jlong x_view_handle,
    jlong y_view_handle, jlong timestamp_view_handle) {
  JNI_NULL_CHECK(env, object_id_view_handle, "input column_view object_id is null", 0);
  JNI_NULL_CHECK(env, x_view_handle, "input column_view x is null", 0);
  JNI_NULL_CHECK(env, y_view_handle, "input column_view y is null", 0);
  JNI_NULL_CHECK(env, timestamp_view_handle, "input column_view timestamp is null", 0);

  using cudf::column;
  using cudf::column_view;

  try {
    column_view *object_id_column_view = reinterpret_cast<column_view *>(object_id_view_handle);
    column_view *x_column_view = reinterpret_cast<column_view *>(x_view_handle);
    column_view *y_column_view = reinterpret_cast<column_view *>(y_view_handle);
    column_view *timestamp_column_view = reinterpret_cast<column_view *>(timestamp_view_handle);
    std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>> result =
        cuspatial::derive_trajectories(*object_id_column_view, *x_column_view, *y_column_view,
                                       *timestamp_column_view);

    std::vector<std::unique_ptr<cudf::column>> extra;
    extra.emplace_back(std::move(result.second));
    return convert_table_for_return(env, result.first, extra);
  }
  CATCH_STD_CUSPATIAL(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cuspatial_CuSpatial_trajectoryDistancesAndSpeedsImpl(
    JNIEnv *env, jclass clazz, jint num_trajectories, jlong object_id_view_handle,
    jlong x_view_handle, jlong y_view_handle, jlong timestamp_view_handle) {
  JNI_NULL_CHECK(env, object_id_view_handle, "input column_view object_id is null", 0);
  JNI_NULL_CHECK(env, x_view_handle, "input column_view x is null", 0);
  JNI_NULL_CHECK(env, y_view_handle, "input column_view y is null", 0);
  JNI_NULL_CHECK(env, timestamp_view_handle, "input column_view timestamp is null", 0);

  using cudf::column;
  using cudf::column_view;
  using cudf::table;

  try {
    cudf::size_type num_trajectories_int32 = reinterpret_cast<cudf::size_type>(num_trajectories);
    column_view *object_id_column_view = reinterpret_cast<column_view *>(object_id_view_handle);
    column_view *x_column_view = reinterpret_cast<column_view *>(x_view_handle);
    column_view *y_column_view = reinterpret_cast<column_view *>(y_view_handle);
    column_view *timestamp_column_view = reinterpret_cast<column_view *>(timestamp_view_handle);
    std::unique_ptr<table> result = cuspatial::trajectory_distances_and_speeds(
        num_trajectories_int32, *object_id_column_view, *x_column_view, *y_column_view,
        *timestamp_column_view);
    return convert_table_for_return(env, result);
  }
  CATCH_STD_CUSPATIAL(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cuspatial_CuSpatial_trajectoryBoundingBoxesImpl(
    JNIEnv *env, jclass clazz, jint num_trajectories, jlong object_id_view_handle,
    jlong x_view_handle, jlong y_view_handle) {
  JNI_NULL_CHECK(env, object_id_view_handle, "input column_view object_id is null", 0);
  JNI_NULL_CHECK(env, x_view_handle, "input column_view x is null", 0);
  JNI_NULL_CHECK(env, y_view_handle, "input column_view y is null", 0);

  using cudf::column;
  using cudf::column_view;
  using cudf::table;

  try {
    cudf::size_type num_trajectories_int32 = reinterpret_cast<cudf::size_type>(num_trajectories);
    column_view *object_id_column_view = reinterpret_cast<column_view *>(object_id_view_handle);
    column_view *x_column_view = reinterpret_cast<column_view *>(x_view_handle);
    column_view *y_column_view = reinterpret_cast<column_view *>(y_view_handle);
    std::unique_ptr<table> result = cuspatial::trajectory_bounding_boxes(
        num_trajectories_int32, *object_id_column_view, *x_column_view, *y_column_view);
    return convert_table_for_return(env, result);
  }
  CATCH_STD_CUSPATIAL(env, NULL);
}

////////
// Native methods for cuspatial/point_in_polygon.hpp
////////

JNIEXPORT jlong JNICALL Java_ai_rapids_cuspatial_CuSpatial_pointInPolygonImpl(
    JNIEnv *env, jclass clazz, jlong test_points_x_view_handle, jlong test_points_y_view_handle,
    jlong poly_offsets_view_handle, jlong poly_ring_offsets_view_handle,
    jlong poly_points_x_view_handle, jlong poly_points_y_view_handle) {
  JNI_NULL_CHECK(env, test_points_x_view_handle, "input column_view test_points_x is null", 0);
  JNI_NULL_CHECK(env, test_points_y_view_handle, "input column_view test_points_y is null", 0);
  JNI_NULL_CHECK(env, poly_offsets_view_handle, "input column_view poly_offsets is null", 0);
  JNI_NULL_CHECK(env, poly_ring_offsets_view_handle, "input column_view poly_ring_offsets is null",
                 0);
  JNI_NULL_CHECK(env, poly_points_x_view_handle, "input column_view poly_points_x is null", 0);
  JNI_NULL_CHECK(env, poly_points_y_view_handle, "input column_view poly_points_y is null", 0);

  using cudf::column;
  using cudf::column_view;

  try {
    column_view *test_points_x_column_view =
        reinterpret_cast<column_view *>(test_points_x_view_handle);
    column_view *test_points_y_column_view =
        reinterpret_cast<column_view *>(test_points_y_view_handle);
    column_view *poly_offsets_column_view =
        reinterpret_cast<column_view *>(poly_offsets_view_handle);
    column_view *poly_ring_offsets_column_view =
        reinterpret_cast<column_view *>(poly_ring_offsets_view_handle);
    column_view *poly_points_x_column_view =
        reinterpret_cast<column_view *>(poly_points_x_view_handle);
    column_view *poly_points_y_column_view =
        reinterpret_cast<column_view *>(poly_points_y_view_handle);
    std::unique_ptr<column> result = cuspatial::point_in_polygon(
        *test_points_x_column_view, *test_points_y_column_view, *poly_offsets_column_view,
        *poly_ring_offsets_column_view, *poly_points_x_column_view, *poly_points_y_column_view);

    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD_CUSPATIAL(env, 0);
}

////////
// Native methods for cuspatial/coordinate_transform.hpp
////////

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cuspatial_CuSpatial_lonlatToCartesianImpl(
    JNIEnv *env, jclass clazz, jdouble origin_lon, jdouble origin_lat, jlong input_lon_view_handle,
    jlong input_lat_view_handle) {
  JNI_NULL_CHECK(env, input_lon_view_handle, "input column_view input_lon is null", 0);
  JNI_NULL_CHECK(env, input_lat_view_handle, "input column_view input_lat is null", 0);

  using cudf::column;
  using cudf::column_view;

  try {
    column_view *input_lon_column_view = reinterpret_cast<column_view *>(input_lon_view_handle);
    column_view *input_lat_column_view = reinterpret_cast<column_view *>(input_lat_view_handle);
    std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> result =
        cuspatial::sinusoidal_projection(origin_lon, origin_lat, *input_lon_column_view,
                                         *input_lat_column_view);

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.emplace_back(std::move(result.first));
    columns.emplace_back(std::move(result.second));
    return convert_columns_for_return(env, columns);
  }
  CATCH_STD_CUSPATIAL(env, NULL);
}

} // extern "C"
