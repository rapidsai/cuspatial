/*
 *
 *  Copyright (c) 2020, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ai.rapids.cuspatial;

import ai.rapids.cudf.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

public final class CuSpatial {
  private static final Logger log = LoggerFactory.getLogger(CuSpatial.class);
  static {
    CuSpatialNativeDepsLoader.loadCuSpatialNativeDeps();
  }

  /**
   * Compute haversine distances between points in set A to the corresponding points in set B.
   *
   * https://en.wikipedia.org/wiki/Haversine_formula
   *
   * @param  aLon: longitude of points in set A
   * @param  aLat:  latitude of points in set A
   * @param  bLon: longitude of points in set B
   * @param  bLat:  latitude of points in set B
   * NOTE: As for radius of the sphere on which the points reside, default value 6371.0 is used (aprx. radius
   * of earth in km)
   * NOTE: Default `device_memory_resource` is used for allocating the output ColumnVector
   *
   * @return array of distances for all (aLon[i], aLat[i]) and (bLon[i], bLat[i]) point pairs
   */
  public static ColumnVector haversineDistance(ColumnVector aLon, ColumnVector aLat, ColumnVector bLon, ColumnVector bLat) {
    return new ColumnVector(haversineDistanceImpl(aLon.getNativeView(), aLat.getNativeView(), bLon.getNativeView(), bLat.getNativeView()));
  }

  /**
   * Find all points (x,y) that fall within a rectangular query window.
   *
   * A point (x, y) is in the window if `x > window_min_x && x < window_min_y && y > window_min_y && y
   * < window_max_y`.
   *
   * Swaps `window_min_x` and `window_max_x` if `window_min_x > window_max_x`.
   * Swaps `window_min_y` and `window_max_y` if `window_min_y > window_max_y`.
   *
   * @param windowMinX lower x-coordinate of the query window
   * @param windowMaxX upper x-coordinate of the query window
   * @param windowMinY lower y-coordinate of the query window
   * @param windowMaxY upper y-coordinate of the query window
   * @param x            x-coordinates of points to be queried
   * @param y            y-coordinates of points to be queried
   * NOTE: Default `device_memory_resource` is used for allocating the output table
   *
   * @return A table with two columns of the same type as the input columns. Columns 0, 1 are the
   * (x, y) coordinates of the points in the input that fall within the query window.
   */
  public static Table pointsInSpatialWindow(double windowMinX, double windowMaxX, double windowMinY, double windowMaxY, ColumnVector x, ColumnVector y) {
    return new Table(pointsInSpatialWindowImpl(windowMinX, windowMaxX, windowMinY, windowMaxY, x.getNativeView(), y.getNativeView()));
  }

  /**
   * Derive trajectories from object ids, points, and timestamps.
   *
   * Groups the input object ids to determine unique trajectories. Returns a
   * table with the trajectory ids, the number of objects in each trajectory,
   * and the offset position of the first object for each trajectory in the
   * input object ids column.
   *
   * @param objectId column of object (e.g., vehicle) ids
   * @param x coordinates (in kilometers)
   * @param y coordinates (in kilometers)
   * @param timestamp column of timestamps in any resolution
   * NOTE: Default `device_memory_resource` is used for allocating the output table
   *
   * @throw cuspatial::logic_error If object_id isn't cudf::type_id::INT32
   * @throw cuspatial::logic_error If x and y are different types
   * @throw cuspatial::logic_error If timestamp isn't a cudf::TIMESTAMP type
   * @throw cuspatial::logic_error If object_id, x, y, or timestamp contain nulls
   * @throw cuspatial::logic_error If object_id, x, y, and timestamp are different
   * sizes
   *
   * @return a Pair<ai.rapids.cudf.Table, ai.rapids.cudf.ColumnVector>.
   *  1. Pair.left is a table of these columns (object_id, x, y, timestamp) sorted by (object_id, timestamp)
   *  2. Pair.right is a column of type int32 for start positions for each trajectory's first object
   */
  public static Pair<Table, ColumnVector> deriveTrajectories(ColumnVector objectId, ColumnVector x, ColumnVector y, ColumnVector timestamp) {
    long[] columnAddressArray = deriveTrajectoriesImpl(objectId.getNativeView(), x.getNativeView(), y.getNativeView(), timestamp.getNativeView());
    long[] tableColumns = Arrays.copyOfRange(columnAddressArray, 0, columnAddressArray.length - 1);
    long indexColumn = columnAddressArray[columnAddressArray.length - 1];
    return new Pair<>(new Table(tableColumns), new ColumnVector(indexColumn));
  }

  public static Table trajectoryDistancesAndSpeeds(int numTrajectories, ColumnVector objectId, ColumnVector x, ColumnVector y, ColumnVector timestamp) {
    long[] columnAddressArray = trajectoryDistancesAndSpeedsImpl(numTrajectories, objectId.getNativeView(), x.getNativeView(), y.getNativeView(), timestamp.getNativeView());
    return new Table(columnAddressArray);
  }

  public static Table trajectoryBoundingBoxes(int numTrajectories, ColumnVector objectId, ColumnVector x, ColumnVector y) {
    long[] columnAddressArray = trajectoryBoundingBoxesImpl(numTrajectories, objectId.getNativeView(), x.getNativeView(), y.getNativeView());
    return new Table(columnAddressArray);
  }

  public static ColumnVector pointInPolygon(ColumnVector testPointsX, ColumnVector testPointsY,
                                            ColumnVector polyOffsets, ColumnVector polyRingOffsets,
                                            ColumnVector polyPointsX, ColumnVector polyPointsY) {
    return new ColumnVector(pointInPolygonImpl(testPointsX.getNativeView(), testPointsY.getNativeView(),
                                               polyOffsets.getNativeView(), polyRingOffsets.getNativeView(),
                                               polyPointsX.getNativeView(), polyPointsY.getNativeView()));
  }

  public static Pair<ColumnVector, ColumnVector> lonlatToCartesian(double originLon, double originLat, ColumnVector inputLon, ColumnVector inputLat) {
    long[] columnAddressArray = lonlatToCartesianImpl(originLon, originLat, inputLon.getNativeView(), inputLat.getNativeView());
    long left = columnAddressArray[0];
    long right = columnAddressArray[1];
    return new Pair<>(new ColumnVector(left), new ColumnVector(right));
  }

  /////////////////////////////////////////////////////////////////////////////
  // NATIVE APIs
  /////////////////////////////////////////////////////////////////////////////

  private static native long haversineDistanceImpl(long aLonViewHandle, long aLatViewHandle, long bLonViewHandle, long bLatViewHandle);
  private static native long[] pointsInSpatialWindowImpl(double windowMinX, double windowMaxX, double windowMinY, double windowMaxY, long xViewHandle, long yViewHandle);
  private static native long[] deriveTrajectoriesImpl(long objectIdViewHandle, long xViewHandle, long yViewHandle, long timestampViewHandle);
  private static native long[] trajectoryDistancesAndSpeedsImpl(int numTrajectories, long objectIdViewHandle, long xViewHandle, long yViewHandle, long timestampViewHandle);
  private static native long[] trajectoryBoundingBoxesImpl(int numTrajectories, long objectIdViewHandle, long xViewHandle, long yViewHandle);
  private static native long pointInPolygonImpl(long testPointsXViewHandle, long testPointsYViewHandle,
                                                long polyOffsetsViewHandle, long polyRingOffsetsViewHandle,
                                                long polyPointsXViewHandle, long polyPointsYViewHandle);
  private static native long[] lonlatToCartesianImpl(double originLon, double originLat, long inputLonViewHandle, long inputLatViewHandle);
}
