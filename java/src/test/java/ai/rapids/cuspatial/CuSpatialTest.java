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

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import static ai.rapids.cudf.QuantileMethod.HIGHER;
import static ai.rapids.cudf.QuantileMethod.LINEAR;
import static ai.rapids.cudf.QuantileMethod.LOWER;
import static ai.rapids.cudf.QuantileMethod.MIDPOINT;
import static ai.rapids.cudf.QuantileMethod.NEAREST;
import static ai.rapids.cudf.TableTest.assertColumnsAreEqual;
import static ai.rapids.cudf.TableTest.assertTablesAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class CuSpatialTest extends CudfTestBase {

  @Test
  void testHaversine() {
    try (
      ColumnVector aLon = ColumnVector.fromDoubles(-180, 180);
      ColumnVector aLat = ColumnVector.fromDoubles(0, 30);
      ColumnVector bLon = ColumnVector.fromDoubles(180, -180);
      ColumnVector bLat = ColumnVector.fromDoubles(0, 30);
      ColumnVector result = CuSpatial.haversineDistance(aLon, aLat, bLon, bLat);
      ColumnVector expected = ColumnVector.fromDoubles(1.5604449514735574e-12, 1.3513849691832763e-12)
    ) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testPointsInSpatialWindow() {
    double windowMinX = 1.5;
    double windowMaxX = 5.5;
    double windowMinY = 1.5;
    double windowMaxY = 5.5;
    try (
      ColumnVector pointsX = ColumnVector.fromDoubles(1.0, 2.0, 3.0, 5.0, 7.0, 1.0, 2.0, 3.0, 6.0, 0.0, 3.0, 6.0);
      ColumnVector pointsY = ColumnVector.fromDoubles(0.0, 1.0, 2.0, 3.0, 1.0, 3.0, 5.0, 6.0, 5.0, 4.0, 7.0, 4.0);
      Table result = CuSpatial.pointsInSpatialWindow(windowMinX, windowMaxX, windowMinY, windowMaxY, pointsX, pointsY);
      ColumnVector expectedPointsX = ColumnVector.fromDoubles(3.0, 5.0, 2.0);
      ColumnVector expectedPointsY = ColumnVector.fromDoubles(2.0, 3.0, 5.0);
    ) {
      assertColumnsAreEqual(expectedPointsX, result.getColumn(0));
      assertColumnsAreEqual(expectedPointsY, result.getColumn(1));
    }
  }

  @Test
  void testDeriveTrajectories() throws Exception {
      try (
              ColumnVector objectId = ColumnVector.fromInts(1, 2, 3, 4);
              ColumnVector timestamp = ColumnVector.timestampSecondsFromLongs(0000000000001L, 2000000000000L, 2000000000001L, 2000000000002L);
              ColumnVector pointsX = ColumnVector.fromDoubles(1.0, 2.0, 3.0, 5.0);
              ColumnVector pointsY = ColumnVector.fromDoubles(0.0, 1.0, 2.0, 3.0);
              Pair<Table, ColumnVector> result = CuSpatial.deriveTrajectories(objectId, pointsX, pointsY, timestamp)
      ) {
        Table resultTable = result.getLeft();
        ColumnVector resultColumn = result.getRight();
      }
  }

  @Test
  void testDeriveTrajectoriesThrowsException() {
    assertThrows(CuSpatialException.class, () -> {
      try (
              ColumnVector objectId = ColumnVector.fromInts(1, 2, 3, 4);
              ColumnVector timestamp = ColumnVector.timestampSecondsFromLongs(0000000000001L, 2000000000000L, 2000000000001L, 2000000000002L);
              ColumnVector pointsX = ColumnVector.fromDoubles(1.0, 2.0, 3.0, 5.0, 3.0);//size mismatch
              ColumnVector pointsY = ColumnVector.fromDoubles(0.0, 1.0, 2.0, 3.0);
              Pair<Table, ColumnVector> result = CuSpatial.deriveTrajectories(objectId, pointsX, pointsY, timestamp)
      ) {}
    });
  }

  @Test
  void testTrajectoryDistancesAndSpeeds() {
    int numTrajectories = 4;
    try (
            ColumnVector objectId = ColumnVector.fromInts(1, 2, 3, 4);
            ColumnVector timestamp = ColumnVector.timestampSecondsFromLongs(0000000000001L, 2000000000000L, 2000000000001L, 2000000000002L);
            ColumnVector pointsX = ColumnVector.fromDoubles(1.0, 2.0, 3.0, 5.0);
            ColumnVector pointsY = ColumnVector.fromDoubles(0.0, 1.0, 2.0, 3.0);
            Table result = CuSpatial.trajectoryDistancesAndSpeeds(numTrajectories, objectId, pointsX, pointsY, timestamp)
    ) {

    }
  }

  @Test
  void testTrajectoryBoundingBoxes() {
    int numTrajectories = 4;
    try (
            ColumnVector objectId = ColumnVector.fromInts(1, 2, 3, 4);
            ColumnVector pointsX = ColumnVector.fromDoubles(1.0, 2.0, 3.0, 5.0);
            ColumnVector pointsY = ColumnVector.fromDoubles(0.0, 1.0, 2.0, 3.0);
            Table result = CuSpatial.trajectoryBoundingBoxes(numTrajectories, objectId, pointsX, pointsY)
    ) {

    }
  }

  @Test
  void testPointInPolygonOnePolygonOneRing() {
    try (
            ColumnVector testPointX = ColumnVector.fromDoubles(-2.0, 2.0, 0.0, 0.0, -0.5, 0.5, 0.0, 0.0);
            ColumnVector testPointY = ColumnVector.fromDoubles(0.0, 0.0, -2.0, 2.0, 0.0, 0.0, -0.5, 0.5);
            ColumnVector polyOffsets = ColumnVector.fromInts(0);
            ColumnVector polyRingOffsets = ColumnVector.fromInts(0);
            ColumnVector polyPointX = ColumnVector.fromDoubles(-1.0, -1.0, 1.0, 1.0, -1.0);
            ColumnVector polyPointY = ColumnVector.fromDoubles(-1.0, 1.0, 1.0, -1.0, -1.0);
            ColumnVector result = CuSpatial.pointInPolygon(testPointX, testPointY, polyOffsets, polyRingOffsets, polyPointX, polyPointY);
            ColumnVector expected = ColumnVector.fromInts(0, 0, 0, 0, 1, 1, 1, 1)
    ) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testPointInPolygonTwoPolygonsOneRingEach() {
    try (
            ColumnVector testPointX = ColumnVector.fromDoubles(-2.0, 2.0, 0.0, 0.0, -0.5, 0.5, 0.0, 0.0);
            ColumnVector testPointY = ColumnVector.fromDoubles(0.0, 0.0, -2.0, 2.0, 0.0, 0.0, -0.5, 0.5);
            ColumnVector polyOffsets = ColumnVector.fromInts(0, 1);
            ColumnVector polyRingOffsets = ColumnVector.fromInts(0, 5);
            ColumnVector polyPointX = ColumnVector.fromDoubles(-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0);
            ColumnVector polyPointY = ColumnVector.fromDoubles(-1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0);
            ColumnVector result = CuSpatial.pointInPolygon(testPointX, testPointY, polyOffsets, polyRingOffsets, polyPointX, polyPointY);
            ColumnVector expected = ColumnVector.fromInts(0b00, 0b00, 0b00, 0b00, 0b11, 0b11, 0b11, 0b11)
    ) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testLonlatToCartesianMultiple() {
    double cameraLon = -90.66511046;
    double cameraLat = 42.49197018;
    try {
      try (
              ColumnVector pointLon = ColumnVector.fromDoubles(-90.664973, -90.665393, -90.664976, -90.664537);
              ColumnVector pointLat = ColumnVector.fromDoubles(42.493894, 42.491520, 42.491420, 42.493823);
              Pair<ColumnVector, ColumnVector> result = CuSpatial.lonlatToCartesian(cameraLon, cameraLat, pointLon, pointLat);
              ColumnVector expectedPointsX = ColumnVector.fromDoubles(-0.01126195531216838, 0.02314864865181343, -0.01101638630252916, -0.04698301003584082);
              ColumnVector expectedPointsY = ColumnVector.fromDoubles(-0.21375777777718794, 0.05002000000015667, 0.06113111111163663, -0.20586888888847929);
      ) {
        assertColumnsAreEqual(expectedPointsX, result.getLeft());
        assertColumnsAreEqual(expectedPointsY, result.getRight());
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

}
