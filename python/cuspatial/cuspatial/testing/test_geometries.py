# Copyright (c) 2023, NVIDIA CORPORATION.
from shapely.geometry import LineString, Point, Polygon

"""The fundamental set of tests. This section is dispatched based
on the feature type. Each feature pairing has a specific set of
comparisons that need to be performed to cover the entire test
space. This section contains specific feature representations
that cover all possible geometric combinations."""


point_polygon = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
features = {
    "point-point-disjoint": (
        """Two points apart.""",
        Point(0.0, 0.0),
        Point(1.0, 0.0),
    ),
    "point-point-equal": (
        """Two points together.""",
        Point(0.0, 0.0),
        Point(0.0, 0.0),
    ),
    "point-linestring-disjoint": (
        """Point and linestring are disjoint.""",
        Point(0.0, 0.0),
        LineString([(1.0, 0.0), (2.0, 0.0)]),
    ),
    "point-linestring-point": (
        """Point and linestring share a point.""",
        Point(0.0, 0.0),
        LineString([(0.0, 0.0), (2.0, 0.0)]),
    ),
    "point-linestring-edge": (
        """Point and linestring intersect.""",
        Point(0.5, 0.0),
        LineString([(0.0, 0.0), (1.0, 0.0)]),
    ),
    "point-polygon-disjoint": (
        """Point and polygon are disjoint.""",
        Point(-0.5, 0.5),
        point_polygon,
    ),
    "point-polygon-point": (
        """Point and polygon share a point.""",
        Point(0.0, 0.0),
        point_polygon,
    ),
    "point-polygon-edge": (
        """Point and polygon intersect.""",
        Point(0.5, 0.0),
        point_polygon,
    ),
    "point-polygon-in": (
        """Point is in polygon interior.""",
        Point(0.5, 0.5),
        point_polygon,
    ),
    "linestring-linestring-disjoint": (
        """
    x---x

    x---x
    """,
        LineString([(0.0, 0.0), (1.0, 0.0)]),
        LineString([(0.0, 1.0), (1.0, 1.0)]),
    ),
    "linestring-linestring-same": (
        """
    x---x
    """,
        LineString([(0.0, 0.0), (1.0, 0.0)]),
        LineString([(0.0, 0.0), (1.0, 0.0)]),
    ),
    "linestring-linestring-covers": (
        """
        x
       x
      /
     x
    x
    """,
        LineString([(0.0, 0.0), (1.0, 1.0)]),
        LineString([(0.25, 0.25), (0.5, 0.5)]),
    ),
    "linestring-linestring-touches": (
        """
    x
    |
    |
    |
    x---x
    """,
        LineString([(0.0, 0.0), (0.0, 1.0)]),
        LineString([(0.0, 0.0), (1.0, 0.0)]),
    ),
    "linestring-linestring-touch-interior": (
        """
    x   x
    |  /
    | /
    |/
    x---x
    """,
        LineString([(0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]),
        LineString([(0.0, 0.0), (1.0, 1.0)]),
    ),
    "linestring-linestring-touch-edge": (
        """
      x
      |
      |
      |
    x-x-x
    """,
        LineString([(0.0, 0.0), (1.0, 0.0)]),
        LineString([(0.5, 0.0), (0.5, 1.0)]),
    ),
    "linestring-linestring-touch-edge-twice": (
        """
        x
       x
      / \\
     x---x
    x
    """,
        LineString([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]),
        LineString([(0.25, 0.25), (1.0, 0.0), (0.5, 0.5)]),
    ),
    "linestring-linestring-crosses": (
        """
      x
      |
    x-|-x
      |
      x
    """,
        LineString([(0.5, 0.0), (0.5, 1.0)]),
        LineString([(0.0, 0.5), (1.0, 0.5)]),
    ),
    "linestring-linestring-touch-and-cross": (
        """
        x
        |
        x
        |\\
      x---x
        x
    """,
        LineString([(0.0, 0.0), (1.0, 1.0)]),
        LineString([(0.5, 0.5), (1.0, 0.1), (-1.0, 0.1)]),
    ),
    "linestring-polygon-disjoint": (
        """
    point_polygon above is drawn as
    -----
    |   |
    |   |
    |   |
    -----
    and the corresponding linestring is drawn as
    x---x
    or
    x
    |
    |
    |
    x
    """
        """
    x -----
    | |   |
    | |   |
    | |   |
    x -----
    """,
        LineString([(-0.5, 0.0), (-0.5, 1.0)]),
        point_polygon,
    ),
    "linestring-polygon-touch-point": (
        """
    x---x----
        |   |
        |   |
        |   |
        -----
    """,
        LineString([(-1.0, 0.0), (0.0, 0.0)]),
        point_polygon,
    ),
    "linestring-polygon-touch-edge": (
        """
        -----
        |   |
    x---x   |
        |   |
        -----
    """,
        LineString([(-1.0, 0.5), (0.0, 0.5)]),
        point_polygon,
    ),
    "linestring-polygon-overlap-edge": (
        """
    x----
    |   |
    |   |
    |   |
    x----
    """,
        LineString([(0.0, 0.0), (0.0, 1.0)]),
        point_polygon,
    ),
    "linestring-polygon-intersect-edge": (
        """
      -----
      |   |
      |   |
      |   |
    x---x--
    """,
        LineString([(-0.5, 0.0), (0.5, 0.0)]),
        point_polygon,
    ),
    "linestring-polygon-intersect-inner-edge": (
        """
    -----
    x   |
    |   |
    x   |
    -----

    The linestring in this case is shorter than the corners of the polygon.
    """,
        LineString([(0.25, 0.0), (0.75, 0.0)]),
        point_polygon,
    ),
    "linestring-polygon-point-interior": (
        """
    ----x
    |  /|
    | / |
    |/  |
    x----
    """,
        LineString([(0.0, 0.0), (1.0, 1.0)]),
        point_polygon,
    ),
    "linestring-polygon-edge-interior": (
        """
    --x--
    | | |
    | | |
    | | |
    --x--
    """,
        LineString([(0.5, 0.0), (0.5, 1.0)]),
        point_polygon,
    ),
    "linestring-polygon-in": (
        """
    -----
    | x |
    | | |
    | x |
    -----
    """,
        LineString([(0.5, 0.25), (0.5, 0.75)]),
        point_polygon,
    ),
    "linestring-polygon-in-out": (
        """
    -----
    |   |
    | x |
    | | |
    --|--
      |
      x
    """,
        LineString([(0.5, 0.5), (0.5, -0.5)]),
        point_polygon,
    ),
    "linestring-polygon-crosses": (
        """
      x
    --|--
    | | |
    | | |
    | | |
    --|--
      x
    """,
        LineString([(0.5, 1.25), (0.5, -0.25)]),
        point_polygon,
    ),
    "linestring-polygon-cross-concave-edge": (
        """
    x  x  x
    |\\ | /|
    | xx- |
    |  |  |
    ---x---
    """,
        LineString([(0.5, 0.0), (0.5, 1.0)]),
        Polygon([(0, 0), (0, 1), (0.3, 0.4), (1, 1), (1, 0)]),
    ),
    "linestring-polygon-half-in": (
        """
    -----
    |   |
    | x |
    |/ \\|
    xx-xx
    """,
        LineString(
            [(0.0, 0.0), (0.25, 0.0), (0.5, 0.5), (0.75, 0.0), (1.0, 0.0)]
        ),
        point_polygon,
    ),
    "linestring-polygon-half-out": (
        """
    -----
    |   |
    |   |
    |   |
    xx-xx
     \\/
      x
    """,
        LineString(
            [(0.0, 0.0), (0.25, 0.0), (0.5, -0.5), (0.75, 0.0), (1.0, 0.0)]
        ),
        point_polygon,
    ),
    "linestring-polygon-two-edges": (
        """
    x----
    |   |
    |   |
    |   |
    x---x
    """,
        LineString([(0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]),
        point_polygon,
    ),
    "linestring-polygon-edge-to-interior": (
        """
    x----
    |   |
    |  -x
    |-/ |
    x----
    """,
        LineString([(0.0, 1.0), (0.0, 0.0), (1.0, 0.5)]),
        point_polygon,
    ),
    "linestring-polygon-edge-cross-to-exterior": (
        """
    x------
    |     |
    |    ---x
    | --- |
    x------
    """,
        LineString([(0.0, 1.0), (0.0, 0.0), (1.5, 0.5)]),
        point_polygon,
    ),
    "polygon-polygon-disjoint": (
        """
    Polygon polygon tests use a triangle for the lhs and a square for the rhs.
    The triangle is drawn as
    x---x
    |  /
    | /
    |/
    x

    The square is drawn as

    -----
    |   |
    |   |
    |   |
    -----
    """,
        Polygon([(0.0, 2.0), (0.0, 3.0), (1.0, 3.0)]),
        point_polygon,
    ),
    "polygon-polygon-touch-point": (
        """
    x---x
    |  /
    | /
    |/
    x----
    |   |
    |   |
    |   |
    -----
    """,
        Polygon([(0.0, 1.0), (0.0, 2.0), (1.0, 2.0)]),
        point_polygon,
    ),
    "polygon-polygon-touch-edge": (
        """
     x---x
     |  /
     | /
     |/
    -x--x
    |   |
    |   |
    |   |
    -----
    """,
        Polygon([(0.25, 1.0), (0.25, 2.0), (1.25, 2.0)]),
        point_polygon,
    ),
    "polygon-polygon-overlap-edge": (
        """
    x
    |\\
    | \\
    |  \\
    x---x
    |   |
    |   |
    |   |
    -----
    """,
        Polygon([(0.0, 1.0), (0.0, 2.0), (1.0, 2.0)]),
        point_polygon,
    ),
    "polygon-polygon-overlap-inside-edge": (
        """
          x
         /|
    x---x |
    \\ /  |
      x   |
     /    |
    x-----x
    """,
        Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
        Polygon([(0.25, 0.25), (0.5, 0.5), (0, 0.5), (0.25, 0.25)]),
    ),
    "polygon-polygon-point-inside": (
        """
      x---x
      |  /
    --|-/
    | |/|
    | x |
    |   |
    -----
    """,
        Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5)]),
        point_polygon,
    ),
    "polygon-polygon-point-outside": (
        """
     x
    -|\\--
    |x-x|
    |   |
    |   |
    -----
    """,
        Polygon([(0.25, 0.75), (0.25, 1.25), (0.75, 0.75)]),
        point_polygon,
    ),
    "polygon-polygon-in-out-point": (
        """
      x
      |\\
    --|-x
    | |/|
    | x |
    |   |
    x----
    """,
        Polygon([(0.5, 0.5), (0.5, 1.5), (1.0, 1.0)]),
        point_polygon,
    ),
    "polygon-polygon-in-point-point": (
        """
    x----
    |\\  |
    | x |
    |/  |
    x----
    """,
        Polygon([(0.0, 0.0), (0.0, 1.0), (0.5, 0.5)]),
        point_polygon,
    ),
    "polygon-polygon-contained": (
        """
    -----
    |  x|
    | /||
    |x-x|
    -----
    """,
        Polygon([(0.25, 0.25), (0.75, 0.75), (0.75, 0.25)]),
        point_polygon,
    ),
    "polygon-polygon-same": (
        """
    x---x
    |   |
    |   |
    |   |
    x---x
    """,
        point_polygon,
        point_polygon,
    ),
}

point_point_dispatch_list = [
    "point-point-disjoint",
    "point-point-equal",
]

point_linestring_dispatch_list = [
    "point-linestring-disjoint",
    "point-linestring-point",
    "point-linestring-edge",
]

point_polygon_dispatch_list = [
    "point-polygon-disjoint",
    "point-polygon-point",
    "point-polygon-edge",
    "point-polygon-in",
]

linestring_linestring_dispatch_list = [
    "linestring-linestring-disjoint",
    "linestring-linestring-same",
    "linestring-linestring-covers",
    "linestring-linestring-touches",
    "linestring-linestring-touch-interior",
    "linestring-linestring-touch-edge",
    "linestring-linestring-touch-edge-twice",
    "linestring-linestring-crosses",
    "linestring-linestring-touch-and-cross",
]

linestring_polygon_dispatch_list = [
    "linestring-polygon-disjoint",
    "linestring-polygon-touch-point",
    "linestring-polygon-touch-edge",
    "linestring-polygon-overlap-edge",
    "linestring-polygon-intersect-edge",
    "linestring-polygon-intersect-inner-edge",
    "linestring-polygon-point-interior",
    "linestring-polygon-edge-interior",
    "linestring-polygon-in",
    "linestring-polygon-crosses",
    "linestring-polygon-cross-concave-edge",
    "linestring-polygon-half-in",
    "linestring-polygon-half-out",
    "linestring-polygon-two-edges",
    "linestring-polygon-edge-to-interior",
    "linestring-polygon-edge-cross-to-exterior",
]

polygon_polygon_dispatch_list = [
    "polygon-polygon-disjoint",
    "polygon-polygon-touch-point",
    "polygon-polygon-touch-edge",
    "polygon-polygon-overlap-edge",
    "polygon-polygon-overlap-inside-edge",
    "polygon-polygon-point-inside",
    "polygon-polygon-point-outside",
    "polygon-polygon-in-out-point",
    "polygon-polygon-in-point-point",
    "polygon-polygon-contained",
    "polygon-polygon-same",
]
