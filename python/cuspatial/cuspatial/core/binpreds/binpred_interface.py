# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from typing import TYPE_CHECKING, Tuple

from cudf import Series

from cuspatial.utils.binpred_utils import _false_series

if TYPE_CHECKING:
    from cuspatial.core.geoseries import GeoSeries


class BinPredConfig:
    """Configuration for a binary predicate.

    Parameters
    ----------
    align : bool
        Whether to align the left-hand and right-hand GeoSeries before
        computing the binary predicate. Defaults to True.
    allpairs : bool
        Whether to compute the binary predicate between all pairs of
        features in the left-hand and right-hand GeoSeries. Defaults to
        False. Only available with the contains predicate.
    mode: str
        The mode to use when computing the binary predicate. Defaults to
        "full". Only available with the contains predicate and used
        for internal operations.
    """

    def __init__(self, **kwargs):
        self.align = kwargs.get("align", True)
        self.kwargs = kwargs

    def __repr__(self):
        return f"BinPredConfig(align={self.align}, kwargs={self.kwargs})"

    def __str__(self):
        return self.__repr__()


class PreprocessorResult:
    """Result of a binary predicate preprocessor. The following classes
    are used to give all implementors of `BinaryItf` a common interface
    for preprocessor results.

    Parameters
    ----------
    lhs : GeoSeries
        The left-hand GeoSeries.
    rhs : GeoSeries
        The right-hand GeoSeries.
    final_rhs : GeoSeries
        The rhs GeoSeries, if modified by the preprocessor. For example
        the contains preprocessor converts any complex feature type into
        a collection of points.
    point_indices : cudf.Series
        A cudf.Series of indices that map each point in `points` to its
        corresponding feature in the right-hand GeoSeries.
    """

    def __init__(
        self,
        lhs: "GeoSeries",
        rhs: "GeoSeries",
        final_rhs: "GeoSeries" = None,
        point_indices: Series = None,
    ):
        self.lhs = lhs
        self.rhs = rhs
        self.final_rhs = final_rhs
        self.point_indices = point_indices

    def __repr__(self):
        return f"PreprocessorResult(lhs={self.lhs}, rhs={self.rhs}, \
        points={self.final_rhs}, point_indices={self.point_indices})"

    def __str__(self):
        return self.__repr__()


class OpResult:
    """Result of a binary predicate operation."""

    pass


class ContainsOpResult(OpResult):
    """Result of a Contains binary predicate operation.

    Parameters
    ----------
    pip_result : cudf.DataFrame
        A cudf.DataFrame containing two columns: "polygon_index" and
        Point_index". The "polygon_index" column contains the index of
        the polygon that contains each point. The "point_index" column
        contains the index of each point that is contained by a polygon.
    intersection_result: Tuple (optional)
        A tuple containing the result of the intersection operation
        between the left-hand GeoSeries and the right-hand GeoSeries.
        Used in .contains_properly.
    """

    def __init__(
        self,
        pip_result: Series,
        preprocessor_result: PreprocessorResult,
        intersection_result: Tuple = None,
    ):
        self.pip_result = pip_result
        self.preprocessor_result = preprocessor_result
        self.intersection_result = intersection_result

    def __repr__(self):
        return f"OpResult(pip_result={self.pip_result},\n \
        preprocessor_result={self.preprocessor_result},\n \
        intersection_result={self.intersection_result})\n"

    def __str__(self):
        return self.__repr__()


class EqualsOpResult(OpResult):
    """Result of an Equals binary predicate operation.

    Parameters
    ----------
    result : cudf.Series
        A cudf.Series of boolean values indicating whether each feature in
        the right-hand GeoSeries is equal to the point in the left-hand
        GeoSeries.
    point_indices: cudf.Series
        A cudf.Series of indices that map each point in `points` to its
        corresponding feature in the right-hand GeoSeries.
    """

    def __init__(self, result: Series, point_indices: Series):
        self.result = result
        self.point_indices = point_indices

    def __repr__(self):
        return f"OpResult(result={self.result}) \
        point_indices={self.point_indices}"

    def __str__(self):
        return self.__repr__()


class IntersectsOpResult(OpResult):
    """Result of an Intersection binary predicate operation."""

    def __init__(self, result: Tuple):
        self.result = result

    def __repr__(self):
        return f"OpResult(result={self.result})"

    def __str__(self):
        return self.__repr__()


class BinPred:
    """Base class for binary predicates. This class is an abstract base class
    and can not be instantiated directly. `BinPred` is the base class that
    implements this interface in the most general case. Child classes exist
    for each combination of left-hand and right-hand GeoSeries types and binary
    predicates. For example, a `PointPointContains` predicate is a child class
    of `BinPred` that implements the `contains_properly` predicate for two
    `Point` GeoSeries. These classes are found in the `feature_<predicateName>`
    files found in this directory.

    Notes
    -----
    BinPred classes are selected using the appropriate dispatch function. For
    example, the `contains_properly` predicate is selected using the
    `CONTAINS_DISPATCH` dispatch function from `binpred_dispatch.py`. The
    dispatch function selects the appropriate BinPred class based on the
    left-hand and right-hand GeoSeries types.

    This enables customized behavior for each combination of left-hand and
    right-hand GeoSeries types and binary predicates. For example, the
    `contains_properly` predicate for two `Point` GeoSeries is implemented
    using a `PointPointContains` BinPred class. The `contains_properly`
    predicate for a `Point` GeoSeries and a `Polygon` GeoSeries is implemented
    using a `PointPolygonContains` BinPred class. Most subclasses will be able
    to use all or 2/3rds of the methods defined in the `RootContains(BinPred)
    class`.

    The `RootContains` class implements the `contains_properly` predicate for
    the most common combination of left-hand and right-hand GeoSeries types.
    The `RootContains` class can be used as a template for implementing the
    `contains_properly` predicate for other combinations of left-hand and
    right-hand GeoSeries types.

    Examples
    --------
    >>> from cuspatial.core.binpreds.binpred_dispatch import CONTAINS_DISPATCH
    >>> from cuspatial.core.geoseries import GeoSeries
    >>> from shapely.geometry import Point, Polygon
    >>> predicate = CONTAINS_DISPATCH[(
    ...     lhs.column_type, rhs.column_type
    ... )](align=True, allpairs=False)
    >>> lhs = GeoSeries([Polygon([(0, 0), (1, 1), (1, 0)])])
    >>> rhs = GeoSeries([Point(0, 0), Point(1, 1)])
    >>> print(predicate(lhs, rhs))
    0    False
    dtype: bool
    """

    def __init__(self, **kwargs):
        """Initialize a binary predicate. Collects any arguments passed
        to the binary predicate to be used at runtime.

        This class stores the config object that can be passed to the binary
        predicate at runtime. The lhs and rhs are set at runtime using the
        __call__ method so that the same binary predicate can be used for
        multiple left-hand and right-hand GeoSeries.

        Parameters
        ----------
        **kwargs
            Any additional arguments to be used at runtime.

        Attributes
        ----------
        kwargs : dict
            Any additional arguments to be used at runtime.

        Methods
        -------
        __call__(self, lhs, rhs)
            System call for the binary predicate. Calls the _call method, which
            is implemented by the subclass.
        _call(self, lhs, rhs)
            Call the binary predicate. This method is implemented by the
            subclass.
        _preprocess(self, lhs, rhs)
            Preprocess the left-hand and right-hand GeoSeries. This method is
            implemented by the subclass.
        _compute_predicate(self, lhs, rhs)
            Compute the binary predicate between two GeoSeries. This method is
            implemented by the subclass.
        _postprocess(self, lhs, rhs, point_indices, op_result)
            Postprocess the output GeoSeries to ensure that they are of the
            correct type for the predicate. This method is implemented by the
            subclass.

        Examples
        --------
        >>> from cuspatial.core.binpreds.binpred_dispatch import (
        ...     CONTAINS_DISPATCH
        ... )
        >>> from cuspatial.core.geoseries import GeoSeries
        >>> from shapely.geometry import Point, Polygon
        >>> predicate = CONTAINS_DISPATCH[(
        ...     lhs.column_type, rhs.column_type
        ... )](align=True, allpairs=False)
        >>> lhs = GeoSeries([Polygon([(0, 0), (1, 1), (1, 0)])])
        >>> rhs = GeoSeries([Point(0, 0), Point(1, 1)])
        >>> print(predicate(lhs, rhs))
        0    False
        dtype: bool
        """
        self.config = BinPredConfig(**kwargs)

    def __call__(self, lhs: "GeoSeries", rhs: "GeoSeries") -> Series:
        """System call for the binary predicate. Calls the _call method,
        which is implemented by the subclass. Executing the binary predicate
        returns the results of the binary predicate as a GeoSeries.

        Parameters
        ----------
        lhs : GeoSeries
            The left-hand GeoSeries.
        rhs : GeoSeries
            The right-hand GeoSeries.

        Returns
        -------
        result : Series
            The results of the binary predicate.

        Examples
        --------
        >>> from cuspatial.core.binpreds.binpred_dispatch import (
        ...     CONTAINS_DISPATCH
        ... )
        >>> from cuspatial.core.geoseries import GeoSeries
        >>> from shapely.geometry import Point, Polygon
        >>> predicate = CONTAINS_DISPATCH[(
        ...     lhs.column_type, rhs.column_type
        ... )](align=True, allpairs=False)
        >>> lhs = GeoSeries([Polygon([(0, 0), (1, 1), (1, 0)])])
        >>> rhs = GeoSeries([Point(0, 0), Point(1, 1)])
        >>> print(predicate(lhs, rhs))
        0    False
        dtype: bool
        """
        return self._call(lhs, rhs)

    def _call(self, lhs: "GeoSeries", rhs: "GeoSeries") -> Series:
        """Call the binary predicate. This method is implemented by the
        subclass.

        Parameters
        ----------
        lhs : GeoSeries
            The left-hand GeoSeries.
        rhs : GeoSeries
            The right-hand GeoSeries.

        Returns
        -------
        result : Series
            A cudf.Series of boolean values indicating whether each feature in
            the right-hand GeoSeries satisfies the requirements of a binary
            predicate with its corresponding feature in the left-hand
            GeoSeries.
        """
        return self._preprocess(lhs, rhs)

    def _preprocess(self, lhs: "GeoSeries", rhs: "GeoSeries") -> Series:
        """Preprocess the left-hand and right-hand GeoSeries. This method
        is implemented by the subclass.

        Preprocessing is used to ensure that the left-hand and right-hand
        GeoSeries are of the correct type for each of the three basic
        predicates: 'contains', 'intersects', and 'equals'. For example,
        `contains` requires that the left-hand GeoSeries be polygons or
        multipolygons and the right-hand GeoSeries be points or multipoints.
        `intersects` requires that the left-hand GeoSeries be linestrings or
        points and the right-hand GeoSeries be linestrings or points.
        `equals` requires that the left-hand and right-hand GeoSeries be
        points.

        Subclasses that implement `_preprocess` are responsible for calling
        `_compute_predicate` to continue the execution of the binary predicate.
        The last line of `_preprocess` as implemented by any subclass should be

            return self._compute_predicate(lhs, rhs, points, point_indices)

        Parameters
        ----------
        lhs : GeoSeries
            The original left-hand GeoSeries.
        rhs : GeoSeries
            The original right-hand GeoSeries.

        Returns
        -------
        result : Series
            A cudf.Series of boolean values indicating whether each feature in
            the right-hand GeoSeries satisfies the requirements of a binary
            predicate with its corresponding feature in the left-hand
            GeoSeries.
        """
        raise NotImplementedError

    def _compute_predicate(
        self,
        lhs: "GeoSeries",
        rhs: "GeoSeries",
        preprocessor_result: PreprocessorResult,
    ) -> Series:
        """Compute the binary predicate between two GeoSeries. This method
        is implemented by the subclass. This method is called by `_preprocess`
        to continue the execution of the binary predicate. `_compute_predicate`
        is responsible for calling `_postprocess` to complete the execution of
        the binary predicate.

        `compute_predicate` is used to compute the binary predicate, or
        composition of binary predicates, between two GeoSeries. The left-hand
        GeoSeries is considered the "base" GeoSeries and the right-hand
        GeoSeries is considered the "other" GeoSeries. The binary predicate is
        computed between each feature in the base GeoSeries and the other
        GeoSeries.  The result is a GeoSeries of boolean values indicating
        whether each feature in the other GeoSeries satisfies the requirements
        of a binary predicate with its corresponding feature in the base
        GeoSeries.

        Subclasses that implement `_compute_predicate` are responsible for
        calling `_postprocess` to complete the execution of the binary
        predicate. The last line of `_compute_predicate` should be

            return self._postprocess(
                lhs,
                rhs,
                OpResult(modified_rhs, point_indices)
            )

        where `modified_rhs` is a GeoSeries of points and `point_indices` is a
        cudf.Series of indices that map each point in `points` to its
        corresponding feature in the right-hand GeoSeries.

        Parameters
        ----------
        lhs : GeoSeries
            The left-hand GeoSeries.
        rhs : GeoSeries
            The right-hand GeoSeries.
        preprocessor_result : PreprocessorResult
            The result of the preprocessing step.
        """
        raise NotImplementedError

    def _postprocess(
        self,
        lhs: "GeoSeries",
        rhs: "GeoSeries",
        op_result: OpResult,
    ) -> Series:
        """Postprocess the output GeoSeries to ensure that they are of the
        correct return type for the predicate. This method is implemented by
        the subclass.

        Postprocessing is used to convert the results of the
        `_compute_predicate` call into countable values. This step converts the
        results of one of the three binary predicates `contains`, `intersects`,
        or `equals` into a `Series` of boolean values. When the `rhs` is a
        non-point type, `_postprocess` is responsible for aggregating the
        results of the `_compute_predicate` call into a single boolean value
        for each feature in the `lhs`.

        Parameters
        ----------
        lhs : GeoSeries
            The left-hand GeoSeries.
        rhs : GeoSeries
            The right-hand GeoSeries.
        op_result : cudf.Series
            The result of the `_compute_predicate` call.

        Returns
        -------
        result : Series
            A Series of boolean values indicating whether each feature in
            the right-hand GeoSeries satisfies the requirements of a binary
            predicate with its corresponding feature in the left-hand
            GeoSeries.

        Notes
        -----
        Arithmetic rules incorporated into `_postprocess` classes:

        (a, b) -> a contains b iff for all points p in b, p is in a
        (a, b) -> a intersects b iff for any point p in b, p is in a

        I'm currently looking into refactoring these arithmetics into a
        syntax that more closely resembles it.
        """
        raise NotImplementedError


class NotImplementedPredicate(BinPred):
    """A class that is used to raise an error when a binary predicate is
    not implemented for a given combination of left-hand and right-hand
    GeoSeries types. This is useful for delineating which binary predicates
    are implemented for which GeoSeries types in their appropriate
    `DispatchDict`.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class ImpossiblePredicate(BinPred):
    """There are many combinations that are impossible. This is the base class
    to simply return a series of False values for these cases.
    """

    def _preprocess(self, lhs, rhs):
        return _false_series(len(lhs))
