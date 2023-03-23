# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from cudf import Series

if TYPE_CHECKING:
    from cuspatial.core.geoseries import GeoSeries


class BinPred(ABC):
    """Base class for binary predicates. This class is an abstract base class
    and should not be instantiated directly. Child classes exist for each
    combination of left-hand and right-hand GeoSeries types and binary
    predicates. For example, a `PointPointContains` predicate is a child class
    of `BinPred` that implements the `contains_properly` predicate for two
    `Point` GeoSeries.

    Parameters
    ----------
    lhs : GeoSeries
        The left-hand GeoSeries.
    rhs : GeoSeries
        The right-hand GeoSeries.
    **kwargs
        Any additional arguments to be used at runtime.

    Attributes
    ----------
    lhs : GeoSeries
        The left-hand GeoSeries.
    rhs : GeoSeries
        The right-hand GeoSeries.
    kwargs : dict
        Any additional arguments to be used at runtime.

    Methods
    -------
    __call__(self, lhs, rhs)
        System call for the binary predicate. Calls the _call method, which is
        implemented by the subclass.
    _call(self, lhs, rhs)
        Call the binary predicate. This method is implemented by the subclass.
    _preprocess(self, lhs, rhs)
        Preprocess the left-hand and right-hand GeoSeries. This method is
        implemented by the subclass.
    _op(self, lhs, rhs)
        Compute the binary predicate between two GeoSeries. This method is
        implemented by the subclass.
    _postprocess(self, lhs, rhs, point_indices, op_result)
        Postprocess the output GeoSeries to ensure that they are of the correct
        type for the predicate. This method is implemented by the subclass.

    Notes
    -----
    BinPred classes are selected using the appropriate dispatch function. For
    example, the `contains_properly` predicate is selected using the
    `contains_dispatch` dispatch function. The dispatch function selects the
    appropriate BinPred class based on the left-hand and right-hand GeoSeries
    types.

    This enables customized behavior for each combination of left-hand and
    right-hand GeoSeries types and binary predicates. For example, the
    `contains_properly` predicate for two `Point` GeoSeries is implemented
    using a `PointPointContains` BinPred class. The `contains_properly`
    predicate for a `Point` GeoSeries and a `Polygon` GeoSeries is implemented
    using a `PointPolygonContains` BinPred class. Most subclasses will be able
    to use all or 2/3rds of the methods defined in the `RootContains(BinPred)
    class`. The `RootContains` class implements the `contains_properly`
    predicate for the most common combination of left-hand and right-hand
    GeoSeries types. The `RootContains` class can be used as a template for
    implementing the `contains_properly` predicate for other combinations of
    left-hand and right-hand GeoSeries types.

    Examples
    --------
    >>> from cuspatial.core.binpred.dispatch import contains_properly_dispatch
    >>> from cuspatial.core.geoseries import GeoSeries
    >>> lhs = GeoSeries([Polygon([(0, 0), (1, 1), (1, 0)])])
    >>> rhs = GeoSeries([Point(0, 0), Point(1, 1)])
    >>> predicate = contains_properly_dispatch[(
    ...     lhs.column_type, rhs.column_type
    ... )](lhs, rhs, align, allpairs)
    >>> print(predicate())
    # TODO Output
    """

    @abstractmethod
    def __init__(self, lhs: "GeoSeries", rhs: "GeoSeries", **kwargs):
        """Initialize a binary predicate. Collects any arguments passed
        to the binary predicate to be used at runtime.

        This class stores the left-hand and right-hand GeoSeries and any
        additional arguments to be used at runtime. The left-hand and
        right-hand GeoSeries are stored as attributes of the class. The
        additional arguments are stored as a dictionary in the `kwargs`
        attribute.

        Parameters
        ----------
        lhs : GeoSeries
            The left-hand GeoSeries.
        rhs : GeoSeries
            The right-hand GeoSeries.
        **kwargs
            Any additional arguments to be used at runtime.

        Attributes
        ----------
        lhs : GeoSeries
            The left-hand GeoSeries.
        rhs : GeoSeries
            The right-hand GeoSeries.
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
        _op(self, lhs, rhs)
            Compute the binary predicate between two GeoSeries. This method is
            implemented by the subclass.
        _postprocess(self, lhs, rhs, point_indices, op_result)
            Postprocess the output GeoSeries to ensure that they are of the
            correct type for the predicate. This method is implemented by the
            subclass.

        Examples
        --------
        >>> from cuspatial.core.binpred.dispatch import (
        ...     contains_properly_dispatch
        ... )
        >>> from cuspatial.core.geoseries import GeoSeries
        >>> lhs = GeoSeries([Polygon([(0, 0), (1, 1), (1, 0)])])
        >>> rhs = GeoSeries([Point(0, 0), Point(1, 1)])
        >>> predicate = contains_properly_dispatch[
        ...     (
        ...         lhs.column_type, rhs.column_type
        ...     )](lhs, rhs, align, allpairs)
        >>> print(predicate())
        # TODO Output
        """
        super().__init__(lhs, rhs, **kwargs)
        self.lhs = lhs
        self.rhs = rhs
        self.kwargs = kwargs

    def __call__(self, lhs: "GeoSeries", rhs: "GeoSeries") -> Series:
        """System call for the binary predicate. Calls the _call method,
        which is implemented by the subclass. Executing the binary predicate
        returns the results of the binary predicate as a GeoSeries.

        Examples
        --------
        >>> from cuspatial.core.binpred.dispatch import (
        ...     contains_properly_dispatch
        ... )
        >>> from cuspatial.core.geoseries import GeoSeries
        >>> lhs = GeoSeries([Polygon([(0, 0), (1, 1), (1, 0)])])
        >>> rhs = GeoSeries([Point(0, 0), Point(1, 1)])
        >>> predicate = contains_properly_dispatch[
        ...     (
        ...         lhs.column_type, rhs.column_type
        ...     )](lhs, rhs, align, allpairs)
        >>> print(predicate())
        # TODO Output
        """
        return self._call(self, lhs, rhs)

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
        result : GeoSeries
            A GeoSeries of boolean values indicating whether each feature in
            the right-hand GeoSeries satisfies the requirements of a binary
            predicate with its corresponding feature in the left-hand
            GeoSeries.
        """
        return self._preprocess(self, lhs, rhs)

    def _preprocess(self, lhs: GeoSeries, rhs: GeoSeries) -> Series:
        """Preprocess the left-hand and right-hand GeoSeries. This method
        is implemented by the subclass.

        Preprocessing is used to ensure that the left-hand and right-hand
        GeoSeries are of the correct type for each of the three basic
        predicates: 'contains', 'intersects', and 'equals'. For example,
        `contains` requires that the left-hand GeoSeries be polygons or
        multipolygons and the right-hand GeoSeries be points or multipoints.
        `intersects` requires that the left-hand GeoSeries be linestrings or
        and the right-hand GeoSeries be linestrings or points.
        `equals` requires that the left-hand and right-hand GeoSeries be
        points.

        Subclasses that implement `_preprocess` are responsible for calling
        `_op` to continue the execution of the binary predicate. The last
        line of `_preprocess` should be

            return self._op(lhs, rhs, points, point_indices)

        # TODO: Change `_op` signature to take `lhs` and `rhs` only?
        where `points` is a GeoSeries of points and `point_indices` is a
        cudf.Series of indices that map each point in `points` to its
        corresponding feature in the right-hand GeoSeries.

        Parameters
        ----------
        lhs : GeoSeries
            The original left-hand GeoSeries.
        rhs : GeoSeries
            The original right-hand GeoSeries.

        Returns
        -------
        result : GeoSeries
            A GeoSeries of boolean values indicating whether each feature in
            the right-hand GeoSeries satisfies the requirements of a binary
            predicate with its corresponding feature in the left-hand
            GeoSeries.
        """
        # TODO: Update this as changing `_op` signature is complete.
        points = None
        point_indices = None
        return self._op(lhs, rhs, points, point_indices)

    # TODO Change `_op` signature to be more generic.
    def _op(
        self,
        lhs: "GeoSeries",
        rhs: "GeoSeries",
        points: "GeoSeries",
        point_indices: Series,
    ) -> Series:
        """Compute the binary predicate between two GeoSeries. This method
        is implemented by the subclass. This method is called by `_preprocess`
        to continue the execution of the binary predicate. `_op` is responsible
        for calling `_postprocess` to complete the execution of the binary
        predicate.

        Op is used to compute the binary predicate, or composition of binary
        predicates, between two GeoSeries. The left-hand GeoSeries is
        considered the "base" GeoSeries and the right-hand GeoSeries is
        considered the "other" GeoSeries. The binary predicate is computed
        between each feature in the base GeoSeries and the other GeoSeries.
        The result is a GeoSeries of boolean values indicating whether each
        feature in the other GeoSeries satisfies the requirements of a binary
        predicate with its corresponding feature in the base GeoSeries.

        Subclasses that implement `_op` are responsible for calling
        `_postprocess` to complete the execution of the binary predicate. The
        last line of `_op` should be

            return self._postprocess(lhs, rhs, points, point_indices)

        where `points` is a GeoSeries of points and `point_indices` is a
        cudf.Series of indices that map each point in `points` to its
        corresponding feature in the right-hand GeoSeries.

        Parameters
        ----------
        lhs : GeoSeries
            The left-hand GeoSeries.
        rhs : GeoSeries
            The right-hand GeoSeries.
        points : GeoSeries
            A GeoSeries of points.
        point_indices : cudf.Series
            A cudf.Series of indices that map each point in `points` to its
            corresponding feature in the right-hand GeoSeries.
        """
        pass

    def _postprocess(
        self,
        lhs: "GeoSeries",
        rhs: "GeoSeries",
        points: "GeoSeries",
        point_indices: Series,
    ) -> Series:
        """Postprocess the output GeoSeries to ensure that they are of the
        correct type for the predicate.

        TODO: Update

        Postprocess for contains_properly has to handle multiple input and
        output configurations.

        The input can be a single polygon, a single multipolygon, or a
        GeoSeries containing a mix of polygons and multipolygons.

        The input to postprocess is `point_indices`, which can be either a
        cudf.DataFrame with one row per point and one column per polygon or
        a cudf.DataFrame containing the point index and the part index for
        each point in the polygon.
        """
        pass


class NotImplementedRoot(BinPred):
    """A class that is used to raise an error when a binary predicate is
    not implemented for a given combination of left-hand and right-hand
    GeoSeries types.
    """

    def __init__(self):
        raise NotImplementedError
