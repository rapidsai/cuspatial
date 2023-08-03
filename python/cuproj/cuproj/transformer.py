import cupy as cp

from cuproj._lib.transform import Transformer as _Transformer


class Transformer:
    """A transformer object to transform coordinates from one CRS to another.

    Notes
    -----
    Currently only the EPSG authority is supported.
    Currently only projection from WGS84 to UTM (and vice versa)
    is supported.

    Examples
    --------
    >>> from cuproj import Transformer
    >>> transformer = Transformer.from_crs("epsg:4326", "epsg:32631")
    >>> transformer.transform(2, 49)
    (500000.0, 5460836.5)
    >>> transformer.transform(500000, 5460836.5, direction="INVERSE")
    (2.0, 49.0)
    """

    def __init__(self, crs_from, crs_to):
        """Construct a Transformer object.

        Parameters
        ----------
        from_crs : CRS
            The source CRS.
        to_crs : CRS
            The target CRS.

        Source and target CRS may be:
        - An authority string [i.e. 'epsg:4326']
        - An EPSG integer code [i.e. 4326]
        - A tuple of (“auth_name”: “auth_code”) [i.e ('epsg', '4326')]

        Notes
        -----
        Currently only the EPSG authority is supported.

        Examples
        --------
        >>> from cuproj import Transformer
        >>> transformer = Transformer("epsg:4326", "epsg:32631")
        >>> transformer = Transformer(4326, 32631)
        """
        self._crs_from = crs_from
        self._crs_to = crs_to
        self._proj = _Transformer(crs_from, crs_to)

    @staticmethod
    def from_crs(crs_from, crs_to):
        """Create a transformer from a source CRS to a target CRS.

        Parameters
        ----------
        crs_from : CRS
            The source CRS.
        crs_to : CRS
            The target CRS.

        Source and target CRS may be:
        - An authority string [i.e. 'epsg:4326']
        - An EPSG integer code [i.e. 4326]
        - A tuple of (“auth_name”: “auth_code”) [i.e ('epsg', '4326')]

        Notes
        -----
        Currently only the EPSG authority is supported.

        Returns
        -------
        Transformer
            A transformer object to transform coordinates from one CRS to
            another.
        """
        return Transformer(crs_from, crs_to)

    def transform(self, x, y, direction="FORWARD"):
        """Transform coordinates from one CRS to another.

        If the input data is already in device memory, and the input implements
        `__cuda_array_interface__
        <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_
        , the data will be used directly. If the data is in host memory, it
        will be copied to the device.

        Parameters
        ----------
        x : float or array_like
            The x coordinate(s) to transform.
        y : float or array_like
            The y coordinate(s) to transform.
        direction : str, optional
            The direction of the transformation. Either "FORWARD" or "INVERSE".
            Default is "FORWARD".

        Returns
        -------
        tuple
            A tuple of transformed x and y coordinates as cupy (device) arrays.
        """

        if direction not in ("FORWARD", "INVERSE"):
            raise ValueError(f"Invalid direction: {direction}")

        isfloat = False
        if isinstance(x, float) and isinstance(y, float):
            isfloat = True
            x = cp.asarray([x], dtype='f8')
            y = cp.asarray([y], dtype='f8')
        else:
            x = cp.asarray(x, x.dtype)
            y = cp.asarray(y, y.dtype)

        resx, resy = self._proj.transform(x, y, direction)

        if isfloat:
            resx, resy = resx.get()[0], resy.get()[0]
        return resx, resy
