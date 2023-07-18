from cuproj._lib.transform import wgs84_to_utm

class Transformer:
    """A transformer object to transform coordinates from one CRS to another.

    Parameters
    ----------
    from_crs : CRS
        The source CRS.
    to_crs : CRS
        The target CRS.
    skip_equivalent : bool, optional
        If True, skip transformation if the source and target CRS are equivalent.
        Default is False.

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
        self._crs_from = crs_from
        self._crs_to = crs_to

    @staticmethod
    def from_crs(crs_from, crs_to):
        """Create a transformer from a source CRS to a target CRS.

        Parameters
        ----------
        crs_from : CRS
            The source CRS.
        crs_to : CRS
            The target CRS.

        Returns
        -------
        Transformer
            A transformer object to transform coordinates from one CRS to another.
        """
        return Transformer(crs_from, crs_to)

    def transform(self, x, y, direction="FORWARD"):
        """Transform coordinates from one CRS to another.

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
            A tuple of transformed x and y coordinates.
        """
        if direction == "FORWARD":
            return self._transform_forward(x, y)
        elif direction == "INVERSE":
            return self._transform_inverse(x, y)
        else:
            raise ValueError(f"Invalid direction: {direction}")
