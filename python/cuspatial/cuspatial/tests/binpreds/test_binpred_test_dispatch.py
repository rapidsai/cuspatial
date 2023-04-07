# Copyright (c) 2023, NVIDIA CORPORATION.

from functools import wraps

import pandas as pd
import pytest
from binpred_test_dispatch import (  # noqa: F401
    feature_test_dispatch,
    geotype_tuple,
    predicate,
    test_type,
)


def skip_on_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            pytest.skip(f"Skipping due to an exception: {e}")

    return wrapper


def test_fixtures(geotype_tuple, predicate, test_type):  # noqa: F811
    """Test that the fixture data is correct."""
    (lhs, rhs) = feature_test_dispatch(
        geotype_tuple[0], geotype_tuple[1], test_type
    )
    gpdlhs = lhs.to_geopandas()
    gpdrhs = rhs.to_geopandas()
    pred_fn = getattr(lhs, predicate)
    got = pred_fn(rhs)
    if predicate == "contains_properly":
        predicate = "contains"
    gpd_pred_fn = getattr(gpdlhs, predicate)
    expected = gpd_pred_fn(gpdrhs)
    try:
        pd.testing.assert_series_equal(expected, got.to_pandas())
    except AssertionError as e:
        print("Binary Predicate Test failed")
        print("----------------------------")
        print(f"lhs: {lhs}")
        print(f"rhs: {rhs}")
        print(f"predicate: {predicate}")
        print(f"test_type: {test_type}")
        print(f"expected: {expected}")
        print(f"got: {got}")
        pytest.fail(f"Assertion failed: {e}")
