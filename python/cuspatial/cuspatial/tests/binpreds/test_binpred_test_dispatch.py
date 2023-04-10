# Copyright (c) 2023, NVIDIA CORPORATION.

from functools import wraps

import pandas as pd
import pytest
from binpred_test_dispatch import (  # noqa: F401
    feature_test_dispatch,
    geotype_tuple,
    predicate,
)

"""Decorator function that xfails a test if an exception is throw
by the test function. Will be removed when all tests are passing."""


def xfail_on_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            pytest.xfail(f"Xfailling due to an exception: {e}")

    return wrapper


"""Parameterized test fixture that runs a binary predicate test
for each combination of geometry types and binary predicates."""


# @xfail_on_exception  # TODO: Remove when all tests are passing
def test_fixtures(predicate, geotype_tuple):  # noqa: F811
    (lhs, rhs) = feature_test_dispatch(geotype_tuple)
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
        print(f"expected: {expected}")
        print(f"got: {got}")
        raise AssertionError(e)  # TODO: Remove when all tests are passing.
        # TODO: Uncomment when all tests are passing
        # pytest.fail(f"Assertion failed: {e}")
