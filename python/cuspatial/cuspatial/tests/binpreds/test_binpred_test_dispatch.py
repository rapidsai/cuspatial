# Copyright (c) 2023, NVIDIA CORPORATION.

from functools import wraps

import pandas as pd
import pytest
from binpred_test_dispatch import predicate, simple_test  # noqa: F401

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


# In the below file, all failing tests are recorded with visualizations.
out_file = open("test_binpred_test_dispatch.log", "w")


# @xfail_on_exception  # TODO: Remove when all tests are passing
def test_simple_features(
    predicate,  # noqa: F811
    simple_test,  # noqa: F811
    predicate_passes,
    predicate_fails,
    feature_passes,
    feature_fails,
    request,
):
    """Parameterized test fixture that runs a binary predicate test
    for each combination of geometry types and binary predicates.

    Uses four fixtures from `conftest.py` to store the number of times
    each binary predicate has passed and failed, and the number of times
    each combination of geometry types has passed and failed. These
    results are saved to CSV files after each test.

    Uses the @xfail_on_exception decorator to mark a test as xfailed
    if an exception is thrown. This is a temporary measure to allow
    the test suite to run to completion while we work on fixing the
    failing tests.

    Parameters
    ----------
    predicate : str
        The name of the binary predicate to test.
    simple_test : tuple
        A tuple containing the name of the test, a docstring that
        describes the test, and the left and right geometry objects.
    predicate_passes : dict
        A dictionary fixture containing the number of times each binary
        predicate has passed.
    predicate_fails : dict
        A dictionary fixture containing the number of times each binary
        predicate has failed.
    feature_passes : dict
        A dictionary fixture containing the number of times each combination
        of geometry types has passed.
    feature_fails : dict
        A dictionary fixture containing the number of times each combination
        of geometry types has failed.
    request : pytest.FixtureRequest
        The pytest request object. Used to print the test name in
        diagnostic output.
    """
    try:
        (lhs, rhs) = simple_test[2], simple_test[3]
        gpdlhs = lhs.to_geopandas()
        gpdrhs = rhs.to_geopandas()
        pred_fn = getattr(lhs, predicate)
        got = pred_fn(rhs)
        gpd_pred_fn = getattr(gpdlhs, predicate)
        expected = gpd_pred_fn(gpdrhs)
        assert (got.values_host == expected.values).all()

        # The test is complete, the rest is just logging.
        try:
            # The test passed, store the results.
            predicate_passes[predicate] = (
                1
                if predicate not in predicate_passes
                else predicate_passes[predicate] + 1
            )
            feature_passes[(lhs.column_type, rhs.column_type)] = (
                1
                if (lhs.column_type, rhs.column_type) not in feature_passes
                else feature_passes[(lhs.column_type, rhs.column_type)] + 1
            )
            passes_df = pd.DataFrame(
                {
                    "predicate": list(predicate_passes.keys()),
                    "predicate_passes": list(predicate_passes.values()),
                }
            )
            passes_df.to_csv("predicate_passes.csv", index=False)
            passes_df = pd.DataFrame(
                {
                    "feature": list(feature_passes.keys()),
                    "feature_passes": list(feature_passes.values()),
                }
            )
            passes_df.to_csv("feature_passes.csv", index=False)
        except Exception as e:
            raise ValueError(e)
    except Exception as e:
        # The test failed, store the results.
        out_file.write(
            f"""{predicate},
------------
{simple_test[0]}\n{simple_test[1]}\nfailed
test: {request.node.name}\n\n"""
        )
        predicate_fails[predicate] = (
            1
            if predicate not in predicate_fails
            else predicate_fails[predicate] + 1
        )
        feature_fails[(lhs.column_type, rhs.column_type)] = (
            1
            if (lhs.column_type, rhs.column_type) not in feature_fails
            else feature_fails[(lhs.column_type, rhs.column_type)] + 1
        )
        predicate_fails_df = pd.DataFrame(
            {
                "predicate": list(predicate_fails.keys()),
                "predicate_fails": list(predicate_fails.values()),
            }
        )
        predicate_fails_df.to_csv("predicate_fails.csv", index=False)
        feature_fails_df = pd.DataFrame(
            {
                "feature": list(feature_fails.keys()),
                "feature_fails": list(feature_fails.values()),
            }
        )
        feature_fails_df.to_csv("feature_fails.csv", index=False)
        raise e
