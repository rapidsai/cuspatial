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


"""Parameterized test fixture that runs a binary predicate test
for each combination of geometry types and binary predicates."""

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
    try:
        (lhs, rhs) = simple_test[2], simple_test[3]
        gpdlhs = lhs.to_geopandas()
        gpdrhs = rhs.to_geopandas()
        pred_fn = getattr(lhs, predicate)
        got = pred_fn(rhs)
        if predicate == "contains_properly":
            predicate = "contains"
        gpd_pred_fn = getattr(gpdlhs, predicate)
        expected = gpd_pred_fn(gpdrhs)
        pd.testing.assert_series_equal(expected, got.to_pandas())
        try:
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
            print(passes_df)
        except Exception as e:
            raise ValueError(e)
    except Exception as e:
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
        # TODO: Uncomment when all tests are passing
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
        raise e  # TODO: Remove when all tests are passing.
        # pytest.fail(f"Assertion failed: {e}")
