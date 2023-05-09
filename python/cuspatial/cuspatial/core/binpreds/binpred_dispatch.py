# Copyright (c) 2023, NVIDIA CORPORATION.

"""`binpred_dispatch.py` contains a collection of dictionaries that
are used to dispatch binary predicate functions to the correct
implementation.

The dictionaries are collected here to make using the dispatch
functionality easier.
"""

from cuspatial.core.binpreds.feature_contains import (  # NOQA F401
    DispatchDict as CONTAINS_DISPATCH,
)
from cuspatial.core.binpreds.feature_contains_properly import (  # NOQA F401
    DispatchDict as CONTAINS_PROPERLY_DISPATCH,
)
from cuspatial.core.binpreds.feature_covers import (  # NOQA F401
    DispatchDict as COVERS_DISPATCH,
)
from cuspatial.core.binpreds.feature_crosses import (  # NOQA F401
    DispatchDict as CROSSES_DISPATCH,
)
from cuspatial.core.binpreds.feature_disjoint import (  # NOQA F401
    DispatchDict as DISJOINT_DISPATCH,
)
from cuspatial.core.binpreds.feature_equals import (  # NOQA F401
    DispatchDict as EQUALS_DISPATCH,
)
from cuspatial.core.binpreds.feature_intersects import (  # NOQA F401
    DispatchDict as INTERSECTS_DISPATCH,
)
from cuspatial.core.binpreds.feature_overlaps import (  # NOQA F401
    DispatchDict as OVERLAPS_DISPATCH,
)
from cuspatial.core.binpreds.feature_touches import (  # NOQA F401
    DispatchDict as TOUCHES_DISPATCH,
)
from cuspatial.core.binpreds.feature_within import (  # NOQA F401
    DispatchDict as WITHIN_DISPATCH,
)
