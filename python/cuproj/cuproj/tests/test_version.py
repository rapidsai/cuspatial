# Copyright (c) 2024, NVIDIA CORPORATION.

import cuproj


def test_version_constants_are_populated():
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(cuproj.__git_commit__, str)

    # __version__ should always be non-empty
    assert isinstance(cuproj.__version__, str)
    assert len(cuproj.__version__) > 0
