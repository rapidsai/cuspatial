# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import ctypes
import os


def load_library():
    try:
        # libcudf must be loaded before libcuspatial because libcuspatial
        # references its symbols
        import libcudf
        libcudf.load_library()
    except ModuleNotFoundError:
        # 'libcuspatial' has a runtime dependency on 'libcudf'. However,
        # that dependency might be satisfied by the 'libcudf' conda package
        # (which does not have any Python modules), instead of the
        # 'libcudf' wheel.
        #
        # In that situation, assume that 'libcudf.so' is in a place where
        # the loader can find it.
        pass

    # Dynamically load libcuspatial.so. Prefer a system library if one is
    # present to avoid clobbering symbols that other packages might expect,
    # but if no other library is present use the one in the wheel.
    libcuspatial_lib = None
    try:
        libcuspatial_lib = ctypes.CDLL("libcuspatial.so", ctypes.RTLD_GLOBAL)
    except OSError:
        # If neither of these directories contain the library, we assume we are
        # in an environment where the C++ library is already installed
        # somewhere else and the CMake build of the libcuspatial Python package
        # was a no-op.
        #
        # Note that this approach won't work for real editable installs of the
        # libcuspatial package. scikit-build-core has limited support for
        # importlib.resources so there isn't a clean way to support that case
        # yet.
        for lib_dir in ("lib", "lib64"):
            if os.path.isfile(
                lib := os.path.join(
                    os.path.dirname(__file__), lib_dir, "libcuspatial.so"
                )
            ):
                libcuspatial_lib = ctypes.CDLL(lib, ctypes.RTLD_GLOBAL)
                break

    # The caller almost never needs to do anything with this library, but no
    # harm in offering the option since this object at least provides a handle
    # to inspect where the library was loaded from.
    return libcuspatial_lib
