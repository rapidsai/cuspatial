# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# This function finds osgeo/proj and sets any additional necessary environment variables.
function(find_and_configure_proj VERSION)
  include(${rapids-cmake-dir}/cpm/find.cmake)

  # Find or install Proj
  rapids_cpm_find(
    PROJ ${VERSION}
    GLOBAL_TARGETS PROJ::proj
    BUILD_EXPORT_SET cuproj-exports
    INSTALL_EXPORT_SET cuproj-exports
    CPM_ARGS
    GIT_REPOSITORY https://github.com/osgeo/proj.git
    GIT_TAG ${VERSION}
    GIT_SHALLOW TRUE
  )

  if(PROJ_ADDED)
    install(TARGETS proj EXPORT proj-exports)

    # write build export rules
    rapids_export(
      BUILD PROJ
      VERSION ${VERSION}
      EXPORT_SET proj-exports
      GLOBAL_TARGETS proj
      NAMESPACE PROJ::)

    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    # When using cuPROJ from the build dir, ensure PROJ is also found in cuPROJ's build dir. This
    # line adds `set(PROJ_ROOT "${CMAKE_CURRENT_LIST_DIR}")` to build/cuproj-dependencies.cmake
    rapids_export_find_package_root(
      BUILD PROJ [=[${CMAKE_CURRENT_LIST_DIR}]=] EXPORT_SET cuproj-exports
    )
  endif()

endfunction()

find_and_configure_proj(9.2.0)
