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

# Use CPM to find or clone thrust
function(find_and_configure_thrust)
        include(${rapids-cmake-dir}/cpm/thrust.cmake)
        include(${rapids-cmake-dir}/cpm/package_override.cmake)

        set(cuspatial_patch_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches")
        rapids_cpm_package_override("${cuspatial_patch_dir}/thrust_override.json")

        rapids_cpm_thrust( NAMESPACE cuspatial
                           BUILD_EXPORT_SET cuspatial-exports
                           INSTALL_EXPORT_SET cuspatial-exports)
endfunction()

find_and_configure_thrust()
