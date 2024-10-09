#=============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#=============================================================================

function(find_and_configure_ranger)

    if(TARGET ranger::ranger)
        return()
    endif()

    set(global_targets ranger::ranger)
    set(find_package_args "")

    rapids_cpm_find(
      ranger 00.01.00
      GLOBAL_TARGETS "${global_targets}"
      CPM_ARGS
      GIT_REPOSITORY https://github.com/harrism/ranger.git
      GIT_TAG main
      GIT_SHALLOW TRUE
      OPTIONS "BUILD_TESTS OFF"
      FIND_PACKAGE_ARGUMENTS "${find_package_args}"
    )
endfunction()

find_and_configure_ranger()
