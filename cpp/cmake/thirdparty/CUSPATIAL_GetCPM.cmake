set(CPM_DOWNLOAD_VERSION 3b404296b539e596f39421c4e92bc803b299d964) # v0.27.5

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "CUSPATIAL: Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(
    DOWNLOAD
    https://raw.githubusercontent.com/TheLartians/CPM.cmake/${CPM_DOWNLOAD_VERSION}/cmake/CPM.cmake
    ${CPM_DOWNLOAD_LOCATION})
endif()

include(${CPM_DOWNLOAD_LOCATION})

function(cuspatial_save_if_enabled var)
    if(CUSPATIAL_${var})
        unset(${var} PARENT_SCOPE)
        unset(${var} CACHE)
    endif()
endfunction()

function(cuspatial_restore_if_enabled var)
    if(CUSPATIAL_${var})
        set(${var} ON CACHE INTERNAL "" FORCE)
    endif()
endfunction()

function(fix_cmake_global_defaults target)
    if(TARGET ${target})
        get_target_property(_is_imported ${target} IMPORTED)
        get_target_property(_already_global ${target} IMPORTED_GLOBAL)
        if(_is_imported AND NOT _already_global)
            set_target_properties(${target} PROPERTIES IMPORTED_GLOBAL TRUE)
        endif()
    endif()
endfunction()
