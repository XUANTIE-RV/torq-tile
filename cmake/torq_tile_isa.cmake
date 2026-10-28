#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

# Retired source switches are discarded even when their groups have been renamed.
get_cmake_property(torq_tile_cache_variables CACHE_VARIABLES)
foreach(variable IN LISTS torq_tile_cache_variables)
    if(variable MATCHES "^TORQ_TILE_ENABLE_")
        message(STATUS "Discarding legacy ${variable} cache entry; ISA selection follows C++ flags")
        unset(${variable} CACHE)
    endif()
endforeach()
unset(torq_tile_cache_variables)

# Results are internal probes, never user-controlled source-group switches.
function(torq_tile_check_isa source result_name)
    string(REPLACE "TORQ_TILE_CAN_COMPILE_" "" feature "${result_name}")
    # Probe the actual C++ flags without requiring a target executable to link.
    # Re-run on every configure: flags, toolchain files and probe bodies can all
    # change independently of a previously cached result.
    cmake_push_check_state(RESET)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    if(CMAKE_CONFIGURATION_TYPES)
        set(configurations ${CMAKE_CONFIGURATION_TYPES})
    elseif(CMAKE_BUILD_TYPE)
        set(configurations "${CMAKE_BUILD_TYPE}")
    else()
        set(configurations "__default__")
    endif()
    set(can_compile ON)
    set(probe "${result_name}")
    foreach(configuration IN LISTS configurations)
        if(configuration STREQUAL "__default__")
            set(CMAKE_TRY_COMPILE_CONFIGURATION "")
        else()
            set(CMAKE_TRY_COMPILE_CONFIGURATION "${configuration}")
        endif()
        unset(${probe} CACHE)
        unset(${probe})
        check_cxx_source_compiles("${source}" ${probe})
        if(NOT ${probe})
            set(can_compile OFF)
        endif()
    endforeach()
    cmake_pop_check_state()
    set(${probe} "${can_compile}" CACHE INTERNAL "C++ probe for ${feature} in all build configurations" FORCE)

    set(${result_name} "${can_compile}" PARENT_SCOPE)
    list(APPEND TORQ_TILE_ISA_GROUPS "${feature}")
    set(TORQ_TILE_ISA_GROUPS "${TORQ_TILE_ISA_GROUPS}" PARENT_SCOPE)
    message(STATUS "${feature}: C++ support=${can_compile}")
endfunction()
