#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

# Match target ordering: global flags, configuration flags, then private options.
# CMAKE_REQUIRED_FLAGS alone precedes configuration flags in try_compile.
function(torq_tile_check_cxx source result_name configuration options)
    cmake_push_check_state(RESET)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    set(CMAKE_REQUIRED_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}")
    string(JOIN " " private_flags ${options})
    if(configuration STREQUAL "__default__")
        set(CMAKE_TRY_COMPILE_CONFIGURATION "")
        set(CMAKE_REQUIRED_FLAGS "${private_flags}")
    else()
        set(CMAKE_TRY_COMPILE_CONFIGURATION "${configuration}")
        string(TOUPPER "${configuration}" upper_configuration)
        set(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS_${upper_configuration}} ${private_flags}")
        set(CMAKE_CXX_FLAGS_${upper_configuration} "")
    endif()
    unset(${result_name} CACHE)
    unset(${result_name})
    check_cxx_source_compiles("${source}" ${result_name})
    set(${result_name} "${${result_name}}" PARENT_SCOPE)
    cmake_pop_check_state()
endfunction()

# Resolve the core mode before ISA probes, using the same per-configuration flags.
function(torq_tile_configure_runtime)
    string(TOUPPER "${TORQ_TILE_FREESTANDING}" requested)
    if(NOT requested MATCHES "^(AUTO|ON|OFF)$")
        message(FATAL_ERROR "TORQ_TILE_FREESTANDING must be AUTO, ON or OFF")
    endif()
    set(core_options "")
    if(requested STREQUAL "ON")
        list(APPEND core_options -ffreestanding)
    endif()
    if(CMAKE_CONFIGURATION_TYPES)
        set(configurations ${CMAKE_CONFIGURATION_TYPES})
    elseif(CMAKE_BUILD_TYPE)
        set(configurations "${CMAKE_BUILD_TYPE}")
    else()
        set(configurations "__default__")
    endif()
    set(resolved "")
    foreach(configuration IN LISTS configurations)
        foreach(mode HOSTED FREESTANDING)
            if(mode STREQUAL "HOSTED")
                set(expected 1)
            else()
                set(expected 0)
            endif()
            unset(TORQ_TILE_PROBE_${mode} CACHE)
            unset(TORQ_TILE_PROBE_${mode})
            torq_tile_check_cxx("
                #if !defined(__STDC_HOSTED__) || __STDC_HOSTED__ != ${expected}
                #error Unexpected compiler execution environment
                #endif
                int probe() { return 0; }
            " TORQ_TILE_PROBE_${mode} "${configuration}" "${core_options}")
        endforeach()
        if(TORQ_TILE_PROBE_HOSTED)
            set(current 0)
        elseif(TORQ_TILE_PROBE_FREESTANDING)
            set(current 1)
        else()
            message(FATAL_ERROR "Cannot determine the core compiler environment for ${configuration}")
        endif()
        if(NOT resolved STREQUAL "" AND NOT resolved STREQUAL current)
            message(FATAL_ERROR "Mixed hosted/freestanding configurations require separate build directories")
        endif()
        set(resolved "${current}")
    endforeach()
    if(requested STREQUAL "OFF" AND resolved)
        message(FATAL_ERROR "TORQ_TILE_FREESTANDING=OFF conflicts with the compiler's freestanding mode")
    endif()
    if(requested STREQUAL "ON" AND NOT resolved)
        message(FATAL_ERROR "TORQ_TILE_FREESTANDING=ON did not enable compiler freestanding mode")
    endif()
    if(resolved)
        list(APPEND core_options -fno-exceptions -fno-rtti)
    endif()
    foreach(configuration IN LISTS configurations)
        unset(TORQ_TILE_PROBE_RUNTIME CACHE)
        unset(TORQ_TILE_PROBE_RUNTIME)
        torq_tile_check_cxx("
            #include \"src/common/tqt_runtime.h\"
            #if TORQ_TILE_FREESTANDING && (defined(__EXCEPTIONS) || defined(__GXX_RTTI))
            #error Freestanding core probes must disable exceptions and RTTI
            #endif
            int probe() { return 0; }
        " TORQ_TILE_PROBE_RUNTIME "${configuration}" "${core_options}")
        if(NOT TORQ_TILE_PROBE_RUNTIME)
            message(FATAL_ERROR "Invalid torq-tile runtime configuration; check compiler flags and runtime macro diagnostics")
        endif()
    endforeach()
    set(TORQ_TILE_BUILT_FREESTANDING "${resolved}" PARENT_SCOPE)
    set(TORQ_TILE_CORE_COMPILE_OPTIONS "${core_options}" PARENT_SCOPE)
    message(STATUS "Core freestanding=${resolved} (${requested})")
endfunction()

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
        unset(${probe} CACHE)
        unset(${probe})
        torq_tile_check_cxx("${source}" ${probe} "${configuration}" "${TORQ_TILE_CORE_COMPILE_OPTIONS}")
        if(NOT ${probe})
            set(can_compile OFF)
        endif()
    endforeach()
    set(${probe} "${can_compile}" CACHE INTERNAL "C++ probe for ${feature} in all build configurations" FORCE)

    set(${result_name} "${can_compile}" PARENT_SCOPE)
    list(APPEND TORQ_TILE_ISA_GROUPS "${feature}")
    set(TORQ_TILE_ISA_GROUPS "${TORQ_TILE_ISA_GROUPS}" PARENT_SCOPE)
    message(STATUS "${feature}: C++ support=${can_compile}")
endfunction()
