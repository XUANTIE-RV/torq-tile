#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

include(CMakePackageConfigHelpers)

function(torq_tile_install)
    # Interfaces are shared by kernel base, including variants with strategy suffixes.
    set(pending_headers "")
    foreach(source IN LISTS TORQ_TILE_ALL_SOURCES)
        get_filename_component(base "${source}" NAME_WE)
        get_filename_component(directory "${source}" DIRECTORY)
        get_filename_component(kernel_base "${directory}" NAME)
        list(APPEND pending_headers
            "${directory}/${base}.h"
            "${directory}/tqt_${kernel_base}_interface.h")
    endforeach()

    # Follow all literal local includes, including conditional declarations.
    # Install only this closure; stale files in the generated tree are never installed.
    set(public_headers "")
    while(pending_headers)
        list(POP_FRONT pending_headers header)
        if(header IN_LIST public_headers)
            continue()
        endif()
        if(NOT EXISTS "${PROJECT_SOURCE_DIR}/${header}")
            message(FATAL_ERROR "Missing public header: ${header}")
        endif()
        list(APPEND public_headers "${header}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${PROJECT_SOURCE_DIR}/${header}")
        file(READ "${PROJECT_SOURCE_DIR}/${header}" content)
        string(REGEX MATCHALL "(^|\n)[ \t]*#[ \t]*include[ \t]+[^\n]*" includes "${content}")
        foreach(directive IN LISTS includes)
            if(directive MATCHES "#[ \t]*include[ \t]+\"([^\"]+)\"")
                set(include_path "${CMAKE_MATCH_1}")
                if(include_path MATCHES "^src/")
                    set(dependency "${PROJECT_SOURCE_DIR}/${include_path}")
                    string(REGEX REPLACE "^src/" "torq_tile/" installed_path "${include_path}")
                    string(REPLACE "\"${include_path}\"" "\"${installed_path}\""
                        installed_directive "${directive}")
                    string(REPLACE "${directive}" "${installed_directive}" content "${content}")
                else()
                    get_filename_component(directory "${header}" DIRECTORY)
                    set(dependency "${PROJECT_SOURCE_DIR}/${directory}/${include_path}")
                endif()
                get_filename_component(dependency "${dependency}" ABSOLUTE)
                file(RELATIVE_PATH dependency "${PROJECT_SOURCE_DIR}" "${dependency}")
                if(NOT dependency MATCHES "^src/" OR
                   NOT EXISTS "${PROJECT_SOURCE_DIR}/${dependency}")
                    message(FATAL_ERROR "Unresolved public include in ${header}: ${include_path}")
                endif()
                list(APPEND pending_headers "${dependency}")
            elseif(NOT directive MATCHES "#[ \t]*include[ \t]+<[^>]+>")
                message(FATAL_ERROR "Unsupported public include in ${header}: ${directive}")
            endif()
        endforeach()

        string(REGEX REPLACE "^src/" "" relative_header "${header}")
        set(generated "${PROJECT_BINARY_DIR}/install/include/torq_tile/${relative_header}")
        # Avoid touching unchanged files on reconfiguration.
        set(previous_content "")
        if(EXISTS "${generated}")
            file(READ "${generated}" previous_content)
        endif()
        if(NOT EXISTS "${generated}" OR NOT content STREQUAL previous_content)
            file(WRITE "${generated}" "${content}")
        endif()
        get_filename_component(destination "${relative_header}" DIRECTORY)
        install(FILES "${generated}" DESTINATION "include/torq_tile/${destination}")
    endwhile()

    install(FILES "${PROJECT_BINARY_DIR}/include/torq_tile/tqt_compile_config.h"
        DESTINATION include/torq_tile)
    install(TARGETS torq_tile
        EXPORT torq_tile_targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
    install(EXPORT torq_tile_targets
        FILE torq_tile_targets.cmake
        NAMESPACE torq_tile::
        DESTINATION lib/cmake/torq_tile)

    configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/torq_tileConfig.cmake.in"
        "${PROJECT_BINARY_DIR}/torq_tileConfig.cmake"
        INSTALL_DESTINATION lib/cmake/torq_tile)
    write_basic_package_version_file("${PROJECT_BINARY_DIR}/torq_tileConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion)
    install(FILES
        "${PROJECT_BINARY_DIR}/torq_tileConfig.cmake"
        "${PROJECT_BINARY_DIR}/torq_tileConfigVersion.cmake"
        DESTINATION lib/cmake/torq_tile)
endfunction()
