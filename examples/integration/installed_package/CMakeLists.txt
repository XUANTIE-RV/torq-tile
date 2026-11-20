#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)
project(tqt_example_consumer LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TQT_BACKEND "RVV" CACHE STRING "Selected backend: RVV, ZTT")
set_property(CACHE TQT_BACKEND PROPERTY STRINGS RVV ZTT)
add_executable(consumer main.cpp)
target_compile_definitions(consumer PRIVATE TQT_INTEGRATION_${TQT_BACKEND}=1)
if(TQT_BACKEND STREQUAL "ZTT")
    target_compile_options(consumer PRIVATE -fno-unwind-tables -fno-asynchronous-unwind-tables)
endif()
find_package(torq_tile CONFIG REQUIRED)
target_link_libraries(consumer PRIVATE torq_tile::torq_tile)
target_compile_definitions(consumer PRIVATE TQT_INSTALLED_CONSUMER=1)
