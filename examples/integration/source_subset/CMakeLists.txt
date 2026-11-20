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
set(TORQ_TILE_SOURCE_DIR "" CACHE PATH "TORQ-Tile source checkout")
if(NOT EXISTS "${TORQ_TILE_SOURCE_DIR}/src/common/tqt_common.h")
    message(FATAL_ERROR "Set TORQ_TILE_SOURCE_DIR to the TORQ-Tile checkout")
endif()
# Only this translation unit and its transitive headers are needed.
if(TQT_BACKEND STREQUAL "RVV")
    target_sources(consumer PRIVATE "${TORQ_TILE_SOURCE_DIR}/src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv.cpp")
endif()
if(TQT_BACKEND STREQUAL "ZTT")
    target_sources(consumer PRIVATE "${TORQ_TILE_SOURCE_DIR}/src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt.cpp")
endif()
target_include_directories(consumer PRIVATE "${TORQ_TILE_SOURCE_DIR}")
