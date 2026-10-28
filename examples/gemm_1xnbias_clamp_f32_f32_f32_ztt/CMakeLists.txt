#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)
project(gemm_1xnbias_clamp_f32_f32_f32_ztt LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TORQ_TILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../..)
add_executable(gemm_1xnbias_clamp_f32_f32_f32_ztt
    gemm_1xnbias_clamp_f32_f32_f32_ztt.cpp
    ${TORQ_TILE_PATH}/src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt.cpp
)
target_include_directories(gemm_1xnbias_clamp_f32_f32_f32_ztt PRIVATE ${TORQ_TILE_PATH})
target_compile_options(gemm_1xnbias_clamp_f32_f32_f32_ztt PRIVATE -fno-unwind-tables -fno-asynchronous-unwind-tables)
target_link_options(gemm_1xnbias_clamp_f32_f32_f32_ztt PRIVATE -static)
target_link_libraries(gemm_1xnbias_clamp_f32_f32_f32_ztt PRIVATE m)
