#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

set(TARGET_NAME gemm_1xnbias_clamp_f32_bf16p_bf16p_rvv_pack_bt)

project(${TARGET_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TORQ_TILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../)

include_directories(
    ${TORQ_TILE_PATH}/
    ${TORQ_TILE_PATH}/src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f32_bf16p_bf16p/
)

set(TORQ_TILE_SOURCES
    ${TORQ_TILE_PATH}/src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f32_bf16p_bf16p/tqt_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv.cpp
)

add_executable(${TARGET_NAME} ${TARGET_NAME}.cpp ${TORQ_TILE_SOURCES})
target_compile_options(${TARGET_NAME} PRIVATE)
target_link_options(${TARGET_NAME} PRIVATE -static)
target_link_libraries(${TARGET_NAME} m)
target_compile_definitions(${TARGET_NAME} PRIVATE $<$<CONFIG:Debug>:TQT_DEBUG>)
