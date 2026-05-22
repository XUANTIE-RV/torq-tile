#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

set(TARGET_NAME dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv)

project(${TARGET_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TORQ_TILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../)

include_directories(
    ${TORQ_TILE_PATH}/
    ${TORQ_TILE_PATH}/src/dwconv/dwconv_f32_f32/dwconvcf_mx1bias_clamp_f32p_f32p_f32p/
)

set(TORQ_TILE_SOURCES
    ${TORQ_TILE_PATH}/src/dwconv/dwconv_f32_f32/dwconvcf_mx1bias_clamp_f32p_f32p_f32p/tqt_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv.cpp
)

add_executable(${TARGET_NAME} ${TARGET_NAME}.cpp ${TORQ_TILE_SOURCES})
target_link_options(${TARGET_NAME} PRIVATE -static)
target_link_libraries(${TARGET_NAME} m)
