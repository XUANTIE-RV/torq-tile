#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)
project(gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_rvv LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(TORQ_TILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../..)
add_executable(${PROJECT_NAME}
    ${PROJECT_NAME}.cpp
    ${TORQ_TILE_PATH}/src/gemm/gemm_i8_u5/gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${TORQ_TILE_PATH})
target_link_libraries(${PROJECT_NAME} PRIVATE m)
target_link_options(${PROJECT_NAME} PRIVATE -static)
