#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
cmake_minimum_required(VERSION 3.16)
set(TARGET_NAME gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv)
project(${TARGET_NAME})
set(KERNEL_DIR src/gemm/gemm_f32_f16/gemm_1xnbias_clamp_f32_f32_f16t)
set(KERNEL_SOURCE ${KERNEL_DIR}/tqt_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv.cpp)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TORQ_TILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../..)
add_executable(${TARGET_NAME} ${TARGET_NAME}.cpp ${TORQ_TILE_PATH}/${KERNEL_SOURCE})
target_include_directories(${TARGET_NAME} PRIVATE ${TORQ_TILE_PATH} ${TORQ_TILE_PATH}/${KERNEL_DIR})
target_link_options(${TARGET_NAME} PRIVATE -static)
target_link_libraries(${TARGET_NAME} m)
target_compile_definitions(${TARGET_NAME} PRIVATE $<$<CONFIG:Debug>:TQT_DEBUG>)
