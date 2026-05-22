#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

# Target name
set(TARGET_NAME convcf_mx1bias_clamp_f16_f16p_f16p_rvv)

project(${TARGET_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TORQ_TILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../)

# Include directories
include_directories(
    ${TORQ_TILE_PATH}/
    ${TORQ_TILE_PATH}/src/conv/conv_f16_f16/convcf_mx1bias_clamp_f16_f16p_f16p/
)

# Source files
set(TORQ_TILE_SOURCES
    ${TORQ_TILE_PATH}/src/gemm/gemm_f16_f16/gemm_mx1bias_clamp_f16_f16p_f16p/tqt_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv.cpp
    ${TORQ_TILE_PATH}/src/conv/conv_f16_f16/convcf_mx1bias_clamp_f16_f16p_f16p/tqt_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv.cpp
)

# Files required to build the executable
add_executable(
    ${TARGET_NAME}
    ${TARGET_NAME}.cpp
    ${TORQ_TILE_SOURCES}
)

# Compile options
target_compile_options(${TARGET_NAME} PRIVATE)

# Link options
target_link_options(${TARGET_NAME} PRIVATE -static)

# Link math library
target_link_libraries(${TARGET_NAME} m)

# Compile definitions
target_compile_definitions(${TARGET_NAME}
    PRIVATE $<$<CONFIG:Debug>:TQT_DEBUG>
)
