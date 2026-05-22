#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

# Target name
set(TARGET_NAME convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_rvv)

project(${TARGET_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TORQ_TILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../)

# Include directories
include_directories(
    ${TORQ_TILE_PATH}/
    ${TORQ_TILE_PATH}/src/conv/conv_i8_i8/convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt/
)

# Source files
set(TORQ_TILE_SOURCES
    ${TORQ_TILE_PATH}/src/gemm/gemm_i8_i8/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv.cpp
    ${TORQ_TILE_PATH}/src/conv/conv_i8_i8/convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt/tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv.cpp
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
