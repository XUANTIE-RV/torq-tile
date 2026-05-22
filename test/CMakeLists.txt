#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

# ============================================================================
# Test Common Library
# ============================================================================
add_library(torq_tile_test_common STATIC
    common/data_generator.cpp
    common/reference_gemm.cpp
    common/reference_conv.cpp
    common/verify.cpp
)

target_include_directories(torq_tile_test_common PUBLIC
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(torq_tile_test_common PUBLIC
    torq_tile
    gtest
)

if(NOT TORQ_TILE_BUILD_SHARED)
    target_link_options(torq_tile_test_common PUBLIC
        -Wl,--whole-archive
        $<TARGET_FILE:torq_tile>
        -Wl,--no-whole-archive
    )
endif()

# ============================================================================
# Macro: Register a test executable
# ============================================================================
macro(torq_tile_add_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE
        torq_tile_test_common
        gtest
        gtest_main
    )
    if(TORQ_TILE_TEST_RUNNER)
        separate_arguments(RUNNER_ARGS UNIX_COMMAND "${TORQ_TILE_TEST_RUNNER}")
        add_test(NAME ${TEST_NAME} COMMAND ${RUNNER_ARGS} ${TEST_NAME})
    else()
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endif()
endmacro()

# ============================================================================
# RVV Tests (auto-detected via TORQ_TILE_ENABLE_RVV)
# ============================================================================
if(TORQ_TILE_ENABLE_RVV)
    # --- Gemm F32 RVV variants ---
    torq_tile_add_test(test_gemm_f32_f32_rvv
        gemm/test_gemm_f32_f32_rvv.cpp)
    # --- Gemm INT8 RVV variants ---
    torq_tile_add_test(test_gemm_i8_i8_rvv
        gemm/test_gemm_i8_i8_rvv.cpp)
    # --- Gemm W4A8 (F32/F16/BF16_QAI8DX_QSI4CX) RVV variants ---
    # F32 path is always compiled; F16/BF16 paths are gated inside the TU
    # by __riscv_zfh/__riscv_zvfh and __riscv_zvfbfwma feature macros.
    torq_tile_add_test(test_gemm_i8_i4_rvv
        gemm/test_gemm_i8_i4_rvv.cpp)
    # --- Conv F32 RVV variants ---
    torq_tile_add_test(test_conv_f32_f32_rvv
        conv/test_conv_f32_f32_rvv.cpp)
    torq_tile_add_test(test_igemm_f32_f32_rvv
        conv/test_igemm_f32_f32_rvv.cpp)
    # --- Conv INT8 RVV variants ---
    torq_tile_add_test(test_conv_i8_i8_rvv
        conv/test_conv_i8_i8_rvv.cpp)
    torq_tile_add_test(test_igemm_i8_i8_rvv
        conv/test_igemm_i8_i8_rvv.cpp)
endif()

if(TORQ_TILE_ENABLE_RVV_FP16)
    # --- Gemm F16 RVV variants ---
    torq_tile_add_test(test_gemm_f16_f16_rvv
        gemm/test_gemm_f16_f16_rvv.cpp)
    # --- Gemm W4A16 (F16_F16_I4 + F32_F16_I4) RVV variants ---
    torq_tile_add_test(test_gemm_f16_i4_rvv
        gemm/test_gemm_f16_i4_rvv.cpp)
    # --- Conv F16 RVV variants ---
    torq_tile_add_test(test_conv_f16_f16_rvv
        conv/test_conv_f16_f16_rvv.cpp)
    torq_tile_add_test(test_igemm_f16_f16_rvv
        conv/test_igemm_f16_f16_rvv.cpp)
endif()

if(TORQ_TILE_ENABLE_RVV_BF16)
    # --- Gemm BF16 RVV variants ---
    torq_tile_add_test(test_gemm_bf16_bf16_rvv
        gemm/test_gemm_bf16_bf16_rvv.cpp)
endif()
