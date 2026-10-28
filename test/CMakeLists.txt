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
    common/reference_sgemm.cpp
    common/reference_conv.cpp
    common/reference_dwconv.cpp
    common/reference_elementwise.cpp
    common/reference_reduction.cpp
    common/reference_activation.cpp
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
# RVV Tests (auto-detected via TORQ_TILE_HAS_RVV)
# ============================================================================
if(TORQ_TILE_HAS_RVV)
    # --- Gemm F32 RVV variants ---
    torq_tile_add_test(test_gemm_f32_f32_rvv
        gemm/test_gemm_f32_f32_rvv.cpp)
    # --- Gemm INT8 RVV variants ---
    torq_tile_add_test(test_gemm_i8_i8_rvv
        gemm/test_gemm_i8_i8_rvv.cpp)
    # --- Gemm UINT8 RVV variants ---
    torq_tile_add_test(test_gemm_u8_u8_rvv
        gemm/test_gemm_u8_u8_rvv.cpp)
    # --- Gemm W4A8 (per-channel qai8dx/qsi4cx + block-quantized
    #     qai8dx|qsi8d2 × qsi4c2|qai4c2 + qsi8d32p × qsi4c32p) RVV variants ---
    # F32 paths are always compiled; optional output types follow the selected
    # library groups, even when the compiler supports additional ISA features.
    torq_tile_add_test(test_gemm_i8_i4_rvv
        gemm/test_gemm_i8_i4_rvv.cpp)
    # --- GEMM block-quantized activation RVV variants ---
    torq_tile_add_test(test_gemm_i8_u4_rvv
        gemm/test_gemm_i8_u4_rvv.cpp)
    torq_tile_add_test(test_gemm_i8_u2_rvv
        gemm/test_gemm_i8_u2_rvv.cpp)
    torq_tile_add_test(test_gemm_i8_i3_rvv
        gemm/test_gemm_i8_i3_rvv.cpp)
    torq_tile_add_test(test_gemm_i8_u5_rvv
        gemm/test_gemm_i8_u5_rvv.cpp)
    torq_tile_add_test(test_gemm_i8_u8_rvv
        gemm/test_gemm_i8_u8_rvv.cpp)
    torq_tile_add_test(test_gemm_i8_i6_rvv
        gemm/test_gemm_i8_i6_rvv.cpp)
    torq_tile_add_test(test_gemm_i8_b1_rvv
        gemm/test_gemm_i8_b1_rvv.cpp)
    torq_tile_add_test(test_gemm_i8_i5_rvv
        gemm/test_gemm_i8_i5_rvv.cpp)
    # --- Gemm i8_i2 RVV variants ---
    torq_tile_add_test(test_gemm_i8_i2_rvv
        gemm/test_gemm_i8_i2_rvv.cpp)
    # --- Conv F32 RVV variants ---
    torq_tile_add_test(test_conv_f32_f32_rvv
        conv/test_conv_f32_f32_rvv.cpp)
    torq_tile_add_test(test_igemm_f32_f32_rvv
        conv/test_igemm_f32_f32_rvv.cpp)
    # --- DWConv F32 RVV variants ---
    torq_tile_add_test(test_dwconv_f32_f32_rvv
        dwconv/test_dwconv_f32_f32_rvv.cpp)
    # --- Conv INT8 RVV variants ---
    torq_tile_add_test(test_conv_i8_i8_rvv
        conv/test_conv_i8_i8_rvv.cpp)
    torq_tile_add_test(test_igemm_i8_i8_rvv
        conv/test_igemm_i8_i8_rvv.cpp)
    # --- Elementwise F32 RVV variants ---
    torq_tile_add_test(test_add_f32_f32_rvv
        elementwise/test_add_f32_f32_rvv.cpp)
    torq_tile_add_test(test_sub_f32_f32_rvv
        elementwise/test_sub_f32_f32_rvv.cpp)
    torq_tile_add_test(test_mul_f32_f32_rvv
        elementwise/test_mul_f32_f32_rvv.cpp)
    torq_tile_add_test(test_div_f32_f32_rvv
        elementwise/test_div_f32_f32_rvv.cpp)
    # --- Reduction F32 RVV variants ---
    torq_tile_add_test(test_sum_f32_f32_rvv
        reduction/test_sum_f32_f32_rvv.cpp)
    torq_tile_add_test(test_mean_f32_f32_rvv
        reduction/test_mean_f32_f32_rvv.cpp)

    # --- Activation F32 RVV variants ---
    torq_tile_add_test(test_relu_f32_f32_rvv
        activation/test_relu_f32_f32_rvv.cpp)
    torq_tile_add_test(test_gelu_f32_f32_rvv
        activation/test_gelu_f32_f32_rvv.cpp)
    torq_tile_add_test(test_silu_f32_f32_rvv
        activation/test_silu_f32_f32_rvv.cpp)
    torq_tile_add_test(test_reglu_f32_f32_f32_rvv
        activation/test_reglu_f32_f32_f32_rvv.cpp)
    torq_tile_add_test(test_geglu_f32_f32_f32_rvv
        activation/test_geglu_f32_f32_f32_rvv.cpp)
    torq_tile_add_test(test_swiglu_f32_f32_f32_rvv
        activation/test_swiglu_f32_f32_f32_rvv.cpp)
endif()

if(TORQ_TILE_HAS_RVV_F64)
    # --- DGEMM F64 RVV variants (BLAS-style) ---
    torq_tile_add_test(test_gemm_f64_f64_rvv
        gemm/test_gemm_f64_f64_rvv.cpp)
endif()

if(TORQ_TILE_HAS_RVV_F16)
    torq_tile_add_test(test_gemm_f32_f16_rvv
        gemm/test_gemm_f32_f16_rvv.cpp)
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
    # --- DWConv F16 RVV variants ---
    torq_tile_add_test(test_dwconv_f16_f16_rvv
        dwconv/test_dwconv_f16_f16_rvv.cpp)
    # --- Elementwise F16 RVV variants ---
    torq_tile_add_test(test_add_f16_f16_rvv
        elementwise/test_add_f16_f16_rvv.cpp)
    torq_tile_add_test(test_sub_f16_f16_rvv
        elementwise/test_sub_f16_f16_rvv.cpp)
    torq_tile_add_test(test_mul_f16_f16_rvv
        elementwise/test_mul_f16_f16_rvv.cpp)
    torq_tile_add_test(test_div_f16_f16_rvv
        elementwise/test_div_f16_f16_rvv.cpp)
    # --- Reduction F16 RVV variants ---
    torq_tile_add_test(test_sum_f16_f16_rvv
        reduction/test_sum_f16_f16_rvv.cpp)
    torq_tile_add_test(test_mean_f16_f16_rvv
        reduction/test_mean_f16_f16_rvv.cpp)

    # --- Activation F16 RVV variants ---
    torq_tile_add_test(test_relu_f16_f16_rvv
        activation/test_relu_f16_f16_rvv.cpp)
    torq_tile_add_test(test_gelu_f16_f16_rvv
        activation/test_gelu_f16_f16_rvv.cpp)
    torq_tile_add_test(test_silu_f16_f16_rvv
        activation/test_silu_f16_f16_rvv.cpp)
    torq_tile_add_test(test_reglu_f16_f16_f16_rvv
        activation/test_reglu_f16_f16_f16_rvv.cpp)
    torq_tile_add_test(test_geglu_f16_f16_f16_rvv
        activation/test_geglu_f16_f16_f16_rvv.cpp)
    torq_tile_add_test(test_swiglu_f16_f16_f16_rvv
        activation/test_swiglu_f16_f16_f16_rvv.cpp)
endif()

if(TORQ_TILE_HAS_RVV_BF16)
    torq_tile_add_test(test_gemm_f32_bf16_rvv
        gemm/test_gemm_f32_bf16_rvv.cpp)
    # --- Gemm BF16 RVV variants ---
    torq_tile_add_test(test_gemm_bf16_bf16_rvv
        gemm/test_gemm_bf16_bf16_rvv.cpp)

    # --- Conv BF16 RVV variants ---
    torq_tile_add_test(test_conv_bf16_bf16_rvv
        conv/test_conv_bf16_bf16_rvv.cpp)
endif()

# ============================================================================
# ============================================================================

# ============================================================================
# ============================================================================
