#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)
project(torq-tile
    VERSION 0.1.0
    LANGUAGES C CXX
)

# ============================================================================
# Project Options
# ============================================================================

# Build options
option(TORQ_TILE_BUILD_SHARED "Build shared library instead of static" OFF)
option(TORQ_TILE_BUILD_TEST "Build tests" OFF)
option(TORQ_TILE_BUILD_BENCHMARK "Build benchmarks" OFF)

# ============================================================================
# C/C++ Standards
# ============================================================================

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Compiler Flags Configuration
# ============================================================================

# Auto-detect architecture feature via try_compile
# Users pass compiler flags via CMAKE_C_FLAGS / CMAKE_CXX_FLAGS (or toolchain file)
include(CheckCSourceCompiles)

# Detect RISC-V Vector support
check_c_source_compiles("
    #if !defined(__riscv) || !defined(__riscv_v)\n #error\n #endif
    int main() { return 0; }
" TORQ_TILE_ENABLE_RVV)

# Detect RISC-V Vector FP16 support (Zfh + Zvfh)
check_c_source_compiles("
    #if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)\n #error\n #endif
    int main() { return 0; }
" TORQ_TILE_ENABLE_RVV_FP16)

# Detect RISC-V Vector BF16 support (Zvfbfwma)
check_c_source_compiles("
    #if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zvfbfwma)\n #error\n #endif
    int main() { return 0; }
" TORQ_TILE_ENABLE_RVV_BF16)

# ============================================================================
# Source Files Collection
# ============================================================================

# RVV (RISC-V Vector) kernel files
set(TORQ_TILE_RVV_SOURCES
    # F32 RVV
    src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv.cpp
    src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv.cpp
    src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv.cpp
    src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32t/tqt_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv.cpp
    src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32p/tqt_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv.cpp
    src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32t/tqt_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv.cpp
    src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32t/tqt_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv.cpp
    src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32p_f32p/tqt_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv.cpp
    src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32p_f32p/tqt_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv.cpp
    src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32p_f32p/tqt_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv.cpp
    src/gemm/gemm_f32_f32/gemm_mx1bias_clamp_f32_f32_f32/tqt_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv.cpp
    src/gemm/gemm_f32_f32/gemm_mx1bias_clamp_f32_f32p_f32p/tqt_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv.cpp
    src/conv/conv_f32_f32/convcf_mx1bias_clamp_f32_f32_f32/tqt_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv.cpp
    src/conv/conv_f32_f32/convcf_mx1bias_clamp_f32_f32p_f32p/tqt_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv.cpp
    src/conv/conv_f32_f32/convcl_1xnbias_clamp_f32_f32_f32t/tqt_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv.cpp
    src/conv/conv_f32_f32/convcl_1xnbias_clamp_f32_f32p_f32p/tqt_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv.cpp
    src/conv/igemm_f32_f32/igemmcf_mx1bias_clamp_f32_f32_f32/tqt_igemmcf_mx1bias_clamp_f32_f32_f32_3vlx8_rvv.cpp
    src/conv/igemm_f32_f32/igemmcf_mx1bias_clamp_f32_f32p_f32/tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv.cpp
    src/conv/igemm_f32_f32/igemmcl_1xnbias_clamp_f32_f32_f32t/tqt_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv.cpp
    src/conv/igemm_f32_f32/igemmcl_1xnbias_clamp_f32_f32_f32p/tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv.cpp
    src/dwconv/dwconv_f32_f32/dwconvcf_mx1bias_clamp_f32_f32_f32/tqt_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv.cpp
    src/dwconv/dwconv_f32_f32/dwconvcl_1xnbias_clamp_f32_f32_f32t/tqt_dwconvcl_1xnbias_clamp_f32_f32_f32t_1x1vl_rvv.cpp
    src/dwconv/dwconv_f32_f32/dwconvcf_mx1bias_clamp_f32p_f32p_f32p/tqt_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv.cpp
    src/dwconv/dwconv_f32_f32/dwconvcl_1xnbias_clamp_f32p_f32p_f32p/tqt_dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv.cpp
    src/dwconv/dwconv_f32_f32/dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p/tqt_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv.cpp
    src/dwconv/dwconv_f32_f32/dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p/tqt_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv.cpp
    src/dwconv/dwconv_f32_f32/dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p/tqt_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv.cpp
    src/dwconv/dwconv_f32_f32/dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p/tqt_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv.cpp
    src/dwconv/dwconv_f32_f32/dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p/tqt_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv.cpp
    src/dwconv/dwconv_f32_f32/dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p/tqt_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv.cpp
    # INT8 RVV
    src/gemm/gemm_i8_i8/gemm_clamp_i32_i8_i8/tqt_gemm_clamp_i32_i8_i8_4x1vl_rvv.cpp
    src/gemm/gemm_i8_i8/gemm_clamp_i32_i8p_i8p/tqt_gemm_clamp_i32_i8p_i8p_4x1vl_rvv.cpp
    src/gemm/gemm_i8_i8/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv.cpp
    src/gemm/gemm_i8_i8/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv.cpp
    src/gemm/gemm_i8_i8/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv.cpp
    src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt/tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv.cpp
    src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp/tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv.cpp
    src/gemm/gemm_i8_i8/gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8/tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv.cpp
    src/gemm/gemm_i8_i8/gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p/tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv.cpp
    src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt/tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv.cpp
    src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp/tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv.cpp
    src/conv/conv_i8_i8/convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8/tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv.cpp
    src/conv/conv_i8_i8/convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p/tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv.cpp
    src/conv/conv_i8_i8/convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt/tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv.cpp
    src/conv/conv_i8_i8/convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp/tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv.cpp
    src/conv/igemm_i8_i8/igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8/tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv.cpp
    src/conv/igemm_i8_i8/igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8/tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8_1vlx4_rvv.cpp
    src/conv/igemm_i8_i8/igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt/tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv.cpp
    src/conv/igemm_i8_i8/igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp/tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv.cpp
)

# RVV FP16 (RISC-V Vector + Zfh + Zvfh) kernel files
set(TORQ_TILE_RVV_FP16_SOURCES
    src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16/tqt_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv.cpp
    src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16/tqt_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv.cpp
    src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16/tqt_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv.cpp
    src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16t/tqt_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv.cpp
    src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16p/tqt_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv.cpp
    src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16t/tqt_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv.cpp
    src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16t/tqt_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv.cpp
    src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16p_f16p/tqt_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv.cpp
    src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16p_f16p/tqt_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv.cpp
    src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16p_f16p/tqt_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv.cpp
    src/gemm/gemm_f16_f16/gemm_mx1bias_clamp_f16_f16_f16/tqt_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv.cpp
    src/gemm/gemm_f16_f16/gemm_mx1bias_clamp_f16_f16p_f16p/tqt_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv.cpp
    src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f16_f16_qsi4c2t/tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv.cpp
    src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f16_f16p_qsi4c2p/tqt_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv.cpp
    src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16_qsi4c2t/tqt_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv.cpp
    src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16p_qsi4c2p/tqt_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv.cpp
    src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt/tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv.cpp
    src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp/tqt_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv.cpp
    src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt/tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv.cpp
    src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp/tqt_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv.cpp
    src/conv/conv_f16_f16/convcf_mx1bias_clamp_f16_f16_f16/tqt_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv.cpp
    src/conv/conv_f16_f16/convcf_mx1bias_clamp_f16_f16p_f16p/tqt_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv.cpp
    src/conv/conv_f16_f16/convcl_1xnbias_clamp_f16_f16_f16t/tqt_convcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv.cpp
    src/conv/conv_f16_f16/convcl_1xnbias_clamp_f16_f16p_f16p/tqt_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv.cpp
    src/conv/igemm_f16_f16/igemmcf_mx1bias_clamp_f16_f16_f16/tqt_igemmcf_mx1bias_clamp_f16_f16_f16_3vlx8_rvv.cpp
    src/conv/igemm_f16_f16/igemmcf_mx1bias_clamp_f16_f16p_f16/tqt_igemmcf_mx1bias_clamp_f16_f16p_f16_3vlx8_rvv.cpp
    src/conv/igemm_f16_f16/igemmcl_1xnbias_clamp_f16_f16_f16t/tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv.cpp
    src/conv/igemm_f16_f16/igemmcl_1xnbias_clamp_f16_f16_f16p/tqt_igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv.cpp
    src/dwconv/dwconv_f16_f16/dwconvcf_mx1bias_clamp_f16_f16_f16/tqt_dwconvcf_mx1bias_clamp_f16_f16_f16_1x1vl_rvv.cpp
    src/dwconv/dwconv_f16_f16/dwconvcl_1xnbias_clamp_f16_f16_f16t/tqt_dwconvcl_1xnbias_clamp_f16_f16_f16t_1x1vl_rvv.cpp
    src/dwconv/dwconv_f16_f16/dwconvcf_mx1bias_clamp_f16p_f16p_f16p/tqt_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv.cpp
    src/dwconv/dwconv_f16_f16/dwconvcl_1xnbias_clamp_f16p_f16p_f16p/tqt_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv.cpp
    src/dwconv/dwconv_f16_f16/dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p/tqt_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv.cpp
    src/dwconv/dwconv_f16_f16/dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p/tqt_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv.cpp
    src/dwconv/dwconv_f16_f16/dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p/tqt_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv.cpp
    src/dwconv/dwconv_f16_f16/dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p/tqt_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv.cpp
    src/dwconv/dwconv_f16_f16/dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p/tqt_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv.cpp
    src/dwconv/dwconv_f16_f16/dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p/tqt_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv.cpp
)

# RVV BF16 (RISC-V Vector + Zvfbfwma) kernel files
set(TORQ_TILE_RVV_BF16_SOURCES
    src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16_bf16/tqt_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv.cpp
    src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16_bf16t/tqt_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv.cpp
    src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16_bf16p/tqt_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv.cpp
    src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16p_bf16p/tqt_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv.cpp
    src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f16_bf16_bf16/tqt_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv.cpp
    src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f16_bf16_bf16t/tqt_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv.cpp
    src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f16_bf16p_bf16p/tqt_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv.cpp
    src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f32_bf16_bf16/tqt_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv.cpp
    src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f32_bf16_bf16t/tqt_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv.cpp
    src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f32_bf16p_bf16p/tqt_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv.cpp
    src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt/tqt_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv.cpp
    src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp/tqt_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv.cpp
)

# ============================================================================
# Collect All Sources Based on Detected Extensions
# ============================================================================

set(TORQ_TILE_ALL_SOURCES "")

# Add RVV sources if detected
if(TORQ_TILE_ENABLE_RVV)
    list(APPEND TORQ_TILE_ALL_SOURCES ${TORQ_TILE_RVV_SOURCES})
endif()

# Add RVV FP16 sources if detected
if(TORQ_TILE_ENABLE_RVV_FP16)
    list(APPEND TORQ_TILE_ALL_SOURCES ${TORQ_TILE_RVV_FP16_SOURCES})
endif()

# Add RVV BF16 sources if detected
if(TORQ_TILE_ENABLE_RVV_BF16)
    list(APPEND TORQ_TILE_ALL_SOURCES ${TORQ_TILE_RVV_BF16_SOURCES})
endif()

# ============================================================================
# Library Target Definition
# ============================================================================

# Create library
if(TORQ_TILE_BUILD_SHARED)
    add_library(torq_tile SHARED ${TORQ_TILE_ALL_SOURCES})
else()
    add_library(torq_tile STATIC ${TORQ_TILE_ALL_SOURCES})
endif()

# Include directories
target_include_directories(torq_tile
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Link libraries
target_link_libraries(torq_tile PUBLIC m)

# Set library properties
set_target_properties(torq_tile PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Set PIC for shared library
if(TORQ_TILE_BUILD_SHARED)
    set_target_properties(torq_tile PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_compile_options(torq_tile PRIVATE)

# ============================================================================
# Subdirectories (Test and Benchmark)
# ============================================================================

# Fetch dependencies via FetchContent
include(FetchContent)

# Fetch Google Test if external source directory not provided
if(DEFINED ENV{FETCHCONTENT_SOURCE_DIR_GOOGLETEST})
    set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST "$ENV{FETCHCONTENT_SOURCE_DIR_GOOGLETEST}")
    message(STATUS "Using external googletest source directory: ${FETCHCONTENT_SOURCE_DIR_GOOGLETEST}")
    FetchContent_Declare(
        googletest
        SOURCE_DIR ${FETCHCONTENT_SOURCE_DIR_GOOGLETEST}
    )
else()
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.17.0
        GIT_SHALLOW    TRUE
    )
endif()

# Fetch Google Benchmark if external source directory not provided
if(DEFINED ENV{FETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK})
    set(FETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK
        "$ENV{FETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK}")
    message(STATUS "Using external googlebenchmark source directory: ${FETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK}")
    FetchContent_Declare(
        googlebenchmark
        SOURCE_DIR ${FETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK}
    )
else()
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.5
        GIT_SHALLOW    TRUE
    )
endif()

# Add test subdirectory if enabled
if(TORQ_TILE_BUILD_TEST)
    enable_testing()

    # Configure Google Test options
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_subdirectory(test)
endif()

# Add benchmark subdirectory if enabled
if(TORQ_TILE_BUILD_BENCHMARK)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)

    add_subdirectory(benchmark)
endif()

# ============================================================================
# Installation Rules
# ============================================================================

# Install library target
install(TARGETS torq_tile
    EXPORT torq_tile_targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install header files
install(DIRECTORY src/
    DESTINATION include/torq_tile
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "todo" EXCLUDE
)

# Install CMake config files
install(EXPORT torq_tile_targets
    FILE torq_tile_targets.cmake
    NAMESPACE torq_tile::
    DESTINATION lib/cmake/torq_tile
)
