//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Bias is an optional FP32 row vector of N elements; bias offsets count bytes.
// C, D and clamp bounds retain the output type. Bias is added before narrowing.

#include <stddef.h>
#include <stdint.h>

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_i8_i4/tqt_params_i8_i4.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Micro-kernel: packed W4A8 block-quantized GEMM with 1xN bias and clamp
/// (D = f16 output). See _interface.h and the f32 variant for details.

size_t tqt_get_m_step_gemm_1xnbiasf32_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv(void);
size_t tqt_get_n_step_gemm_1xnbiasf32_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv(void);

size_t tqt_get_a_packed_offset_gemm_1xnbiasf32_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t m_idx,
                                                                                   size_t k_idx,
                                                                                   size_t lda,
                                                                                   size_t actual_m);

size_t tqt_get_b_packed_offset_gemm_1xnbiasf32_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t n_idx,
                                                                                   size_t k_idx,
                                                                                   size_t ldb,
                                                                                   size_t actual_n);

size_t tqt_get_scale_a_offset_gemm_1xnbiasf32_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t m_idx,
                                                                                  size_t b_idx,
                                                                                  size_t ldsa);

size_t tqt_get_scale_b_offset_gemm_1xnbiasf32_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t n_idx,
                                                                                  size_t b_idx,
                                                                                  size_t ldsb);

size_t tqt_get_c_offset_gemm_1xnbiasf32_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t m_idx,
                                                                            size_t n_idx,
                                                                            size_t ldc);
size_t tqt_get_bias_offset_gemm_1xnbiasf32_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t n_idx);
size_t tqt_get_d_offset_gemm_1xnbiasf32_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t m_idx,
                                                                            size_t n_idx,
                                                                            size_t ldd);
size_t tqt_get_d_size_gemm_1xnbiasf32_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t m, size_t n);
size_t tqt_get_a_packed_size_gemm_1xnbiasf32_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t m,
                                                                                 size_t k);
size_t tqt_get_b_packed_size_gemm_1xnbiasf32_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t n,
                                                                                 size_t k);

void tqt_run_a_pack_gemm_1xnbiasf32_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv(
    size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx, const void *A, void *A_packed);

void tqt_run_bt_pack_gemm_1xnbiasf32_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed);

/// @note If clamp_min is negative infinity and clamp_max is positive infinity,
///       clamping is skipped. All other bounds retain the existing clamp behavior.
///       This does not guarantee preservation of NaN payloads or exception flags.
/// F32 bias is added before the final output conversion; bias offsets count bytes.
void tqt_run_gemm_1xnbiasf32_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv(
    size_t m, size_t n, size_t k, size_t bl, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float16_t clamp_min, float16_t clamp_max,
    const struct tqt_qsi8d2_qsi4c2_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
