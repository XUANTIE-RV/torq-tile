//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include "src/common/tqt_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Micro-kernel: GEMM with 1xN bias and clamp, BF16 inputs and FP32 output.
///
/// This is a rvv (8 x vl tile) implementation where vl = csrr_vlenb() / sizeof(bfloat16_t).
/// Computes: D = clamp(C + A*B + bias, min, max)
/// where B is in standard format (stored as k x n), and bias has shape [1, N] and is broadcast
/// across rows.
/// Internally accumulates in FP32 using vfwmaccbf16.vf and stores fp32 directly (no narrow).

/// --------------------------------------------------

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv(void);
size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv(void);
size_t tqt_get_a_offset_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv(size_t m_idx, size_t k_idx,
                                                                   size_t lda);
size_t tqt_get_b_offset_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv(size_t n_idx, size_t k_idx,
                                                                   size_t ldb);
size_t tqt_get_c_offset_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                   size_t ldc);
size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv(size_t n_idx);
size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                   size_t ldd);
size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv(size_t m, size_t n);

/// Runs the GEMM microkernel with bias addition and clamp operation.
///
/// Computes: D = clamp(C + A*B + bias, clamp_min, clamp_max)
///
/// @note A, B are bfloat16_t; C, D, bias are float.
///
void tqt_run_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv(size_t m, size_t n, size_t k, const void *A,
                                                        size_t lda, size_t k_idx_a, const void *B,
                                                        size_t ldb, size_t k_idx_b, const void *C,
                                                        size_t ldc, void *D, size_t ldd,
                                                        const void *bias, float clamp_min,
                                                        float clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
