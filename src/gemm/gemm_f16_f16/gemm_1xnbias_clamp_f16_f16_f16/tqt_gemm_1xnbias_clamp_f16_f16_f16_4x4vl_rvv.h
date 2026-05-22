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

/// Micro-kernel: GEMM with 1xN bias and clamp for FP16
///
/// This is a rvv (4 x 4vl tile) implementation using LMUL=4 configuration.
/// vl = csrr_vlenb() / sizeof(float16_t), n_step = vl * 4.
/// Computes: D = clamp(C + A*B + bias, min, max)
/// where B is in standard format (stored as k x n), and bias has shape [1, N] and is broadcast
/// across rows.

size_t tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv(void);
size_t tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv(void);
size_t tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv(size_t m_idx, size_t k_idx,
                                                                 size_t lda);
size_t tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv(size_t n_idx, size_t k_idx,
                                                                 size_t ldb);
size_t tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv(size_t m_idx, size_t n_idx,
                                                                 size_t ldc);
size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv(size_t n_idx);
size_t tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv(size_t m_idx, size_t n_idx,
                                                                 size_t ldd);
size_t tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv(size_t m, size_t n);

/// Runs the GEMM microkernel with bias addition and clamp operation.
/// Computes: D = clamp(C + A*B + bias, clamp_min, clamp_max)
/// Uses LMUL=4 with runtime-variable vl to handle any N tail in [1, n_step].
void tqt_run_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv(size_t m, size_t n, size_t k, const void *A,
                                                      size_t lda, size_t k_idx_a, const void *B,
                                                      size_t ldb, size_t k_idx_b, const void *C,
                                                      size_t ldc, void *D, size_t ldd,
                                                      const void *bias, float16_t clamp_min,
                                                      float16_t clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
