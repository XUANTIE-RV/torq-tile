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

/// Micro-kernel: GEMM with Mx1 bias and clamp for BF16
///
/// This is a rvv (8 x vl tile) implementation where vl = csrr_vlenb() / sizeof(bfloat16_t).
/// Computes: D = clamp(C + A*B + bias, min, max)
/// where B is in standard format (stored as k x n), and bias has shape [M, 1] and is broadcast
/// across columns.
/// Internally accumulates in FP32 using vfwmaccbf16.vf, then narrows back to BF16.

/// --------------------------------------------------

/// Gets m step value. The starting row index must be divisible by `m_step`.
/// @return The m step value (8).
size_t tqt_get_m_step_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv(void);

/// Gets n step value. The starting column index must be divisible by `n_step`.
/// @return The n step value (vl, where vl = csrr_vlenb() / sizeof(bfloat16_t)).
size_t tqt_get_n_step_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv(void);

/// Gets the offset in bytes to the data element in the A matrix buffer.
size_t tqt_get_a_offset_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv(size_t m_idx, size_t k_idx,
                                                                    size_t lda);

/// Gets the offset in bytes to the data element in the B matrix buffer.
size_t tqt_get_b_offset_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv(size_t n_idx, size_t k_idx,
                                                                    size_t ldb);

/// Gets the offset in bytes to the data element in the C matrix buffer.
size_t tqt_get_c_offset_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                    size_t ldc);

/// Gets the offset in bytes to the data element in the bias vector buffer.
/// @param[in] m_idx Row index.
/// @return The offset in bytes to the bias element.
size_t tqt_get_bias_offset_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv(size_t m_idx);

/// Gets the offset in bytes to the data element in the D (output) matrix buffer.
size_t tqt_get_d_offset_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                    size_t ldd);

/// Gets the size in bytes of the D (output) matrix buffer.
size_t tqt_get_d_size_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv(size_t m, size_t n);

/// Runs the GEMM microkernel with bias addition and clamp operation.
///
/// Computes: D = clamp(C + A*B + bias, clamp_min, clamp_max)
///
/// @note All matrices are in row-major format with bfloat16_t elements.
/// @note B is in standard format, stored as [k, n].
/// @note The bias vector has shape [M, 1] and is added to each column of the result.
void tqt_run_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
    size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
    bfloat16_t clamp_min, bfloat16_t clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
