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
/// This is a rvv (8 x 3*vl tile) implementation where vl = csrr_vlenb() / sizeof(float16_t).
/// Computes: D = clamp(C + A*B + bias, min, max)
/// where B is transposed (stored as n x k), and bias has shape [1, N] and is broadcast
/// across rows.

/// --------------------------------------------------

/// Gets m step value.
///
/// The starting row index must be divisible by `m_step`.
///
/// @return The m step value (8).
size_t tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(void);

/// Gets n step value.
///
/// The starting column index must be divisible by `n_step`.
///
/// @return The n step value (3 * vl, where vl = csrr_vlenb() / sizeof(float16_t)).
size_t tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(void);

/// Gets the offset in bytes to the data element in the A matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] k_idx Column index.
/// @param[in] lda Leading dimension of A.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(size_t m_idx, size_t k_idx,
                                                                  size_t lda);

/// Gets the offset in bytes to the data element in the B matrix buffer.
///
/// @param[in] n_idx Column index.
/// @param[in] k_idx Row index.
/// @param[in] ldb Leading dimension of B.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(size_t n_idx, size_t k_idx,
                                                                  size_t ldb);

/// Gets the offset in bytes to the data element in the C matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] ldc Leading dimension of C.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                  size_t ldc);

/// Gets the offset in bytes to the data element in the bias vector buffer.
///
/// @param[in] n_idx Column index.
///
/// @return The offset in bytes to the bias element.
size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(size_t n_idx);

/// Gets the offset in bytes to the data element in the D (output) matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] ldd Leading dimension of D.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                  size_t ldd);

/// Gets the size in bytes of the D (output) matrix buffer.
///
/// @param[in] m Number of rows.
/// @param[in] n Number of columns.
///
/// @return The size in bytes of the output matrix buffer.
size_t tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(size_t m, size_t n);

/// Runs the GEMM microkernel with bias addition and clamp operation.
///
/// Computes: D = clamp(C + A*B + bias, clamp_min, clamp_max)
///
/// @param[in]  m Number of rows in A and D.
/// @param[in]  n Number of columns in B and D.
/// @param[in]  k Common dimension (columns of A, rows of B).
/// @param[in]  A Input matrix A [m, k] in row-major format.
/// @param[in]  lda Leading dimension of A.
/// @param[in]  k_idx_a K index in A buffer.
/// @param[in]  B Input matrix B [n, k] in row-major format (transposed).
/// @param[in]  ldb Leading dimension of B.
/// @param[in]  k_idx_b K index in B buffer.
/// @param[in]  C Input matrix C [m, n] in row-major format.
/// @param[in]  ldc Leading dimension of C.
/// @param[out] D Output matrix D [m, n] in row-major format.
/// @param[in]  ldd Leading dimension of D.
/// @param[in]  bias Bias vector [1, n], broadcast across all rows.
/// @param[in]  clamp_min Minimum value for clamping.
/// @param[in]  clamp_max Maximum value for clamping.
///
/// @note All matrices are in row-major format with float16_t elements.
/// @note B is transposed, stored as [n, k].
/// @note The bias vector has shape [1, n] and is added to each row of the result.
/// @note k_idx_a and k_idx_b are used when A and B are sub-buffers with different sizes.
///
void tqt_run_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(size_t m, size_t n, size_t k, const void *A,
                                                       size_t lda, size_t k_idx_a, const void *B,
                                                       size_t ldb, size_t k_idx_b, const void *C,
                                                       size_t ldc, void *D, size_t ldd,
                                                       const void *bias, float16_t clamp_min,
                                                       float16_t clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
