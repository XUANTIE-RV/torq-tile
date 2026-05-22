//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_i8_i8/tqt_params_i8_i8.h"

#ifdef __cplusplus
extern "C" {
#endif

/// gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv
///
/// - A: per-row asymmetric int8, [M, K] row-major (not packed)
/// - B: per-channel symmetric int8, [N, K] row-major (qsi8cxt = transposed view)
/// - D: f32 output, [M, N] row-major
/// - bias: optional f32, length N (1xN row-broadcast)
///
/// Tile: m_step = 4, n_step = vlmax (RLEN/8 bytes).
/// Inner loop: vsext.vf2 + vwmacc.vx (SEW=16) with per-row LHS zero-point correction.
///
/// C residual is currently not supported and must be NULL.

/// Gets m step value.
///
/// The starting row index must be divisible by `m_step`.
///
/// @return The m step value (4).
size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv(void);

/// Gets n step value.
///
/// The starting column index must be divisible by `n_step`.
///
/// @return The n step value (vl, where vl = csrr_vlenb() / sizeof(int8)).
size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv(void);

/// Gets the offset in bytes to the data element in the A matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] k_idx Column index.
/// @param[in] lda Leading dimension of A.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_a_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv(size_t m_idx, size_t k_idx,
                                                                        size_t lda);

/// Gets the offset in bytes to the data element in the B matrix buffer.
///
/// B is stored in transposed layout [N, K]; n_idx selects the row of the
/// transposed view (i.e. the output channel) and k_idx selects the column.
///
/// @param[in] n_idx Output channel index (row of [N, K] view).
/// @param[in] k_idx Column index along K.
/// @param[in] ldb Leading dimension of B (K stride).
///
/// @return The offset in bytes to the data element.
size_t tqt_get_b_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv(size_t n_idx, size_t k_idx,
                                                                        size_t ldb);

/// Gets the offset in bytes to the data element in the C matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] ldc Leading dimension of C.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_c_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                        size_t ldc);

/// Gets the offset in bytes to the bias vector buffer.
///
/// @param[in] n_idx Column index.
///
/// @return The offset in bytes to the bias element.
size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv(size_t n_idx);

/// Gets the offset in bytes to the data element in the D (output) matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] ldd Leading dimension of D.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                        size_t ldd);

/// Gets the size in bytes of the D (output) matrix buffer.
///
/// @param[in] m Number of rows.
/// @param[in] n Number of columns.
///
/// @return The size in bytes of the output matrix buffer.
size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv(size_t m, size_t n);

/// Runs the GEMM microkernel with 1xN bias and clamp.
///
/// Computes: D = clamp(dequant(A * B^T) + bias, clamp_min, clamp_max)
/// where A is per-row asymmetric int8, B is per-channel symmetric int8 (transposed),
/// and D is fp32 (no output quantization).
///
/// @param[in]  m Number of rows in A and D.
/// @param[in]  n Number of columns in D (output channels of B).
/// @param[in]  k Common dimension (columns of A, columns of B^T view).
/// @param[in]  A Input matrix A [m, k] in row-major format (int8).
/// @param[in]  lda Leading dimension of A.
/// @param[in]  k_idx_a K index in A buffer (unused for non-packed variant).
/// @param[in]  B Input matrix B [n, k] in row-major format (int8, transposed view).
/// @param[in]  ldb Leading dimension of B (K stride).
/// @param[in]  k_idx_b K index in B buffer (unused for non-packed variant).
/// @param[in]  C Residual matrix (currently must be NULL).
/// @param[in]  ldc Leading dimension of C (unused).
/// @param[out] D Output matrix D [m, n] in row-major format (fp32).
/// @param[in]  ldd Leading dimension of D.
/// @param[in]  bias Bias vector [1, n] (fp32), broadcast across all rows. May be NULL.
/// @param[in]  clamp_min Minimum value for clamping (fp32).
/// @param[in]  clamp_max Maximum value for clamping (fp32).
/// @param[in]  params Quantization parameters (per-row scale_a/zero_point_a, per-channel scale_b).
///
/// @note All matrices are in row-major format.
/// @note B is stored as [n, k] (transposed view of [k, n]).
/// @note C residual is currently not supported and must be NULL.
void tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
    size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
    float clamp_min, float clamp_max, const struct tqt_qai8dxp_qsi8cxp_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif
