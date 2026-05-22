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

/// gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv
///
/// - A: per-row asymmetric int8, packed (rows interleaved by mr along K)
/// - B: per-channel symmetric int8, packed (cols interleaved by nr along K)
/// - D: f16 output, [M, N] row-major
/// - bias: optional f16, length N (1xN row-broadcast)
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
size_t tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(void);

/// Gets n step value.
///
/// The starting column index must be divisible by `n_step`.
///
/// @return The n step value (vl, where vl = csrr_vlenb() / sizeof(int8)).
size_t tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(void);

/// Gets the offset in bytes to the data element in the packed A buffer.
///
/// Within each m_step-row tile, rows are interleaved at byte granularity along K.
///
/// @param[in] m_idx Row index (must be divisible by m_step).
/// @param[in] k_idx Column index along K.
/// @param[in] lda Leading dimension of packed A.
/// @param[in] actual_m Number of valid rows in the current tile (<= m_step).
///
/// @return The offset in bytes to the data element.
size_t tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(size_t m_idx,
                                                                                size_t k_idx,
                                                                                size_t lda,
                                                                                size_t actual_m);

/// Gets the offset in bytes to the data element in the packed B buffer.
///
/// Within each n_step-column tile, columns are interleaved at byte granularity along K.
///
/// @param[in] n_idx Output channel index (must be divisible by n_step).
/// @param[in] k_idx Row index along K.
/// @param[in] ldb Leading dimension of packed B.
/// @param[in] actual_n Number of valid columns in the current tile (<= n_step).
///
/// @return The offset in bytes to the data element.
size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(size_t n_idx,
                                                                                size_t k_idx,
                                                                                size_t ldb,
                                                                                size_t actual_n);

/// Gets the offset in bytes to the data element in the C matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] ldc Leading dimension of C.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_c_offset_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                         size_t ldc);

/// Gets the offset in bytes to the bias vector buffer.
///
/// @param[in] n_idx Column index.
///
/// @return The offset in bytes to the bias element.
size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(size_t n_idx);

/// Gets the offset in bytes to the data element in the D (output) matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] ldd Leading dimension of D.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_d_offset_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                         size_t ldd);

/// Gets the size in bytes of the D (output) matrix buffer.
///
/// @param[in] m Number of rows.
/// @param[in] n Number of columns.
///
/// @return The size in bytes of the output matrix buffer.
size_t tqt_get_d_size_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(size_t m, size_t n);

/// Gets the size in bytes of the packed A buffer.
///
/// @param[in] m Number of rows.
/// @param[in] k Common dimension.
///
/// @return The size in bytes of the packed A buffer.
size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(size_t m, size_t k);

/// Gets the size in bytes of the packed B buffer.
///
/// @param[in] n Number of columns (output channels).
/// @param[in] k Common dimension.
///
/// @return The size in bytes of the packed B buffer.
size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(size_t n, size_t k);

/// Packs the A matrix from row-major [M, K] into the kernel's interleaved layout.
///
/// @param[in]  m Number of rows.
/// @param[in]  k Common dimension.
/// @param[in]  lda Leading dimension of source A.
/// @param[in]  lda_packed Leading dimension of packed A.
/// @param[in]  k_idx K index in source buffer (sub-buffer support).
/// @param[in]  A Source A matrix [m, k] (int8).
/// @param[out] A_packed Destination packed A buffer.
void tqt_run_a_pack_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(size_t m, size_t k, size_t lda,
                                                                     size_t lda_packed,
                                                                     size_t k_idx, const void *A,
                                                                     void *A_packed);

/// Packs the B matrix from row-major [K, N] into the kernel's interleaved layout.
///
/// @param[in]  n Number of columns (output channels).
/// @param[in]  k Common dimension.
/// @param[in]  ldb Leading dimension of source B (N stride).
/// @param[in]  ldb_packed Leading dimension of packed B.
/// @param[in]  k_idx K index in source buffer (sub-buffer support).
/// @param[in]  B Source B matrix [k, n] (int8).
/// @param[out] B_packed Destination packed B buffer.
void tqt_run_b_pack_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(size_t n, size_t k, size_t ldb,
                                                                     size_t ldb_packed,
                                                                     size_t k_idx, const void *B,
                                                                     void *B_packed);

/// Packs the B matrix from transposed layout [N, K] into the kernel's interleaved layout.
///
/// @param[in]  n Number of columns (output channels).
/// @param[in]  k Common dimension.
/// @param[in]  ldb Leading dimension of source B (K stride).
/// @param[in]  ldb_packed Leading dimension of packed B.
/// @param[in]  k_idx K index in source buffer (sub-buffer support).
/// @param[in]  B Source B matrix [n, k] (int8, transposed view).
/// @param[out] B_packed Destination packed B buffer.
void tqt_run_bt_pack_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(size_t n, size_t k,
                                                                      size_t ldb, size_t ldb_packed,
                                                                      size_t k_idx, const void *B,
                                                                      void *B_packed);

/// Runs the GEMM microkernel with 1xN bias and clamp on packed A and B.
///
/// Computes: D = clamp(dequant(A_packed * B_packed) + bias, clamp_min, clamp_max)
/// where A is per-row asymmetric int8 (packed), B is per-channel symmetric int8 (packed),
/// and D is fp16 (no output quantization).
///
/// @param[in]  m Number of rows in A and D.
/// @param[in]  n Number of columns in D (output channels of B).
/// @param[in]  k Common dimension.
/// @param[in]  A_packed Packed A buffer.
/// @param[in]  lda Leading dimension of packed A.
/// @param[in]  k_idx_a K index in A buffer (sub-buffer support).
/// @param[in]  B_packed Packed B buffer.
/// @param[in]  ldb Leading dimension of packed B.
/// @param[in]  k_idx_b K index in B buffer (sub-buffer support).
/// @param[in]  C Residual matrix (currently must be NULL).
/// @param[in]  ldc Leading dimension of C (unused).
/// @param[out] D Output matrix D [m, n] in row-major format (fp16).
/// @param[in]  ldd Leading dimension of D.
/// @param[in]  bias Bias vector [1, n] (fp16), broadcast across all rows. May be NULL.
/// @param[in]  clamp_min Minimum value for clamping (fp16).
/// @param[in]  clamp_max Maximum value for clamping (fp16).
/// @param[in]  params Quantization parameters (per-row scale_a/zero_point_a, per-channel scale_b).
///
/// @note D is in row-major format.
/// @note C residual is currently not supported and must be NULL.
void tqt_run_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float16_t clamp_min, float16_t clamp_max,
    const struct tqt_qai8dxp_qsi8cxp_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif
