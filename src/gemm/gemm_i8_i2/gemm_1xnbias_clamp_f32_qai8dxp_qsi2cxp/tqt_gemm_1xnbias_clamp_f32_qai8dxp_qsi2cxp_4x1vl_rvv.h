//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_i8_i2/tqt_params_i8_i2.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Micro-kernel: GEMM with 1xN bias and clamp for qai8dxp (A packed, per-row asymmetric int8)
/// x qsi2cxp (B packed, per-channel symmetric int2) -> f32 (D).
///
/// 4 x vl tile, RVV. Computes:
///     D[m][n] = clamp( bias[n] + scale_a[m]*scale_b[n] *
///                          ( Σ_k q_A[m][k] * lut[u2_b[n][k]]
///                            - zp_a[m] * Σ_k lut[u2_b[n][k]] ),
///                       clamp_min, clamp_max )
///
/// Quantization scheme:
/// - A: per-dimension asymmetric int8 (M scales + M zero points)
/// - B: per-channel symmetric int2 (N scales, no zero point), packed 4-per-byte
///
/// K must be a multiple of 4. The default LUT is {-2, -1, 0, 1}; pass nullptr for `lut`
/// to use the default. Custom 4-entry int8 LUTs are accepted.

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(void);
size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(void);

/// A_packed has the layout `[M/m_step, K, m_step]` int8 (interleaved across rows).
size_t tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t m_idx,
                                                                                size_t k_idx,
                                                                                size_t lda,
                                                                                size_t actual_m);

/// B_packed has the layout `[N/n_step, K/4, n_step]` uint8 (interleaved across columns,
/// with each byte holding four 2-bit weights `s3 s2 s1 s0`).
size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t n_idx,
                                                                                size_t k_idx,
                                                                                size_t ldb,
                                                                                size_t actual_n);

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t n_idx);
size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                         size_t ldd);
size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t m, size_t n);
size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t m, size_t k);
size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t n, size_t k);

/// Pack matrix A: [M, K] int8 row-major -> [M/m_step, K, m_step] int8.
void tqt_run_a_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t m, size_t k, size_t lda,
                                                                     size_t lda_packed,
                                                                     size_t k_idx, const void *A,
                                                                     void *A_packed);

/// Pack matrix B (non-transposed): [K/4, N] uint8 row-major -> [N/n_step, K/4, n_step] uint8.
/// The K dimension is given in K elements (must be a multiple of 4); each input byte holds
/// 4 elements packed `s3 s2 s1 s0`.
void tqt_run_b_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t n, size_t k, size_t ldb,
                                                                     size_t ldb_packed,
                                                                     size_t k_idx, const void *B,
                                                                     void *B_packed);

/// Pack matrix B (transposed): [N, K/4] uint8 row-major -> [N/n_step, K/4, n_step] uint8.
void tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t n, size_t k,
                                                                      size_t ldb, size_t ldb_packed,
                                                                      size_t k_idx, const void *B,
                                                                      void *B_packed);

/// Runs the GEMM micro-kernel.
///
/// @param[in]  m         Number of rows in A and D.
/// @param[in]  n         Number of columns in B and D.
/// @param[in]  k         Common dimension. Must be a multiple of 4.
/// @param[in]  A_packed  Packed A buffer (qai8dxp).
/// @param[in]  lda       Leading dimension of A_packed in elements (= K when packed contiguous).
/// @param[in]  k_idx_a   K index in A buffer (for sub-buffer scenarios).
/// @param[in]  B_packed  Packed B buffer (qsi2cxp).
/// @param[in]  ldb       Leading dimension of B_packed in *bytes* per packed N-block (= K/4).
/// @param[in]  k_idx_b   K index in B buffer.
/// @param[out] D         Output matrix [M, N] fp32 row-major.
/// @param[in]  ldd       Leading dimension of D in elements.
/// @param[in]  bias      Optional [1, N] fp32 bias broadcast across rows; nullptr means no bias.
/// @param[in]  clamp_min Minimum clamp value.
/// @param[in]  clamp_max Maximum clamp value.
/// @param[in]  lut       Optional 4-entry int8 LUT for u2 -> i8 dequant. nullptr = default
///                       {-2, -1, 0, 1}.
/// @param[in]  params    Quantization parameters (scale_a[M], zero_point_a[M],
///                       scale_b[N], sum_b[N]).
/// @note If clamp_min is negative infinity and clamp_max is positive infinity,
///       clamping is skipped. All other bounds retain the existing clamp behavior.
///       This does not guarantee preservation of NaN payloads or exception flags.
void tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, void *D, size_t ldd, const void *bias,
    float clamp_min, float clamp_max, const int8_t *lut,
    const struct tqt_qai8dx_qsi2cx_f32_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
