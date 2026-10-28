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

/// Micro-kernel: GEMM with 1xN bias and clamp for qai8p (A packed, per-tensor asymmetric int8)
/// x qsi2cxp (B packed, per-channel symmetric int2) -> qai8 (D).
///
/// 4 x vl tile, RVV. Computes:
///     acc = bias[n] + Σ_k (q_A[m][k] - zero_point_a) * lut[u2_b[n][k]]
///     D[m][n] = clamp(RNE(float(acc) * scale_a * scale_b[n] / scale_d)
///                     + zero_point_d, clamp_min, clamp_max)
///
/// Quantization scheme:
/// - A: per-tensor asymmetric int8 (scalar scale_a and zero_point_a)
/// - B: per-channel symmetric int2 (N scales, no zero point), packed 4-per-byte
/// - D: per-tensor asymmetric int8 (scalar scale_d and zero_point_d)
/// - bias: optional I32 accumulator bias, length N
///
/// K must be a multiple of 4. The default LUT is {-2, -1, 0, 1}; pass nullptr for `lut`
/// to use the default. Custom 4-entry int8 LUTs are accepted.

size_t tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv(void);
size_t tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv(void);

/// A_packed has the layout `[M/m_step, K, m_step]` int8 (interleaved across rows).
size_t tqt_get_a_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv(
    size_t m_idx, size_t k_idx, size_t lda, size_t actual_m);

/// B_packed has the layout `[N/n_step, K/4, n_step]` uint8 (interleaved across columns,
/// with each byte holding four 2-bit weights `s3 s2 s1 s0`).
size_t tqt_get_b_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv(
    size_t n_idx, size_t k_idx, size_t ldb, size_t actual_n);

size_t tqt_get_bias_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv(size_t n_idx);
size_t tqt_get_d_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv(size_t m_idx,
                                                                                   size_t n_idx,
                                                                                   size_t ldd);
size_t tqt_get_d_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv(size_t m,
                                                                                 size_t n);
size_t tqt_get_a_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv(size_t m,
                                                                                        size_t k);
size_t tqt_get_b_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv(size_t n,
                                                                                        size_t k);

/// Pack matrix A: [M, K] int8 row-major -> [M/m_step, K, m_step] int8.
void tqt_run_a_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv(
    size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx, const void *A, void *A_packed);

/// Pack matrix B (non-transposed): [K/4, N] uint8 row-major -> [N/n_step, K/4, n_step] uint8.
/// The K dimension is given in K elements (must be a multiple of 4); each input byte holds
/// 4 elements packed `s3 s2 s1 s0`.
void tqt_run_b_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed);

/// Pack matrix B (transposed): [N, K/4] uint8 row-major -> [N/n_step, K/4, n_step] uint8.
void tqt_run_bt_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed);

/// Runs the GEMM micro-kernel.
///
/// @param[in]  m         Number of rows in A and D.
/// @param[in]  n         Number of columns in B and D.
/// @param[in]  k         Common dimension. Must be a multiple of 4.
/// @param[in]  A_packed  Packed A buffer (qai8p).
/// @param[in]  lda       Leading dimension of A_packed in elements (= K when packed contiguous).
/// @param[in]  k_idx_a   K index in A buffer (for sub-buffer scenarios).
/// @param[in]  B_packed  Packed B buffer (qsi2cxp).
/// @param[in]  ldb       Packed B stride in int2 elements; divisible by 4 (K if contiguous).
/// @param[in]  k_idx_b   K index in B buffer.
/// @param[out] D         Output matrix [M, N] int8 row-major.
/// @param[in]  ldd       Leading dimension of D in elements.
/// @param[in]  bias      Optional [1, N] I32 accumulator bias broadcast across rows; nullptr means
/// no bias.
/// @param[in]  clamp_min Minimum clamp value.
/// @param[in]  clamp_max Maximum clamp value.
/// @param[in]  lut       Optional 4-entry int8 LUT for u2 -> i8 dequant. nullptr = default
///                       {-2, -1, 0, 1}.
/// @param[in]  params    Scalar scale_a, zero_point_a, scale_d, zero_point_d, and scale_b[N].
/// K and both K offsets must be multiples of four. Pad A with its zero point for a K tail.
/// The kernel computes its own matching decoded weight sums for each logical K window.
void tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, void *D, size_t ldd, const int32_t *bias,
    int8_t clamp_min, int8_t clamp_max, const int8_t *lut,
    const struct tqt_qai8_qai8_qsi2cx_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
