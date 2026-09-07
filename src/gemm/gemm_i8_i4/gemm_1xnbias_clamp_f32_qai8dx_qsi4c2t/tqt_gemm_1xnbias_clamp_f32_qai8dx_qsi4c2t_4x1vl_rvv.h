//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_i8_i4/tqt_params_i8_i4.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Micro-kernel: W4A8 block-quantized GEMM with 1xN bias and clamp (D = f32 output)
///
/// rvv (4 x vl tile) implementation where vl = csrr_vlenb()
/// (i8 elements per vector at e8/m1).
/// Computes:
///   D = clamp(C + dequant(A_qai8dx) * dequant(B_qsi4c2t) + bias, min, max)
/// where:
///   - A is qai8dx: per-row asymmetric int8 (no block dimension), [m, k] row-major.
///     scale_a is [m] fp32 (per-row). zero_point_a is [m] int32 (per-row).
///   - B is qsi4c2t: per-block symmetric int4 along K (block size = bl),
///     [n, k] transposed (K-contiguous), stored as [n, k/2] uint8 with two
///     nibbles per byte (lo = k+0, hi = k+1) using ZP=8 unsigned encoding
///     (signed [-8, 7] stored as [0, 15]). scale_b is [n, k/bl] fp32.
///   - bias is f32[n] broadcast across rows (optional).
///   - C is f32[m, n] (optional, can be nullptr).
///   - D is f32[m, n].
///
/// Algorithm (per output cell):
///   acc = scale_a[m] * sum_blk(scale_b[n, blk]
///                   * sum_{k in blk}((qa[m, k] - zp_a[m]) * qb_signed[n, k]))
///   D[m, n] = clamp(C[m, n] + acc + bias[n], min, max)
///
/// k must be a multiple of bl. bl must be >= 2 and a multiple of 2.

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv(void);

size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv(void);

/// Gets A offset in bytes (A is i8 row-major [m, k]).
size_t tqt_get_a_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv(size_t m_idx, size_t k_idx,
                                                                        size_t lda, size_t bl);

/// Gets B offset in bytes. B is [n, k] int4 stored as [n, k/2] uint8.
/// ldb is in int4 elements (= k). k_idx must be a multiple of bl.
size_t tqt_get_b_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv(size_t n_idx, size_t k_idx,
                                                                        size_t ldb, size_t bl);

/// Gets scale_b offset in bytes. scale_b is [n, k/bl] fp32 row-major.
size_t tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv(size_t n_idx,
                                                                              size_t b_idx,
                                                                              size_t ldsb);

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                        size_t ldc);

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv(size_t n_idx);

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                        size_t ldd);

size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv(size_t m, size_t n);

/// Runs the W4A8 block-quantized GEMM microkernel.
///
/// @param[in]  m         Number of rows in A and D.
/// @param[in]  n         Number of columns in D.
/// @param[in]  k         Common dimension. Must be a multiple of bl.
/// @param[in]  bl        Block length for quantization scales, >= 2 and multiple of 2.
/// @param[in]  A         Input matrix A [m, k] qai8dx (int8) row-major.
/// @param[in]  lda       Leading dimension of A (in i8 elements).
/// @param[in]  k_idx_a   K index in A buffer (reserved for K-tiling).
/// @param[in]  B         Input matrix B [n, k] qsi4c2t (int4 transposed),
///                        stored as [n, k/2] uint8.
/// @param[in]  ldb       Leading dimension of B (in int4 elements, = k).
/// @param[in]  k_idx_b   K index in B buffer (reserved for K-tiling).
/// @param[in]  C         Input matrix C [m, n] f32 (optional, can be nullptr).
/// @param[in]  ldc       Leading dimension of C (in f32 elements).
/// @param[out] D         Output matrix D [m, n] f32.
/// @param[in]  ldd       Leading dimension of D (in f32 elements).
/// @param[in]  bias      Bias vector [1, n] f32 (optional, can be nullptr).
/// @param[in]  clamp_min Minimum clamp value.
/// @param[in]  clamp_max Maximum clamp value.
/// @param[in]  params    Quantization parameters
///                        (scale_a[m], zero_point_a[m], scale_b[n, k/bl] + ldsb).
/// @note If clamp_min is negative infinity and clamp_max is positive infinity,
///       clamping is skipped. All other bounds retain the existing clamp behavior.
///       This does not guarantee preservation of NaN payloads or exception flags.
void tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv(
    size_t m, size_t n, size_t k, size_t bl, const void *A, size_t lda, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
    const void *bias, float clamp_min, float clamp_max,
    const struct tqt_qai8dx_qsi4c2_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
