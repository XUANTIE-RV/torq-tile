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

/// Micro-kernel: W4A8 GEMM with 1xN bias and clamp (D = f16 output)
///
/// This is a rvv (4 x vl tile) implementation where vl = csrr_vlenb()
/// (i8 elements per vector at e8/m1).
/// Computes:
///   D = clamp(C + dequant(A_qai8dx) * dequant(B_qsi4cxt) + bias, min, max)
/// where:
///   - A is qai8dx: per-row asymmetric int8, [m, k] row-major.
///   - B is qsi4cxt: per-channel symmetric int4, [n, k] transposed (K-contig),
///     stored as [n, k/2] uint8 with two nibbles per byte (lo = k+0, hi = k+1)
///     using ZP=8 unsigned encoding (signed [-8,7] stored as [0,15]).
///   - bias is f16[n] broadcast across rows.
///   - C is f16[m, n] (optional, can be NULL).
///   - D is f16[m, n].
///
/// Algorithm (per output cell):
///   acc_i32 = sum_k(qa[m, k] * qb_signed[n, k])
///   acc_i32 -= zp_a[m] * sum_k(qb_signed[n, k])
///   D[m, n] = clamp(C[m, n] + scale_a[m] * scale_b[n] * acc_i32 + bias[n], min, max)
///
/// k must be a multiple of 2.

size_t tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(void);

size_t tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(void);

/// Gets A offset in bytes (A is i8 row-major [m, k]).
size_t tqt_get_a_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(size_t m_idx, size_t k_idx,
                                                                        size_t lda);

/// Gets B offset in bytes. B is [n, k] int4 stored as [n, k/2] uint8.
/// ldb is in int4 elements (= k). k_idx must be a multiple of 2.
size_t tqt_get_b_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(size_t n_idx, size_t k_idx,
                                                                        size_t ldb);

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                        size_t ldc);

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(size_t n_idx);

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                        size_t ldd);

size_t tqt_get_d_size_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(size_t m, size_t n);

/// Runs the W4A8 GEMM microkernel.
///
/// @param[in]  m         Number of rows in A and D.
/// @param[in]  n         Number of columns in D.
/// @param[in]  k         Common dimension. Must be a multiple of 2.
/// @param[in]  A         Input matrix A [m, k] qai8dx (int8) row-major.
/// @param[in]  lda       Leading dimension of A (in i8 elements).
/// @param[in]  k_idx_a   K index in A buffer (reserved for K-tiling).
/// @param[in]  B         Input matrix B [n, k] qsi4cxt (int4 transposed),
///                        stored as [n, k/2] uint8.
/// @param[in]  ldb       Leading dimension of B (in int4 elements, = k).
/// @param[in]  k_idx_b   K index in B buffer (reserved for K-tiling).
/// @param[in]  C         Input matrix C [m, n] f16 (optional, can be NULL).
/// @param[in]  ldc       Leading dimension of C (in f16 elements).
/// @param[out] D         Output matrix D [m, n] f16.
/// @param[in]  ldd       Leading dimension of D (in f16 elements).
/// @param[in]  bias      Bias vector [1, n] f16 (optional, can be NULL).
/// @param[in]  clamp_min Minimum clamp value.
/// @param[in]  clamp_max Maximum clamp value.
/// @param[in]  params    Quantization parameters (scale_a[m], zp_a[m], scale_b[n]).
void tqt_run_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
    size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
    float16_t clamp_min, float16_t clamp_max, const struct tqt_qai8dx_qsi4cx_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
