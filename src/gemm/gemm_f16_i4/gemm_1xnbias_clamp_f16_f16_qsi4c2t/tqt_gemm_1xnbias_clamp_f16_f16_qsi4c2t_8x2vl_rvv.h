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

/// Micro-kernel: W4A16 GEMM with 1xN bias and clamp
///
/// This is a rvv (8 x 2*vl tile) implementation where vl = csrr_vlenb() / sizeof(float16_t).
/// Computes: D = clamp(C + A * dequant(B, scale_b) + bias, min, max)
/// where B is [n, k] int4 (transposed, K-contiguous), scale_b is [n, k/bl] f16,
/// and bias has shape [1, N] broadcast across rows.

size_t tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(void);

size_t tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(void);

/// Gets A offset in bytes. k_idx must be a multiple of bl.
size_t tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(size_t m_idx, size_t k_idx,
                                                                     size_t lda, size_t bl);

/// Gets B offset in bytes. B is [n,k] int4, ldb = k (int4 elements). k_idx must be a multiple of
/// bl.
size_t tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(size_t n_idx, size_t k_idx,
                                                                     size_t ldb, size_t bl);

/// Gets scale_b offset in bytes. b_idx = block index = k_idx / bl.
size_t tqt_get_scale_b_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(size_t n_idx,
                                                                           size_t b_idx,
                                                                           size_t ldsb);

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldc);

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(size_t n_idx);

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldd);

size_t tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(size_t m, size_t n);

/// Runs the W4A16 GEMM microkernel.
///
/// Computes: D = clamp(C + A * dequant(B, scale_b) + bias, clamp_min, clamp_max)
///
/// @param[in]  m         Number of rows in A and D.
/// @param[in]  n         Number of columns in D.
/// @param[in]  k         Common dimension. Must be a multiple of bl.
/// @param[in]  bl        Block length for scale quantization. Must be >= 2 and a multiple of 2.
/// @param[in]  A         Input matrix A [m, k] f16 row-major.
/// @param[in]  lda       Leading dimension of A (in f16 elements).
/// @param[in]  k_idx_a   K index in A buffer (reserved for K-tiling).
/// @param[in]  B         Input matrix B [n, k] int4, K-contiguous (transposed).
/// @param[in]  ldb       Leading dimension of B (in int4 elements, = k).
/// @param[in]  k_idx_b   K index in B buffer (reserved for K-tiling).
/// @param[in]  scale_b   Scale matrix [n, k/bl] f16 row-major.
/// @param[in]  ldsb      Leading dimension of scale_b (in f16 elements, = k/bl).
/// @param[in]  C         Input matrix C [m, n] f16 (optional, can be NULL).
/// @param[in]  ldc       Leading dimension of C.
/// @param[out] D         Output matrix D [m, n] f16.
/// @param[in]  ldd       Leading dimension of D.
/// @param[in]  bias      Bias vector [1, n] f16 (optional, can be NULL).
/// @param[in]  clamp_min Minimum clamp value.
/// @param[in]  clamp_max Maximum clamp value.
void tqt_run_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(
    size_t m, size_t n, size_t k, size_t bl, const void *A, size_t lda, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b, const void *scale_b, size_t ldsb, const void *C,
    size_t ldc, void *D, size_t ldd, const void *bias, float16_t clamp_min, float16_t clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
