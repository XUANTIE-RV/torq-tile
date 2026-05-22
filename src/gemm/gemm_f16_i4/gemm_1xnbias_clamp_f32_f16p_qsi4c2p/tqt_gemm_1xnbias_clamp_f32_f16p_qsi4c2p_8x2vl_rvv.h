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

/// Micro-kernel: W4A16 GEMM with 1xN bias and clamp, fp32 output (packed A and B).
///
/// This is a rvv (8 x 2*vl tile) implementation where vl = csrr_vlenb() / sizeof(float16_t).
/// Computes: D = clamp(C + A_packed * dequant(B_packed, scale_b) + bias, min, max)
/// where A is pre-packed [m/8, k, 8] f16, B is pre-packed [n/(2*vl_e8), k/2, 2*vl_e8] int4,
/// scale_b is [n, k/bl] f16, bias is [1, n] f32 broadcast across rows,
/// C is [m, n] f32 (optional), and D is [m, n] f32. Internal accumulation
/// is performed in f16; results are widened to f32 only at the store step.

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(void);

size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(void);

/// Gets the offset in bytes to the data element in the packed A matrix buffer.
size_t tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(size_t m_idx,
                                                                             size_t k_idx,
                                                                             size_t lda,
                                                                             size_t actual_m);

/// Gets the offset in bytes to the data element in the packed B matrix buffer.
/// B is [n, k] int4 packed as [n/(2*vl_e8), k/2, 2*vl_e8] uint8.
size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(size_t n_idx,
                                                                             size_t k_idx,
                                                                             size_t ldb,
                                                                             size_t actual_n);

/// Gets scale_b offset in bytes. b_idx = block index = k_idx / bl.
size_t tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(size_t n_idx,
                                                                            size_t b_idx,
                                                                            size_t ldsb);

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(size_t m_idx, size_t n_idx,
                                                                      size_t ldc);

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(size_t n_idx);

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(size_t m_idx, size_t n_idx,
                                                                      size_t ldd);

size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(size_t m, size_t n);

/// Gets the size in bytes of the packed A matrix buffer.
size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(size_t m, size_t k);

/// Gets the size in bytes of the packed B matrix buffer.
size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(size_t n, size_t k);

/// Packs matrix A for optimal memory access.
/// A is [m, k] f16 row-major -> packed [m/8, k, 8] f16.
void tqt_run_a_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(size_t m, size_t k, size_t lda,
                                                                  size_t lda_packed, size_t k_idx,
                                                                  const void *A, void *A_packed);

/// Packs transposed matrix B for optimal memory access.
/// B is [n, k] int4 (stored as [n, k/2] uint8) -> packed layout.
void tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(size_t n, size_t k, size_t ldb,
                                                                   size_t ldb_packed, size_t k_idx,
                                                                   const void *B, void *B_packed);

/// Runs the W4A16 GEMM microkernel with packed matrices and fp32 output.
///
/// Computes: D = clamp(C + A_packed * dequant(B_packed, scale_b) + bias, clamp_min, clamp_max)
///
/// @param[in]  m         Number of rows in A and D.
/// @param[in]  n         Number of columns in D.
/// @param[in]  k         Common dimension. Must be a multiple of bl.
/// @param[in]  bl        Block length for scale quantization. Must be >= 2 and a multiple of 2.
/// @param[in]  A_packed  Packed input matrix A.
/// @param[in]  lda       Leading dimension of original A (in f16 elements).
/// @param[in]  k_idx_a   K index in A buffer (for sub-buffer scenarios).
/// @param[in]  B_packed  Packed input matrix B.
/// @param[in]  ldb       Leading dimension of original B (in int4 elements, = k).
/// @param[in]  k_idx_b   K index in B buffer (for sub-buffer scenarios).
/// @param[in]  scale_b   Scale matrix [n, k/bl] f16 row-major.
/// @param[in]  ldsb      Leading dimension of scale_b (in f16 elements, = k/bl).
/// @param[in]  C         Input matrix C [m, n] f32 (optional, can be NULL).
/// @param[in]  ldc       Leading dimension of C (in f32 elements).
/// @param[out] D         Output matrix D [m, n] f32.
/// @param[in]  ldd       Leading dimension of D (in f32 elements).
/// @param[in]  bias      Bias vector [1, n] f32 (optional, can be NULL).
/// @param[in]  clamp_min Minimum clamp value (f32).
/// @param[in]  clamp_max Maximum clamp value (f32).
void tqt_run_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv(
    size_t m, size_t n, size_t k, size_t bl, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *scale_b, size_t ldsb,
    const void *C, size_t ldc, void *D, size_t ldd, const void *bias, float clamp_min,
    float clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
