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

/// Micro-kernel: W4A8 GEMM with 1xN bias and clamp (D = f32, packed A and B)
///
/// This is a rvv (4 x vl tile) implementation where vl = csrr_vlenb()
/// (i8 elements per vector at e8/m1).
/// Computes:
///   D = clamp(C + dequant(A_packed) * dequant(B_packed) + bias, min, max)
///
/// Layouts:
///   - A is qai8dxp: pre-packed [m/m_step, k, m_step] int8 (interleaved).
///   - B is qsi4cxp: pre-packed [n/n_step, k/2, n_step] uint8 (int4, ZP=8).
///   - bias is f32[n] broadcast across rows.
///   - C is f32[m, n] (optional).
///   - D is f32[m, n].

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv(void);

size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv(void);

size_t tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv(size_t m_idx,
                                                                                size_t k_idx,
                                                                                size_t lda,
                                                                                size_t actual_m);

size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv(size_t n_idx,
                                                                                size_t k_idx,
                                                                                size_t ldb,
                                                                                size_t actual_n);

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                         size_t ldc);

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv(size_t n_idx);

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                         size_t ldd);

size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv(size_t m, size_t n);

size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv(size_t m, size_t k);

size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv(size_t n, size_t k);

/// Packs A matrix [m, k] int8 row-major → [m/m_step, k, m_step] int8 interleaved.
void tqt_run_a_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv(size_t m, size_t k, size_t lda,
                                                                     size_t lda_packed,
                                                                     size_t k_idx, const void *A,
                                                                     void *A_packed);

/// Packs transposed B matrix [n, k] int4 (stored [n, k/2] uint8) →
/// [n/n_step, k/2, n_step] uint8 packed layout.
void tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv(size_t n, size_t k,
                                                                      size_t ldb, size_t ldb_packed,
                                                                      size_t k_idx, const void *B,
                                                                      void *B_packed);

/// Runs the packed W4A8 GEMM microkernel.
///
/// @param[in]  m         Number of rows in A and D.
/// @param[in]  n         Number of columns in D.
/// @param[in]  k         Common dimension. Must be a multiple of 2.
/// @param[in]  A_packed  Pre-packed A buffer (see tqt_run_a_pack_*).
/// @param[in]  lda       Original A leading dimension (in i8 elements).
/// @param[in]  k_idx_a   K index in A_packed buffer (reserved for K-tiling).
/// @param[in]  B_packed  Pre-packed B buffer (see tqt_run_bt_pack_*).
/// @param[in]  ldb       Original B leading dimension (in int4 elements, = k).
/// @param[in]  k_idx_b   K index in B_packed buffer (reserved for K-tiling).
/// @param[in]  C         Input matrix C [m, n] f32 (optional, can be NULL).
/// @param[in]  ldc       Leading dimension of C (in f32 elements).
/// @param[out] D         Output matrix D [m, n] f32.
/// @param[in]  ldd       Leading dimension of D (in f32 elements).
/// @param[in]  bias      Bias vector [1, n] f32 (optional, can be NULL).
/// @param[in]  clamp_min Minimum clamp value.
/// @param[in]  clamp_max Maximum clamp value.
/// @param[in]  params    Quantization parameters (scale_a[m], zp_a[m], scale_b[n]).
void tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float clamp_min, float clamp_max,
    const struct tqt_qai8dx_qsi4cx_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
