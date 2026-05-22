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

/// Micro-kernel: W4A8 GEMM with 1xN bias and clamp (D = bf16, packed A and B)
///
/// This is a rvv (4 x vl tile) implementation where vl = csrr_vlenb()
/// (i8 elements per vector at e8/m1).
/// Computes:
///   D = clamp(C + dequant(A_packed) * dequant(B_packed) + bias, min, max)
///
/// Layouts:
///   - A is qai8dxp: pre-packed [m/m_step, k, m_step] int8 (interleaved).
///   - B is qsi4cxp: pre-packed [n/n_step, k/2, n_step] uint8 (int4, ZP=8).
///   - bias is bf16[n] broadcast across rows.
///   - C is bf16[m, n] (optional).
///   - D is bf16[m, n].

size_t tqt_get_m_step_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv(void);

size_t tqt_get_n_step_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv(void);

size_t tqt_get_a_packed_offset_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv(size_t m_idx,
                                                                                 size_t k_idx,
                                                                                 size_t lda,
                                                                                 size_t actual_m);

size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv(size_t n_idx,
                                                                                 size_t k_idx,
                                                                                 size_t ldb,
                                                                                 size_t actual_n);

size_t tqt_get_c_offset_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv(size_t m_idx,
                                                                          size_t n_idx, size_t ldc);

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv(size_t n_idx);

size_t tqt_get_d_offset_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv(size_t m_idx,
                                                                          size_t n_idx, size_t ldd);

size_t tqt_get_d_size_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv(size_t m, size_t n);

size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv(size_t m, size_t k);

size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv(size_t n, size_t k);

/// Packs A matrix [m, k] int8 row-major → [m/m_step, k, m_step] int8 interleaved.
void tqt_run_a_pack_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv(size_t m, size_t k,
                                                                      size_t lda, size_t lda_packed,
                                                                      size_t k_idx, const void *A,
                                                                      void *A_packed);

/// Packs transposed B matrix [n, k] int4 (stored [n, k/2] uint8) →
/// [n/n_step, k/2, n_step] uint8 packed layout.
void tqt_run_bt_pack_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed);

void tqt_run_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, bfloat16_t clamp_min, bfloat16_t clamp_max,
    const struct tqt_qai8dx_qsi4cx_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
