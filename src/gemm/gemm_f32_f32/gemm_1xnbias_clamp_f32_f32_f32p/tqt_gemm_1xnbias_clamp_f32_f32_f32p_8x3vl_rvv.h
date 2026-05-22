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

/// Micro-kernel: GEMM with 1xN bias and clamp for FP32 with packed B
///
/// This is a rvv (8 x 3*vl tile) implementation where vl = csrr_vlenb() / sizeof(float).
/// Computes: D = clamp(C + A*B_packed + bias, min, max)
/// where A is in original row-major layout and B is pre-packed for optimal memory access.

/// --------------------------------------------------

/// Gets m step value.
size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(void);

/// Gets n step value.
size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(void);

/// Gets the offset in bytes to the data element in the A matrix buffer.
size_t tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(size_t m_idx, size_t k_idx,
                                                                  size_t lda);

/// Gets the offset in bytes to the data element in the packed B matrix buffer.
size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(size_t n_idx, size_t k_idx,
                                                                         size_t ldb,
                                                                         size_t actual_n);

/// Gets the offset in bytes to the data element in the C matrix buffer.
size_t tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                  size_t ldc);

/// Gets the offset in bytes to the data element in the bias vector buffer.
size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(size_t n_idx);

/// Gets the offset in bytes to the data element in the D (output) matrix buffer.
size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                  size_t ldd);

/// Gets the size in bytes of the D (output) matrix buffer.
size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(size_t m, size_t n);

/// Gets the size in bytes of the packed B matrix buffer.
size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(size_t n, size_t k);

/// Packs matrix B (non-transposed, stored as [k, n]) for optimal memory access.
void tqt_run_b_pack_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(size_t n, size_t k, size_t ldb,
                                                              size_t ldb_packed, size_t k_idx,
                                                              const void *B, void *B_packed);

/// Packs transposed matrix B (stored as [n, k]) for optimal memory access.
void tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(size_t n, size_t k, size_t ldb,
                                                               size_t ldb_packed, size_t k_idx,
                                                               const void *B, void *B_packed);

/// Runs the GEMM microkernel with original A and packed B, bias addition and clamp.
void tqt_run_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(size_t m, size_t n, size_t k, const void *A,
                                                       size_t lda, size_t k_idx_a,
                                                       const void *B_packed, size_t ldb,
                                                       size_t k_idx_b, const void *C, size_t ldc,
                                                       void *D, size_t ldd, const void *bias,
                                                       float clamp_min, float clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
