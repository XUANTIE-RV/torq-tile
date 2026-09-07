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

/// Micro-kernel: GEMM with 1xN bias and clamp for FP16 with packed A and B
/// (fp32 widening accumulator)
///
/// This is a rvv (8 x vl tile) implementation where vl = csrr_vlenb() / sizeof(float16_t).
/// Computes: D = clamp(C + A_packed*B_packed + bias, min, max)
/// where A and B are pre-packed for optimal memory access.
/// Internally accumulates in FP32 using vfwmacc.vf, then narrows back to FP16; inputs and
/// output stay f16, so the public contract matches the default same-type-accumulator variant.

/// --------------------------------------------------

/// Gets m step value.
///
/// The starting row index must be divisible by `m_step`.
///
/// @return The m step value (8).
size_t tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(void);

/// Gets n step value.
///
/// The starting column index must be divisible by `n_step`.
///
/// @return The n step value (vl, where vl = csrr_vlenb() / sizeof(float16_t)).
size_t tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(void);

/// Gets the offset in bytes to the data element in the packed A matrix buffer.
size_t tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(size_t m_idx,
                                                                                 size_t k_idx,
                                                                                 size_t lda,
                                                                                 size_t actual_m);

/// Gets the offset in bytes to the data element in the packed B matrix buffer.
size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(size_t n_idx,
                                                                                 size_t k_idx,
                                                                                 size_t ldb,
                                                                                 size_t actual_n);

/// Gets the offset in bytes to the data element in the C matrix buffer.
size_t tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(size_t m_idx,
                                                                          size_t n_idx, size_t ldc);

/// Gets the offset in bytes to the data element in the bias vector buffer.
size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(size_t n_idx);

/// Gets the offset in bytes to the data element in the D (output) matrix buffer.
size_t tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(size_t m_idx,
                                                                          size_t n_idx, size_t ldd);

/// Gets the size in bytes of the D (output) matrix buffer.
size_t tqt_get_d_size_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(size_t m, size_t n);

/// Gets the size in bytes of the packed A matrix buffer.
size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(size_t m, size_t k);

/// Gets the size in bytes of the packed B matrix buffer.
size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(size_t n, size_t k);

/// Packs matrix A for optimal memory access.
void tqt_run_a_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(size_t m, size_t k,
                                                                      size_t lda, size_t lda_packed,
                                                                      size_t k_idx, const void *A,
                                                                      void *A_packed);

/// Packs matrix B for optimal memory access.
void tqt_run_b_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(size_t n, size_t k,
                                                                      size_t ldb, size_t ldb_packed,
                                                                      size_t k_idx, const void *B,
                                                                      void *B_packed);

/// Packs transposed matrix B for optimal memory access.
void tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed);

/// Runs the GEMM microkernel with packed matrices, bias addition and clamp operation.
///
/// Computes: D = clamp(C + A_packed*B_packed + bias, clamp_min, clamp_max)
///
/// @note All matrices use float16_t elements.
/// @note A_packed and B_packed are pre-packed using the pack functions.
/// @note The bias vector has shape [1, n] and is added to each row of the result.
/// @note k_idx_a and k_idx_b are used when A and B are sub-buffers with different sizes.
///
void tqt_run_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float16_t clamp_min, float16_t clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
