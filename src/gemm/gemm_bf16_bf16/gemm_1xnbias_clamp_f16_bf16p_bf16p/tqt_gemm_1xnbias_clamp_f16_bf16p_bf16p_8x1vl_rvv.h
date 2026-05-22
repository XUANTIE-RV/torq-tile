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

/// Micro-kernel: GEMM with 1xN bias and clamp, BF16 packed inputs and FP16 output.
///
/// This is a rvv (8 x vl tile) implementation where vl = csrr_vlenb() / sizeof(bfloat16_t).
/// Computes: D = clamp(C + A_packed*B_packed + bias, min, max)
/// where A and B are pre-packed for optimal memory access (still bf16 in the packed buffer).
/// Internally accumulates in FP32 using vfwmaccbf16.vf, then narrows to FP16.

/// --------------------------------------------------

size_t tqt_get_m_step_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv(void);
size_t tqt_get_n_step_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv(void);

size_t tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv(size_t m_idx,
                                                                            size_t k_idx,
                                                                            size_t lda,
                                                                            size_t actual_m);
size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv(size_t n_idx,
                                                                            size_t k_idx,
                                                                            size_t ldb,
                                                                            size_t actual_n);
size_t tqt_get_c_offset_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldc);
size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv(size_t n_idx);
size_t tqt_get_d_offset_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldd);
size_t tqt_get_d_size_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv(size_t m, size_t n);
size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv(size_t m, size_t k);
size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv(size_t n, size_t k);

/// Packs matrix A for optimal memory access (bfloat16_t input).
void tqt_run_a_pack_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv(size_t m, size_t k, size_t lda,
                                                                 size_t lda_packed, size_t k_idx,
                                                                 const void *A, void *A_packed);

/// Packs matrix B for optimal memory access (bfloat16_t input, K x N layout).
void tqt_run_b_pack_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv(size_t n, size_t k, size_t ldb,
                                                                 size_t ldb_packed, size_t k_idx,
                                                                 const void *B, void *B_packed);

/// Packs transposed matrix B for optimal memory access (bfloat16_t input, N x K layout).
void tqt_run_bt_pack_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv(size_t n, size_t k, size_t ldb,
                                                                  size_t ldb_packed, size_t k_idx,
                                                                  const void *B, void *B_packed);

/// Runs the GEMM microkernel with packed matrices, bias addition and clamp operation.
///
/// @note A_packed and B_packed contain bfloat16_t data; C, D, bias are float16_t.
///
void tqt_run_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float16_t clamp_min, float16_t clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
