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

/// Micro-kernel: W4A8 block-quantized GEMM with 1xN bias and clamp (D = f16 output)
///
/// rvv (4 x vl tile) implementation where vl = csrr_vlenb()
/// (i8 elements per vector at e8/m1).
/// Computes:
///   D = clamp(C + dequant(A_qai8dx) * dequant(B_qsi4c2t) + bias, min, max)
/// where:
///   - A is qai8dx: per-row asymmetric int8 (no block dimension), [m, k] row-major.
///     scale_a is [m] fp32 (per-row). zero_point_a is [m] int32 (per-row).
///   - B is qsi4c2t: per-block symmetric int4 along K (block size = bl),
///     [n, k] transposed, stored as [n, k/2] uint8 with two nibbles per byte
///     using ZP=8 unsigned encoding. scale_b is [n, k/bl] fp32.
///   - bias is f16[n] broadcast across rows (optional).
///   - C is f16[m, n] (optional, can be nullptr).
///   - D is f16[m, n].
///
/// k must be a multiple of bl. bl must be >= 2 and a multiple of 2.

size_t tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dx_qsi4c2t_4x1vl_rvv(void);

size_t tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dx_qsi4c2t_4x1vl_rvv(void);

size_t tqt_get_a_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi4c2t_4x1vl_rvv(size_t m_idx, size_t k_idx,
                                                                        size_t lda, size_t bl);

size_t tqt_get_b_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi4c2t_4x1vl_rvv(size_t n_idx, size_t k_idx,
                                                                        size_t ldb, size_t bl);

size_t tqt_get_scale_b_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi4c2t_4x1vl_rvv(size_t n_idx,
                                                                              size_t b_idx,
                                                                              size_t ldsb);

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi4c2t_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                        size_t ldc);

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi4c2t_4x1vl_rvv(size_t n_idx);

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi4c2t_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                        size_t ldd);

size_t tqt_get_d_size_gemm_1xnbias_clamp_f16_qai8dx_qsi4c2t_4x1vl_rvv(size_t m, size_t n);

void tqt_run_gemm_1xnbias_clamp_f16_qai8dx_qsi4c2t_4x1vl_rvv(
    size_t m, size_t n, size_t k, size_t bl, const void *A, size_t lda, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
    const void *bias, float16_t clamp_min, float16_t clamp_max,
    const struct tqt_qai8dx_qsi4c2_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
