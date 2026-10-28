//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Bias is an optional FP32 row vector of N elements; bias offsets count bytes.
// C, D and clamp bounds retain the output type. Bias is added before narrowing.

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
///   D = clamp(C + dequant(A_qsi8d2) * dequant(B_qai4c2t) + bias, min, max)
/// where:
///   - A is qsi8d2: per-row block-quantized symmetric int8 along K, scale_a [m, k/bl] fp32.
///   - B is qai4c2t: per-block ASYMMETRIC int4 along K, stored as [n, k/2] uint8 (ZP=0).
///     scale_b [n, k/bl] fp32, offset_b [n, k/bl] fp32.
///   - bias is f32[n], C is f16[m, n], D is f16[m, n].
///
/// k must be a multiple of bl. bl must be >= 2 and a multiple of 2.

size_t tqt_get_m_step_gemm_1xnbiasf32_clamp_f16_qsi8d2_qai4c2t_4x1vl_rvv(void);

size_t tqt_get_n_step_gemm_1xnbiasf32_clamp_f16_qsi8d2_qai4c2t_4x1vl_rvv(void);

size_t tqt_get_a_offset_gemm_1xnbiasf32_clamp_f16_qsi8d2_qai4c2t_4x1vl_rvv(size_t m_idx,
                                                                           size_t k_idx, size_t lda,
                                                                           size_t bl);

size_t tqt_get_b_offset_gemm_1xnbiasf32_clamp_f16_qsi8d2_qai4c2t_4x1vl_rvv(size_t n_idx,
                                                                           size_t k_idx, size_t ldb,
                                                                           size_t bl);

size_t tqt_get_scale_a_offset_gemm_1xnbiasf32_clamp_f16_qsi8d2_qai4c2t_4x1vl_rvv(size_t m_idx,
                                                                                 size_t b_idx,
                                                                                 size_t ldsa);

size_t tqt_get_scale_b_offset_gemm_1xnbiasf32_clamp_f16_qsi8d2_qai4c2t_4x1vl_rvv(size_t n_idx,
                                                                                 size_t b_idx,
                                                                                 size_t ldsb);

size_t tqt_get_offset_b_offset_gemm_1xnbiasf32_clamp_f16_qsi8d2_qai4c2t_4x1vl_rvv(size_t n_idx,
                                                                                  size_t b_idx,
                                                                                  size_t ldob);

size_t tqt_get_c_offset_gemm_1xnbiasf32_clamp_f16_qsi8d2_qai4c2t_4x1vl_rvv(size_t m_idx,
                                                                           size_t n_idx,
                                                                           size_t ldc);

size_t tqt_get_bias_offset_gemm_1xnbiasf32_clamp_f16_qsi8d2_qai4c2t_4x1vl_rvv(size_t n_idx);

size_t tqt_get_d_offset_gemm_1xnbiasf32_clamp_f16_qsi8d2_qai4c2t_4x1vl_rvv(size_t m_idx,
                                                                           size_t n_idx,
                                                                           size_t ldd);

size_t tqt_get_d_size_gemm_1xnbiasf32_clamp_f16_qsi8d2_qai4c2t_4x1vl_rvv(size_t m, size_t n);

/// @note If clamp_min is negative infinity and clamp_max is positive infinity,
///       clamping is skipped. All other bounds retain the existing clamp behavior.
///       This does not guarantee preservation of NaN payloads or exception flags.
/// F32 bias is added before the final output conversion; bias offsets count bytes.
void tqt_run_gemm_1xnbiasf32_clamp_f16_qsi8d2_qai4c2t_4x1vl_rvv(
    size_t m, size_t n, size_t k, size_t bl, const void *A, size_t lda, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
    const void *bias, float16_t clamp_min, float16_t clamp_max,
    const struct tqt_qsi8d2_qai4c2_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
