//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

#ifdef __cplusplus
extern "C" {
#endif

// Implicit GEMM micro-kernel type: igemmcf_mx1bias_clamp_f16_f16_f16
// Channel-first implicit GEMM convolution (NCDHW), non-packed weight, Mx1 bias.
//
// Fuses im2col + GEMM into a single pass: input elements are fetched on-the-fly
// from the original input tensor using conv_params, eliminating the im2col buffer.
//
// Vectorization direction: M (OC) is vectorized, N (spatial) is scalar.
//   A = weight [OC_per_group, K] (row-major, K = IC_per_group * KD * KH * KW)
//   B = implicit im2col of input [K, N] (N = OD * OH * OW, NOT materialized)
//
// K-dimension alignment requirement:
//   k_idx_a and k_idx_b MUST be multiples of (KD * KH * KW).
//   k MUST be a multiple of (KD * KH * KW), except for the last chunk.
//   This ensures K tiling aligns to input channel (IC) boundaries.

/// Helper functions ("get" methods)
typedef size_t (*tqt_igemmcf_mx1bias_clamp_f16_f16_f16_get_m_step_func_t)(void);
typedef size_t (*tqt_igemmcf_mx1bias_clamp_f16_f16_f16_get_n_step_func_t)(void);
typedef size_t (*tqt_igemmcf_mx1bias_clamp_f16_f16_f16_get_a_offset_func_t)(size_t m_idx,
                                                                            size_t k_idx,
                                                                            size_t lda);
typedef size_t (*tqt_igemmcf_mx1bias_clamp_f16_f16_f16_get_c_offset_func_t)(size_t m_idx,
                                                                            size_t n_idx,
                                                                            size_t ldc);
typedef size_t (*tqt_igemmcf_mx1bias_clamp_f16_f16_f16_get_bias_offset_func_t)(size_t m_idx);
typedef size_t (*tqt_igemmcf_mx1bias_clamp_f16_f16_f16_get_d_offset_func_t)(size_t m_idx,
                                                                            size_t n_idx,
                                                                            size_t ldd);
typedef size_t (*tqt_igemmcf_mx1bias_clamp_f16_f16_f16_get_d_size_func_t)(size_t m, size_t n);

/// Core function: fused implicit-im2col + GEMM convolution micro-kernel.
///
/// Computes a region of the convolution output:
///   D[0:m, n_idx:n_idx+n] += A[0:m, k_idx_a:k_idx_a+k]
///                           x B_implicit[k_idx_b:k_idx_b+k, n_idx:n_idx+n]
///
/// M and N tiling is handled internally (tiles M by m_step, N by n_step).
/// The caller handles K tiling and group iteration.
///
/// K-tiling behavior:
///   - k_idx_b == 0 (first K chunk): accumulators initialized from bias (or zero).
///   - k_idx_b > 0 (subsequent K chunks): accumulators initialized from C.
///   - k_idx_b + k == K_total (last K chunk): clamp applied before storing to D.
///   - k_idx_b + k < K_total (not last chunk): partial sums stored to D without clamp.
typedef void (*tqt_igemmcf_mx1bias_clamp_f16_f16_f16_run_igemm_func_t)(
    size_t m, size_t n, size_t k, const void *input, size_t group_idx, size_t n_idx, size_t k_idx_b,
    const void *A, size_t lda, size_t k_idx_a, const void *C, size_t ldc, void *D, size_t ldd,
    const void *bias, float16_t clamp_min, float16_t clamp_max, const tqt_conv_params *params);

/// Micro-kernel interface
struct tqt_igemmcf_mx1bias_clamp_f16_f16_f16_ukernel
{
    tqt_igemmcf_mx1bias_clamp_f16_f16_f16_get_m_step_func_t get_m_step;
    tqt_igemmcf_mx1bias_clamp_f16_f16_f16_get_n_step_func_t get_n_step;
    tqt_igemmcf_mx1bias_clamp_f16_f16_f16_get_a_offset_func_t get_a_offset;
    tqt_igemmcf_mx1bias_clamp_f16_f16_f16_get_c_offset_func_t get_c_offset;
    tqt_igemmcf_mx1bias_clamp_f16_f16_f16_get_bias_offset_func_t get_bias_offset;
    tqt_igemmcf_mx1bias_clamp_f16_f16_f16_get_d_offset_func_t get_d_offset;
    tqt_igemmcf_mx1bias_clamp_f16_f16_f16_get_d_size_func_t get_d_size;
    tqt_igemmcf_mx1bias_clamp_f16_f16_f16_run_igemm_func_t run_igemm;
};

#ifdef __cplusplus
}  // extern "C"
#endif
