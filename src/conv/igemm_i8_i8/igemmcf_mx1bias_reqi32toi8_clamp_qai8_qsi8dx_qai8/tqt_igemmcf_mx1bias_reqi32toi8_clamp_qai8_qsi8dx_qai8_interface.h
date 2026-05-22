//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "src/common/tqt_conv.h"
#include "src/gemm/gemm_i8_i8/tqt_params_i8_i8.h"

#ifdef __cplusplus
extern "C" {
#endif

// Implicit GEMM micro-kernel type: igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8
// Channel-first implicit GEMM convolution (NCDHW), non-packed weight, Mx1 bias,
// requantization (int32->int8), and clamp.
//
// Fuses im2col + GEMM: input elements fetched on-the-fly from the original input tensor.
//
// Vectorization direction: M (OC) is vectorized, N (spatial) is scalar.
//   A = weight [OC_per_group, K] row-major (K = IC_per_group * KD * KH * KW)
//   B = implicit im2col of input [K, N] (N = OD * OH * OW, NOT materialized)
//
// Quantization scheme:
//   A (weight): symmetric per-channel quantization (scale_a[m], quant_channel_a = OC)
//   B (input):  asymmetric quantization (scale_b, zero_point_b)
//   D (output): asymmetric quantization (scale_d, zero_point_d)
//
// NO K-tiling: full K is processed in a single call (k_idx params are unused).

/// Helper functions ("get" methods)
typedef size_t (*tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_get_m_step_func_t)(void);
typedef size_t (*tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_get_n_step_func_t)(void);
typedef size_t (*tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_get_a_offset_func_t)(
    size_t m_idx, size_t k_idx, size_t lda);
typedef size_t (*tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_get_c_offset_func_t)(
    size_t m_idx, size_t n_idx, size_t ldc);
typedef size_t (*tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_get_bias_offset_func_t)(
    size_t m_idx);
typedef size_t (*tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_get_d_offset_func_t)(
    size_t m_idx, size_t n_idx, size_t ldd);
typedef size_t (*tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_get_d_size_func_t)(size_t m,
                                                                                          size_t n);

/// Core function: fused implicit-im2col + GEMM convolution micro-kernel with requantization.
///
/// M and N tiling is handled internally.
/// NO K-tiling: full K is always processed in one call.
typedef void (*tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_run_igemm_func_t)(
    size_t m, size_t n, size_t k, const void *input, size_t group_idx, size_t n_idx, size_t k_idx_b,
    const void *A, size_t lda, size_t k_idx_a, const void *C, size_t ldc, void *D, size_t ldd,
    const void *bias, int8_t clamp_min, int8_t clamp_max, const tqt_conv_params *conv_params,
    const struct tqt_qai8_qsi8dx_qai8_params *quant_params);

/// Micro-kernel interface
struct tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_ukernel
{
    tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_get_m_step_func_t get_m_step;
    tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_get_n_step_func_t get_n_step;
    tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_get_a_offset_func_t get_a_offset;
    tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_get_c_offset_func_t get_c_offset;
    tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_get_bias_offset_func_t get_bias_offset;
    tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_get_d_offset_func_t get_d_offset;
    tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_get_d_size_func_t get_d_size;
    tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_run_igemm_func_t run_igemm;
};

#ifdef __cplusplus
}  // extern "C"
#endif
