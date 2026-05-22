//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include "src/common/tqt_conv.h"

#ifdef __cplusplus
extern "C" {
#endif

// Implicit GEMM micro-kernel type: igemmcf_mx1bias_clamp_f32_f32p_f32
// Channel-first implicit GEMM convolution (NCDHW), packed weight, Mx1 bias.
//
// Weight is pre-packed from [OC, K] to [OC/3vl, K, 3vl] for efficient vector loads.
// Input is NOT packed — elements fetched on-the-fly via implicit im2col.
//
// K-dimension alignment requirement:
//   k_idx_a and k_idx_b MUST be multiples of (KD * KH * KW).

/// Helper functions ("get" methods)
typedef size_t (*tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_get_m_step_func_t)(void);
typedef size_t (*tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_get_n_step_func_t)(void);
typedef size_t (*tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_get_a_packed_offset_func_t)(
    size_t m_idx, size_t k_idx, size_t lda, size_t actual_m);
typedef size_t (*tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_get_c_offset_func_t)(size_t m_idx,
                                                                             size_t n_idx,
                                                                             size_t ldc);
typedef size_t (*tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_get_bias_offset_func_t)(size_t m_idx);
typedef size_t (*tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_get_d_offset_func_t)(size_t m_idx,
                                                                             size_t n_idx,
                                                                             size_t ldd);
typedef size_t (*tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_get_d_size_func_t)(size_t m, size_t n);
typedef size_t (*tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_get_a_packed_size_func_t)(size_t m,
                                                                                  size_t k);

/// Weight pack function: [OC, K] -> [OC/3vl, K, 3vl]
typedef void (*tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_run_a_pack_func_t)(
    size_t m, size_t k, size_t ldw, size_t ldw_packed, size_t k_idx, const void *weight,
    void *weight_packed);

/// Core function: fused implicit-im2col + GEMM with packed weight
typedef void (*tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_run_igemm_func_t)(
    size_t m, size_t n, size_t k, const void *input, size_t group_idx, size_t n_idx, size_t k_idx_b,
    const void *A_packed, size_t lda, size_t k_idx_a, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float clamp_min, float clamp_max, const tqt_conv_params *params);

/// Micro-kernel interface
struct tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_ukernel
{
    tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_get_m_step_func_t get_m_step;
    tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_get_n_step_func_t get_n_step;
    tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_get_a_packed_offset_func_t get_a_packed_offset;
    tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_get_c_offset_func_t get_c_offset;
    tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_get_bias_offset_func_t get_bias_offset;
    tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_get_d_offset_func_t get_d_offset;
    tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_get_d_size_func_t get_d_size;
    tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_get_a_packed_size_func_t get_a_packed_size;
    tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_run_a_pack_func_t run_a_pack;
    tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_run_igemm_func_t run_igemm;
};

#ifdef __cplusplus
}  // extern "C"
#endif
