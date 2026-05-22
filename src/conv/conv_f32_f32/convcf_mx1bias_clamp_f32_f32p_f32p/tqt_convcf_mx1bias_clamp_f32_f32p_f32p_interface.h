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

// Conv micro-kernel type: convcf_mx1bias_clamp_f32_f32p_f32p
// Channel-first convolution (NCDHW), packed, Mx1 bias.
// A = weight [OC_per_group, IC_per_group * KD * KH * KW] (needs packing, no im2col)
// B = im2col(activation) [IC_per_group * KD * KH * KW, OD * OH * OW] (needs im2col + packing)

/// Helper functions ("get" methods)
typedef size_t (*tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_m_step_func_t)(void);
typedef size_t (*tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_n_step_func_t)(void);
typedef size_t (*tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_a_packed_offset_func_t)(
    size_t m_idx, size_t k_idx, size_t lda, size_t actual_m);
typedef size_t (*tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_b_im2colpack_offset_func_t)(
    size_t n_idx, size_t k_idx, size_t ldb, size_t actual_n);
typedef size_t (*tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_c_offset_func_t)(size_t m_idx,
                                                                             size_t n_idx,
                                                                             size_t ldc);
typedef size_t (*tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_bias_offset_func_t)(size_t m_idx);
typedef size_t (*tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_d_offset_func_t)(size_t m_idx,
                                                                             size_t n_idx,
                                                                             size_t ldd);
typedef size_t (*tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_d_size_func_t)(size_t m, size_t n);
typedef size_t (*tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_a_packed_size_func_t)(size_t m,
                                                                                  size_t k);
typedef size_t (*tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_b_im2colpack_size_func_t)(size_t n,
                                                                                      size_t k);

/// Pack functions
typedef void (*tqt_convcf_mx1bias_clamp_f32_f32p_f32p_run_a_pack_func_t)(
    size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx, const void *A, void *A_packed);

/// Fused im2col+pack for B (activation)
typedef void (*tqt_convcf_mx1bias_clamp_f32_f32p_f32p_run_b_im2colpack_func_t)(
    size_t group_idx, const void *activation, size_t ldb_packed, size_t k_idx, void *B_packed,
    const tqt_conv_params *params);

/// Core function
typedef void (*tqt_convcf_mx1bias_clamp_f32_f32p_f32p_run_conv_func_t)(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float clamp_min, float clamp_max);

/// Micro-kernel interface
struct tqt_convcf_mx1bias_clamp_f32_f32p_f32p_ukernel
{
    tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_m_step_func_t get_m_step;
    tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_n_step_func_t get_n_step;
    tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_a_packed_offset_func_t get_a_packed_offset;
    tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_b_im2colpack_offset_func_t get_b_im2colpack_offset;
    tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_c_offset_func_t get_c_offset;
    tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_bias_offset_func_t get_bias_offset;
    tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_d_offset_func_t get_d_offset;
    tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_d_size_func_t get_d_size;
    tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_a_packed_size_func_t get_a_packed_size;
    tqt_convcf_mx1bias_clamp_f32_f32p_f32p_get_b_im2colpack_size_func_t get_b_im2colpack_size;
    tqt_convcf_mx1bias_clamp_f32_f32p_f32p_run_a_pack_func_t run_a_pack;
    tqt_convcf_mx1bias_clamp_f32_f32p_f32p_run_b_im2colpack_func_t run_b_im2colpack;
    tqt_convcf_mx1bias_clamp_f32_f32p_f32p_run_conv_func_t run_conv;
};

#ifdef __cplusplus
}  // extern "C"
#endif
