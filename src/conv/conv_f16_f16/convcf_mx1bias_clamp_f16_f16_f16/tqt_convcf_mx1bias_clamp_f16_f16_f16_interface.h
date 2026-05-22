//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

#ifdef __cplusplus
extern "C" {
#endif

// Conv micro-kernel type: convcf_mx1bias_clamp_f16_f16_f16
// Channel-first convolution (NCDHW), non-packed, Mx1 bias.
// A = weight [OC_per_group, IC_per_group * KD * KH * KW] (row-major, no im2col needed)
// B = im2col(activation) [IC_per_group * KD * KH * KW, OD * OH * OW] (row-major)

/// Helper functions ("get" methods)
typedef size_t (*tqt_convcf_mx1bias_clamp_f16_f16_f16_get_m_step_func_t)(void);
typedef size_t (*tqt_convcf_mx1bias_clamp_f16_f16_f16_get_n_step_func_t)(void);
typedef size_t (*tqt_convcf_mx1bias_clamp_f16_f16_f16_get_a_offset_func_t)(size_t m_idx,
                                                                           size_t k_idx,
                                                                           size_t lda);
typedef size_t (*tqt_convcf_mx1bias_clamp_f16_f16_f16_get_c_offset_func_t)(size_t m_idx,
                                                                           size_t n_idx,
                                                                           size_t ldc);
typedef size_t (*tqt_convcf_mx1bias_clamp_f16_f16_f16_get_bias_offset_func_t)(size_t m_idx);
typedef size_t (*tqt_convcf_mx1bias_clamp_f16_f16_f16_get_d_offset_func_t)(size_t m_idx,
                                                                           size_t n_idx,
                                                                           size_t ldd);
typedef size_t (*tqt_convcf_mx1bias_clamp_f16_f16_f16_get_d_size_func_t)(size_t m, size_t n);

/// Conv im2col functions (B = activation needs im2col)
typedef size_t (*tqt_convcf_mx1bias_clamp_f16_f16_f16_get_b_im2col_size_func_t)(size_t k, size_t n);
typedef size_t (*tqt_convcf_mx1bias_clamp_f16_f16_f16_get_b_im2col_offset_func_t)(
    size_t n_idx, size_t k_idx, size_t ldb_im2col);
typedef void (*tqt_convcf_mx1bias_clamp_f16_f16_f16_run_im2col_func_t)(
    size_t group_idx, const void *activation, void *im2col_buf, const tqt_conv_params *params);

/// Core function
typedef void (*tqt_convcf_mx1bias_clamp_f16_f16_f16_run_conv_func_t)(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
    size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
    float16_t clamp_min, float16_t clamp_max);

/// Micro-kernel interface
struct tqt_convcf_mx1bias_clamp_f16_f16_f16_ukernel
{
    tqt_convcf_mx1bias_clamp_f16_f16_f16_get_m_step_func_t get_m_step;
    tqt_convcf_mx1bias_clamp_f16_f16_f16_get_n_step_func_t get_n_step;
    tqt_convcf_mx1bias_clamp_f16_f16_f16_get_a_offset_func_t get_a_offset;
    tqt_convcf_mx1bias_clamp_f16_f16_f16_get_c_offset_func_t get_c_offset;
    tqt_convcf_mx1bias_clamp_f16_f16_f16_get_bias_offset_func_t get_bias_offset;
    tqt_convcf_mx1bias_clamp_f16_f16_f16_get_d_offset_func_t get_d_offset;
    tqt_convcf_mx1bias_clamp_f16_f16_f16_get_d_size_func_t get_d_size;
    tqt_convcf_mx1bias_clamp_f16_f16_f16_get_b_im2col_size_func_t get_b_im2col_size;
    tqt_convcf_mx1bias_clamp_f16_f16_f16_get_b_im2col_offset_func_t get_b_im2col_offset;
    tqt_convcf_mx1bias_clamp_f16_f16_f16_run_im2col_func_t run_im2col;
    tqt_convcf_mx1bias_clamp_f16_f16_f16_run_conv_func_t run_conv;
};

#ifdef __cplusplus
}  // extern "C"
#endif
