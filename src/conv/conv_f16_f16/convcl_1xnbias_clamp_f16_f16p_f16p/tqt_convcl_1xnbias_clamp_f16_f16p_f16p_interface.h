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

// Conv micro-kernel type: convcl_1xnbias_clamp_f16_f16p_f16p
// Channel-last convolution (NDHWC), packed, 1xN bias.
// A = im2col(activation) [OD * OH * OW, KD * KH * KW * IC_per_group] (needs im2col + packing)
// B = weight [OC_per_group, KD * KH * KW * IC_per_group] (transposed, needs packing, no im2col)

/// Helper functions ("get" methods)
typedef size_t (*tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_m_step_func_t)(void);
typedef size_t (*tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_n_step_func_t)(void);
typedef size_t (*tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_a_im2colpack_offset_func_t)(
    size_t m_idx, size_t k_idx, size_t lda, size_t actual_m);
typedef size_t (*tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_b_packed_offset_func_t)(
    size_t n_idx, size_t k_idx, size_t ldb, size_t actual_n);
typedef size_t (*tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_c_offset_func_t)(size_t m_idx,
                                                                             size_t n_idx,
                                                                             size_t ldc);
typedef size_t (*tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_bias_offset_func_t)(size_t n_idx);
typedef size_t (*tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_d_offset_func_t)(size_t m_idx,
                                                                             size_t n_idx,
                                                                             size_t ldd);
typedef size_t (*tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_d_size_func_t)(size_t m, size_t n);
typedef size_t (*tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_a_im2colpack_size_func_t)(size_t m,
                                                                                      size_t k);
typedef size_t (*tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_b_packed_size_func_t)(size_t n,
                                                                                  size_t k);

/// Pack functions
typedef void (*tqt_convcl_1xnbias_clamp_f16_f16p_f16p_run_bt_pack_func_t)(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed);

/// Fused im2col+pack for A (activation)
typedef void (*tqt_convcl_1xnbias_clamp_f16_f16p_f16p_run_a_im2colpack_func_t)(
    size_t group_idx, const void *activation, size_t lda_packed, size_t k_idx, void *A_packed,
    const tqt_conv_params *params);

/// Core function
typedef void (*tqt_convcl_1xnbias_clamp_f16_f16p_f16p_run_conv_func_t)(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float16_t clamp_min, float16_t clamp_max);

/// Micro-kernel interface
struct tqt_convcl_1xnbias_clamp_f16_f16p_f16p_ukernel
{
    tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_m_step_func_t get_m_step;
    tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_n_step_func_t get_n_step;
    tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_a_im2colpack_offset_func_t get_a_im2colpack_offset;
    tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_b_packed_offset_func_t get_b_packed_offset;
    tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_c_offset_func_t get_c_offset;
    tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_bias_offset_func_t get_bias_offset;
    tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_d_offset_func_t get_d_offset;
    tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_d_size_func_t get_d_size;
    tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_a_im2colpack_size_func_t get_a_im2colpack_size;
    tqt_convcl_1xnbias_clamp_f16_f16p_f16p_get_b_packed_size_func_t get_b_packed_size;
    tqt_convcl_1xnbias_clamp_f16_f16p_f16p_run_bt_pack_func_t run_bt_pack;
    tqt_convcl_1xnbias_clamp_f16_f16p_f16p_run_a_im2colpack_func_t run_a_im2colpack;
    tqt_convcl_1xnbias_clamp_f16_f16p_f16p_run_conv_func_t run_conv;
};

#ifdef __cplusplus
}  // extern "C"
#endif
