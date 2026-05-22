//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"
#include "src/gemm/gemm_i8_i8/tqt_params_i8_i8.h"

#ifdef __cplusplus
extern "C" {
#endif

// Conv micro-kernel type: convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt
// Channel-last convolution (NDHWC), non-packed, 1xN bias with requantize.
// A = im2col(activation) [OD * OH * OW, KD * KH * KW * IC_per_group] (row-major, asymmetric)
// B = weight [OC_per_group, KD * KH * KW * IC_per_group] (row-major, transposed, per-channel
//     symmetric, no im2col needed)

/// Helper functions ("get" methods)
typedef size_t (*tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_get_m_step_func_t)(void);
typedef size_t (*tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_get_n_step_func_t)(void);
typedef size_t (*tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_get_b_offset_func_t)(
    size_t n_idx, size_t k_idx, size_t ldb);
typedef size_t (*tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_get_c_offset_func_t)(
    size_t m_idx, size_t n_idx, size_t ldc);
typedef size_t (*tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_get_bias_offset_func_t)(
    size_t n_idx);
typedef size_t (*tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_get_d_offset_func_t)(
    size_t m_idx, size_t n_idx, size_t ldd);
typedef size_t (*tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_get_d_size_func_t)(size_t m,
                                                                                          size_t n);

/// Conv im2col functions (A = activation needs im2col)
typedef size_t (*tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_get_a_im2col_size_func_t)(
    size_t m, size_t k);
typedef size_t (*tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_get_a_im2col_offset_func_t)(
    size_t m_idx, size_t k_idx, size_t lda_im2col);
typedef void (*tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_run_im2col_func_t)(
    size_t group_idx, const void *activation, void *im2col_buf, const tqt_conv_params *params);

/// Core function
typedef void (*tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_run_conv_func_t)(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
    size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
    int8_t clamp_min, int8_t clamp_max, const struct tqt_qai8_qai8_qsi8cx_params *params);

/// Micro-kernel interface
struct tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_ukernel
{
    tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_get_m_step_func_t get_m_step;
    tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_get_n_step_func_t get_n_step;
    tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_get_b_offset_func_t get_b_offset;
    tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_get_c_offset_func_t get_c_offset;
    tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_get_bias_offset_func_t get_bias_offset;
    tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_get_d_offset_func_t get_d_offset;
    tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_get_d_size_func_t get_d_size;
    tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_get_a_im2col_size_func_t
        get_a_im2col_size;
    tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_get_a_im2col_offset_func_t
        get_a_im2col_offset;
    tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_run_im2col_func_t run_im2col;
    tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_run_conv_func_t run_conv;
};

#ifdef __cplusplus
}  // extern "C"
#endif
