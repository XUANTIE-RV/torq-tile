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

// Conv micro-kernel type: convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p
// Channel-first convolution (NCDHW), packed, Mx1 bias with requantize.
// A = weight [OC_per_group, IC_per_group * KD * KH * KW] (per-channel symmetric, needs packing)
// B = im2col(activation) [IC_per_group * KD * KH * KW, OD * OH * OW] (asymmetric, im2col+packing)

/// Helper functions ("get" methods)
typedef size_t (*tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_m_step_func_t)(void);
typedef size_t (*tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_n_step_func_t)(void);
typedef size_t (*tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_a_packed_offset_func_t)(
    size_t m_idx, size_t k_idx, size_t lda, size_t actual_m);
typedef size_t (
    *tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_b_im2colpack_offset_func_t)(
    size_t n_idx, size_t k_idx, size_t ldb, size_t actual_n);
typedef size_t (*tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_c_offset_func_t)(
    size_t m_idx, size_t n_idx, size_t ldc);
typedef size_t (*tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_bias_offset_func_t)(
    size_t m_idx);
typedef size_t (*tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_d_offset_func_t)(
    size_t m_idx, size_t n_idx, size_t ldd);
typedef size_t (*tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_d_size_func_t)(
    size_t m, size_t n);
typedef size_t (*tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_a_packed_size_func_t)(
    size_t m, size_t k);
typedef size_t (
    *tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_b_im2colpack_size_func_t)(size_t n,
                                                                                          size_t k);

/// Pack functions
typedef void (*tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_run_a_pack_func_t)(
    size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx, const void *A, void *A_packed);

/// Fused im2col+pack for B (activation)
typedef void (*tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_run_b_im2colpack_func_t)(
    size_t group_idx, const void *activation, size_t ldb_packed, size_t k_idx, void *B_packed,
    const tqt_conv_params *params);

/// Core function
typedef void (*tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_run_conv_func_t)(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, int8_t clamp_min, int8_t clamp_max,
    const struct tqt_qai8_qsi8dx_qai8_params *params);

/// Micro-kernel interface
struct tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_ukernel
{
    tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_m_step_func_t get_m_step;
    tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_n_step_func_t get_n_step;
    tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_a_packed_offset_func_t
        get_a_packed_offset;
    tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_b_im2colpack_offset_func_t
        get_b_im2colpack_offset;
    tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_c_offset_func_t get_c_offset;
    tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_bias_offset_func_t get_bias_offset;
    tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_d_offset_func_t get_d_offset;
    tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_d_size_func_t get_d_size;
    tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_a_packed_size_func_t
        get_a_packed_size;
    tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_b_im2colpack_size_func_t
        get_b_im2colpack_size;
    tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_run_a_pack_func_t run_a_pack;
    tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_run_b_im2colpack_func_t run_b_im2colpack;
    tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_run_conv_func_t run_conv;
};

#ifdef __cplusplus
}  // extern "C"
#endif
