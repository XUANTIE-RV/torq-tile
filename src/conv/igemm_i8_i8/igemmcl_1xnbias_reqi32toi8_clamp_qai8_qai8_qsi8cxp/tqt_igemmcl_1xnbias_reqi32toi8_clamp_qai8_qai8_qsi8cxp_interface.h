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

// Implicit GEMM micro-kernel type: igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp
// Channel-last implicit GEMM convolution (NDHWC), packed weight,
// 1xN bias, requantization (int32->int8), and clamp.
//
// Weight packed from [OC, K] to [OC/vl, K, vl] for contiguous vector loads.
//
// NO K-tiling: full K is processed in a single call (k_idx params are unused).

/// Helper functions
typedef size_t (*tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_get_m_step_func_t)(void);
typedef size_t (*tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_get_n_step_func_t)(void);
typedef size_t (*tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_get_b_packed_offset_func_t)(
    size_t n_idx, size_t k_idx, size_t ldb, size_t actual_n);
typedef size_t (*tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_get_c_offset_func_t)(
    size_t m_idx, size_t n_idx, size_t ldc);
typedef size_t (*tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_get_bias_offset_func_t)(
    size_t n_idx);
typedef size_t (*tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_get_d_offset_func_t)(
    size_t m_idx, size_t n_idx, size_t ldd);
typedef size_t (*tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_get_d_size_func_t)(
    size_t m, size_t n);
typedef size_t (*tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_get_b_packed_size_func_t)(
    size_t n, size_t k);

/// Weight pack function: [OC, K] -> [OC/vl, K, vl]
typedef void (*tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_run_bt_pack_func_t)(
    size_t n, size_t k, size_t ldw, size_t ldw_packed, size_t k_idx, const void *weight,
    void *weight_packed);

/// Core function
typedef void (*tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_run_igemm_func_t)(
    size_t m, size_t n, size_t k, const void *input, size_t group_idx, size_t m_idx, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, int8_t clamp_min, int8_t clamp_max,
    const tqt_conv_params *conv_params, const struct tqt_qai8_qai8_qsi8cx_params *quant_params);

/// Micro-kernel interface
struct tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_ukernel
{
    tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_get_m_step_func_t get_m_step;
    tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_get_n_step_func_t get_n_step;
    tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_get_b_packed_offset_func_t
        get_b_packed_offset;
    tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_get_c_offset_func_t get_c_offset;
    tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_get_bias_offset_func_t get_bias_offset;
    tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_get_d_offset_func_t get_d_offset;
    tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_get_d_size_func_t get_d_size;
    tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_get_b_packed_size_func_t
        get_b_packed_size;
    tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_run_bt_pack_func_t run_bt_pack;
    tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_run_igemm_func_t run_igemm;
};

#ifdef __cplusplus
}  // extern "C"
#endif
