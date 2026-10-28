//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "src/gemm/gemm_i8_i2/tqt_params_i8_i2.h"

#ifdef __cplusplus
extern "C" {
#endif

// All micro-kernel variants of the same kernel base share the same interface.
// Kernel base: gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp

typedef size_t (*tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_get_m_step_func_t)(void);
typedef size_t (*tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_get_n_step_func_t)(void);
typedef size_t (*tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_get_a_packed_offset_func_t)(
    size_t m_idx, size_t k_idx, size_t lda, size_t actual_m);
typedef size_t (*tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_get_b_packed_offset_func_t)(
    size_t n_idx, size_t k_idx, size_t ldb, size_t actual_n);
typedef size_t (*tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_get_bias_offset_func_t)(
    size_t n_idx);
typedef size_t (*tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_get_d_offset_func_t)(
    size_t m_idx, size_t n_idx, size_t ldd);
typedef size_t (*tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_get_d_size_func_t)(size_t m,
                                                                                         size_t n);
typedef size_t (*tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_get_a_packed_size_func_t)(
    size_t m, size_t k);
typedef size_t (*tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_get_b_packed_size_func_t)(
    size_t n, size_t k);

typedef void (*tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_run_a_pack_func_t)(
    size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx, const void *A, void *A_packed);
typedef void (*tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_run_b_pack_func_t)(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed);
typedef void (*tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_run_bt_pack_func_t)(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed);
typedef void (*tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_run_gemm_func_t)(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, void *D, size_t ldd, const int32_t *bias,
    int8_t clamp_min, int8_t clamp_max, const int8_t *lut,
    const struct tqt_qai8_qai8_qsi2cx_params *params);

struct tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_ukernel
{
    tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_get_m_step_func_t get_m_step;
    tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_get_n_step_func_t get_n_step;
    tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_get_a_packed_offset_func_t
        get_a_packed_offset;
    tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_get_b_packed_offset_func_t
        get_b_packed_offset;
    tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_get_bias_offset_func_t get_bias_offset;
    tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_get_d_offset_func_t get_d_offset;
    tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_get_d_size_func_t get_d_size;
    tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_get_a_packed_size_func_t get_a_packed_size;
    tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_get_b_packed_size_func_t get_b_packed_size;
    tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_run_a_pack_func_t run_a_pack;
    tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_run_b_pack_func_t run_b_pack;
    tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_run_bt_pack_func_t run_bt_pack;
    tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_run_gemm_func_t run_gemm;
};

#ifdef __cplusplus
}  // extern "C"
#endif
