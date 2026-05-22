//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "src/gemm/gemm_i8_i8/tqt_params_i8_i8.h"

#ifdef __cplusplus
extern "C" {
#endif

// All micro-kernels variants of the same type share the same interfaces
// In this case, the micro-kernel type is: gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p

/// Micro-kernel helper functions ("get" methods)
typedef size_t (*tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_m_step_func)(void);
typedef size_t (*tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_n_step_func)(void);
typedef size_t (*tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_a_packed_offset_func)(
    size_t m_idx, size_t k_idx, size_t lda, size_t actual_m);
typedef size_t (*tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_b_packed_offset_func)(
    size_t n_idx, size_t k_idx, size_t ldb, size_t actual_n);
typedef size_t (*tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_c_offset_func)(
    size_t m_idx, size_t n_idx, size_t ldc);
typedef size_t (*tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_bias_offset_func)(
    size_t m_idx);
typedef size_t (*tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_d_offset_func)(
    size_t m_idx, size_t n_idx, size_t ldd);
typedef size_t (*tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_d_size_func)(size_t m,
                                                                                       size_t n);
typedef size_t (*tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_a_packed_size_func)(
    size_t m, size_t k);
typedef size_t (*tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_b_packed_size_func)(
    size_t n, size_t k);

/// Micro-kernel core functions ("run" methods)
typedef void (*tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_run_a_pack_func)(
    size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx, const void *A, void *A_packed);
typedef void (*tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_run_b_pack_func)(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed);
typedef void (*tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_run_bt_pack_func)(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed);
typedef void (*tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_run_gemm_func)(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, int8_t clamp_min, int8_t clamp_max,
    const struct tqt_qai8_qsi8dx_qai8_params *params);

/// Micro-kernel interface
struct tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_ukernel
{
    tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_m_step_func get_m_step;
    tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_n_step_func get_n_step;
    tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_a_packed_offset_func
        get_a_packed_offset;
    tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_b_packed_offset_func
        get_b_packed_offset;
    tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_c_offset_func get_c_offset;
    tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_bias_offset_func get_bias_offset;
    tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_d_offset_func get_d_offset;
    tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_d_size_func get_d_size;
    tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_a_packed_size_func get_a_packed_size;
    tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_get_b_packed_size_func get_b_packed_size;
    tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_run_a_pack_func run_a_pack;
    tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_run_b_pack_func run_b_pack;
    tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_run_bt_pack_func run_bt_pack;
    tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_run_gemm_func run_gemm;
};

#ifdef __cplusplus
}  // extern "C"
#endif
