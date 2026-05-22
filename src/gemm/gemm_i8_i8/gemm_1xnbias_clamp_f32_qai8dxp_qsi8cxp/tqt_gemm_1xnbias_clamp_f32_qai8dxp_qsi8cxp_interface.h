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
// In this case, the micro-kernel type is: gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp
//
// A: per-row asymmetric int8, packed (rows interleaved by mr along K)
// B: per-channel symmetric int8, packed (cols interleaved by nr along K)
// D: f32 output, [M, N] row-major
// bias: f32, length N (1xN row-broadcast)

/// Micro-kernel helper functions ("get" methods)
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_m_step_func_t)(void);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_n_step_func_t)(void);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_a_packed_offset_func_t)(
    size_t m_idx, size_t k_idx, size_t lda, size_t actual_m);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_b_packed_offset_func_t)(
    size_t n_idx, size_t k_idx, size_t ldb, size_t actual_n);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_c_offset_func_t)(size_t m_idx,
                                                                                 size_t n_idx,
                                                                                 size_t ldc);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_bias_offset_func_t)(size_t n_idx);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_d_offset_func_t)(size_t m_idx,
                                                                                 size_t n_idx,
                                                                                 size_t ldd);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_d_size_func_t)(size_t m, size_t n);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_a_packed_size_func_t)(size_t m,
                                                                                      size_t k);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_b_packed_size_func_t)(size_t n,
                                                                                      size_t k);

/// Micro-kernel core functions ("run" methods)
typedef void (*tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_run_a_pack_func_t)(
    size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx, const void *A, void *A_packed);
typedef void (*tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_run_b_pack_func_t)(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed);
typedef void (*tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_run_bt_pack_func_t)(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed);

/// Note: C residual is currently not supported and must be NULL.
typedef void (*tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_run_gemm_func_t)(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float clamp_min, float clamp_max,
    const struct tqt_qai8dxp_qsi8cxp_params *params);

/// Micro-kernel interface
struct tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_ukernel
{
    tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_m_step_func_t get_m_step;
    tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_n_step_func_t get_n_step;
    tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_a_packed_offset_func_t get_a_packed_offset;
    tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_b_packed_offset_func_t get_b_packed_offset;
    tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_c_offset_func_t get_c_offset;
    tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_bias_offset_func_t get_bias_offset;
    tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_d_offset_func_t get_d_offset;
    tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_d_size_func_t get_d_size;
    tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_a_packed_size_func_t get_a_packed_size;
    tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_get_b_packed_size_func_t get_b_packed_size;
    tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_run_a_pack_func_t run_a_pack;
    tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_run_b_pack_func_t run_b_pack;
    tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_run_bt_pack_func_t run_bt_pack;
    tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_run_gemm_func_t run_gemm;
};

#ifdef __cplusplus
}  // extern "C"
#endif
