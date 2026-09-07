//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include "src/gemm/gemm_i8_i5/tqt_params_i8_i5.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_m_step_func_t)(void);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_n_step_func_t)(void);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_a_packed_offset_func_t)(size_t,
                                                                                          size_t,
                                                                                          size_t,
                                                                                          size_t);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_b_packed_offset_func_t)(size_t,
                                                                                          size_t,
                                                                                          size_t,
                                                                                          size_t);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_scale_a_offset_func_t)(size_t,
                                                                                         size_t,
                                                                                         size_t);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_scale_b_offset_func_t)(size_t,
                                                                                         size_t,
                                                                                         size_t);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_c_offset_func_t)(size_t, size_t,
                                                                                   size_t);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_bias_offset_func_t)(size_t);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_d_offset_func_t)(size_t, size_t,
                                                                                   size_t);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_d_size_func_t)(size_t, size_t);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_a_packed_size_func_t)(size_t,
                                                                                        size_t);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_b_packed_size_func_t)(size_t,
                                                                                        size_t);
typedef void (*tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_run_a_pack_func_t)(size_t, size_t,
                                                                               size_t, size_t,
                                                                               size_t, const void *,
                                                                               void *);
typedef void (*tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_run_bt_pack_func_t)(
    size_t, size_t, size_t, size_t, size_t, const void *, void *);
typedef void (*tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_run_gemm_func_t)(
    size_t, size_t, size_t, const void *, size_t, size_t, const void *, size_t, size_t,
    const void *, size_t, void *, size_t, const void *, float, float,
    const struct tqt_qsi8d32_qsi5c32_params *);

struct tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_ukernel
{
    tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_m_step_func_t get_m_step;
    tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_n_step_func_t get_n_step;
    tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_a_packed_offset_func_t get_a_packed_offset;
    tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_b_packed_offset_func_t get_b_packed_offset;
    tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_scale_a_offset_func_t get_scale_a_offset;
    tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_scale_b_offset_func_t get_scale_b_offset;
    tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_c_offset_func_t get_c_offset;
    tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_bias_offset_func_t get_bias_offset;
    tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_d_offset_func_t get_d_offset;
    tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_d_size_func_t get_d_size;
    tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_a_packed_size_func_t get_a_packed_size;
    tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_get_b_packed_size_func_t get_b_packed_size;
    tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_run_a_pack_func_t run_a_pack;
    tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_run_bt_pack_func_t run_bt_pack;
    tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_run_gemm_func_t run_gemm;
};

#ifdef __cplusplus
}  // extern "C"
#endif
