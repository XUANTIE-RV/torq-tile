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

// Specialized dwconv (cf-packed family), 5x5 stride=1, Mx1 bias, clamp, FP32, packed.
//   input  layout: cf (NCHW) packed -> NC1HWC0 (2D only, KD=ID=OD=1)
//   weight layout: cf [OC, 1, KH=5, KW=5] packed -> [OC1, 5, 5, OC0]
//   bias   layout: [OC]
//   output layout: NC1HWC0 (caller unpacks via run_unpack_output)
// Caller invokes run_dwconv per OH chunk of size get_oh_step() (= 4).
// dilation=1 only. KD=ID=OD=1 asserted.

typedef size_t (*tqt_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_get_output_size_func_t)(
    const tqt_conv_params *params);
typedef size_t (*tqt_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_get_packed_size_func_t)(
    const tqt_conv_params *params);
typedef size_t (*tqt_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_get_oh_step_func_t)(void);
typedef void (*tqt_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_run_pack_func_t)(
    const void *src, void *dst, const tqt_conv_params *params);
typedef void (*tqt_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_run_dwconv_func_t)(
    size_t oh_start, size_t oh_end, const void *packed_input, const void *packed_weight,
    const void *bias, void *packed_output, const tqt_conv_params *params, float clamp_min,
    float clamp_max);

struct tqt_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_ukernel
{
    tqt_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_get_output_size_func_t get_output_size;
    tqt_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_get_packed_size_func_t get_packed_input_size;
    tqt_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_get_packed_size_func_t get_packed_weight_size;
    tqt_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_get_oh_step_func_t get_oh_step;
    tqt_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_run_pack_func_t run_pack_input;
    tqt_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_run_pack_func_t run_pack_weight;
    tqt_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_run_pack_func_t run_unpack_output;
    tqt_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_run_dwconv_func_t run_dwconv;
};

#ifdef __cplusplus
}
#endif
