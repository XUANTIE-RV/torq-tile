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

// Specialized dwconv (cl family), 3x3 stride=1, 1xN bias, clamp, FP32.
//   input  layout: cl (NHWC) raw, not packed
//   weight layout: cl [KH=3, KW=3, 1, OC] -> packed [OC1, 3, 3, OC0]
//   bias   layout: [OC]
//   output layout: NHWC raw, not packed
//   Raw weight/bias/output channels: oc = ic * (OC / IC) + m.
//   Packed weight blocks: oc1 = m * IC1 + ic1, with compact IC tails.
// Caller invokes run_dwconv per OH chunk of size get_oh_step() (= 4).
// dilation=1 only. KD=ID=OD=1 asserted.

typedef size_t (*tqt_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_get_input_size_func_t)(
    const tqt_conv_params *params);
typedef size_t (*tqt_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_get_output_size_func_t)(
    const tqt_conv_params *params);
typedef size_t (*tqt_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_get_packed_weight_size_func_t)(
    const tqt_conv_params *params);
typedef size_t (*tqt_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_get_oh_step_func_t)(void);
typedef void (*tqt_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_run_pack_weight_func_t)(
    const void *src, void *dst, const tqt_conv_params *params);
typedef void (*tqt_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_run_dwconv_func_t)(
    size_t oh_start, size_t oh_end, const void *input, const void *packed_weight, const void *bias,
    void *output, const tqt_conv_params *params, float clamp_min, float clamp_max);

struct tqt_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_ukernel
{
    tqt_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_get_input_size_func_t get_input_size;
    tqt_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_get_output_size_func_t get_output_size;
    tqt_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_get_packed_weight_size_func_t
        get_packed_weight_size;
    tqt_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_get_oh_step_func_t get_oh_step;
    tqt_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_run_pack_weight_func_t run_pack_weight;
    tqt_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_run_dwconv_func_t run_dwconv;
};

#ifdef __cplusplus
}
#endif
