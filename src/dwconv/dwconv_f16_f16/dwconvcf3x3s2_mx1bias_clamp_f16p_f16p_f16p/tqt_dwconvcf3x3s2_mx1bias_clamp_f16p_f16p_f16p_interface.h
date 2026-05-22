//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

#ifdef __cplusplus
extern "C" {
#endif

// Specialized dwconv (cf-packed family), 3x3 stride=2, Mx1 bias, clamp, FP16, packed.
//   input  layout: cf (NCHW) packed -> NC1HWC0 (2D only, KD=ID=OD=1)
//   weight layout: cf [OC, 1, KH=3, KW=3] packed -> [OC1, 3, 3, OC0]
//   bias   layout: [OC]
//   output layout: NC1HWC0 (caller unpacks via run_unpack_output)
// Caller invokes run_dwconv per OH chunk of size get_oh_step() (= 2).
// dilation=1 only. KD=ID=OD=1 asserted.

typedef size_t (*tqt_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_get_output_size_func_t)(
    const tqt_conv_params *params);
typedef size_t (*tqt_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_get_packed_size_func_t)(
    const tqt_conv_params *params);
typedef size_t (*tqt_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_get_oh_step_func_t)(void);
typedef void (*tqt_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_run_pack_func_t)(
    const void *src, void *dst, const tqt_conv_params *params);
typedef void (*tqt_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_run_dwconv_func_t)(
    size_t oh_start, size_t oh_end, const void *packed_input, const void *packed_weight,
    const void *bias, void *packed_output, const tqt_conv_params *params, float16_t clamp_min,
    float16_t clamp_max);

struct tqt_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_ukernel
{
    tqt_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_get_output_size_func_t get_output_size;
    tqt_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_get_packed_size_func_t get_packed_input_size;
    tqt_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_get_packed_size_func_t get_packed_weight_size;
    tqt_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_get_oh_step_func_t get_oh_step;
    tqt_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_run_pack_func_t run_pack_input;
    tqt_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_run_pack_func_t run_pack_weight;
    tqt_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_run_pack_func_t run_unpack_output;
    tqt_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_run_dwconv_func_t run_dwconv;
};

#ifdef __cplusplus
}
#endif
