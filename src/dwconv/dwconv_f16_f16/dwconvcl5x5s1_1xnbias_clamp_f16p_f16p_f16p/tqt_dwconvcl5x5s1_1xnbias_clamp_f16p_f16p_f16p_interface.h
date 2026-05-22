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

// Specialized dwconv (cl-packed family), 5x5 stride=1, 1xN bias, clamp, FP16, packed.
//   input  layout: cl (NHWC) packed -> NC1HWC0 (2D only, KD=ID=OD=1)
//   weight layout: cl [KH=5, KW=5, 1, OC] packed -> [OC1, 5, 5, OC0]
//   bias   layout: [OC]
//   output layout: NC1HWC0 (caller unpacks via run_unpack_output)
// Caller invokes run_dwconv per OH chunk of size get_oh_step() (= 4).
// dilation=1 only. KD=ID=OD=1 asserted.

typedef size_t (*tqt_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_get_output_size_func_t)(
    const tqt_conv_params *params);
typedef size_t (*tqt_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_get_packed_size_func_t)(
    const tqt_conv_params *params);
typedef size_t (*tqt_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_get_oh_step_func_t)(void);
typedef void (*tqt_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_run_pack_func_t)(
    const void *src, void *dst, const tqt_conv_params *params);
typedef void (*tqt_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_run_dwconv_func_t)(
    size_t oh_start, size_t oh_end, const void *packed_input, const void *packed_weight,
    const void *bias, void *packed_output, const tqt_conv_params *params, float16_t clamp_min,
    float16_t clamp_max);

struct tqt_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_ukernel
{
    tqt_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_get_output_size_func_t get_output_size;
    tqt_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_get_packed_size_func_t get_packed_input_size;
    tqt_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_get_packed_size_func_t get_packed_weight_size;
    tqt_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_get_oh_step_func_t get_oh_step;
    tqt_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_run_pack_func_t run_pack_input;
    tqt_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_run_pack_func_t run_pack_weight;
    tqt_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_run_pack_func_t run_unpack_output;
    tqt_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_run_dwconv_func_t run_dwconv;
};

#ifdef __cplusplus
}
#endif
