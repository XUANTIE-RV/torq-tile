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

// Generic dwconv (channel-last, NDHWC), 1xN bias, clamp, FP16, non-packed.
//   input  layout: [ID, IH, IW, IC]
//   weight layout: [KD, KH, KW, 1, OC]   (HWIO style)
//   bias   layout: [OC] or NULL
//   output layout: [OD, OH, OW, OC]
// Constraints: groups == IC, OC % IC == 0, oc = ic + m * IC.

typedef size_t (*tqt_dwconvcl_1xnbias_clamp_f16_f16_f16t_get_output_size_func_t)(
    const tqt_conv_params *params);
typedef void (*tqt_dwconvcl_1xnbias_clamp_f16_f16_f16t_run_dwconv_func_t)(
    const void *input, const void *weight, const void *bias, void *output,
    const tqt_conv_params *params, float16_t clamp_min, float16_t clamp_max);

struct tqt_dwconvcl_1xnbias_clamp_f16_f16_f16t_ukernel
{
    tqt_dwconvcl_1xnbias_clamp_f16_f16_f16t_get_output_size_func_t get_output_size;
    tqt_dwconvcl_1xnbias_clamp_f16_f16_f16t_run_dwconv_func_t run_dwconv;
};

#ifdef __cplusplus
}
#endif
