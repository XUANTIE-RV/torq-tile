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

// Generic dwconv (cl-packed family), 1xN bias, clamp, FP16, packed (NC1DHWC0).
//   input  layout: cl (NDHWC) -> pack -> NC1DHWC0
//   weight layout: cl [KD, KH, KW, 1, OC] -> pack -> [OC1, KD, KH, KW, OC0]
//   bias   layout: [OC]
//   output layout: NC1DHWC0 -> unpack -> NDHWC
// Compute core is identical to cf-packed; only pack/unpack helpers differ.

typedef size_t (*tqt_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_get_output_size_func_t)(
    const tqt_conv_params *params);
typedef size_t (*tqt_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_get_packed_size_func_t)(
    const tqt_conv_params *params);

typedef void (*tqt_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_run_pack_func_t)(
    const void *src, void *dst, const tqt_conv_params *params);

typedef void (*tqt_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_run_dwconv_func_t)(
    const void *packed_input, const void *packed_weight, const void *bias, void *packed_output,
    const tqt_conv_params *params, float16_t clamp_min, float16_t clamp_max);

struct tqt_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_ukernel
{
    tqt_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_get_output_size_func_t get_output_size;
    tqt_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_get_packed_size_func_t get_packed_input_size;
    tqt_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_get_packed_size_func_t get_packed_weight_size;
    tqt_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_run_pack_func_t run_pack_input;
    tqt_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_run_pack_func_t run_pack_weight;
    tqt_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_run_pack_func_t run_unpack_output;
    tqt_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_run_dwconv_func_t run_dwconv;
};

#ifdef __cplusplus
}
#endif
