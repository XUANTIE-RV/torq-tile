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

// Generic dwconv (cf-packed family), Mx1 bias, clamp, FP32, packed (NC1DHWC0).
//   input  layout: cf (NCDHW) -> pack -> NC1DHWC0
//   weight layout: cf [OC, 1, KD, KH, KW] -> pack -> [OC1, KD, KH, KW, OC0]
//   bias   layout: [OC] (NEVER packed)
//   output layout: NC1DHWC0 -> unpack -> NCDHW
//
// All packed buffers are SINGLE-BATCH; caller loops batch.
// Tail channel block: vl = IC - (IC1-1)*C0; never zero-padded.

typedef size_t (*tqt_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_get_output_size_func_t)(
    const tqt_conv_params *params);
typedef size_t (*tqt_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_get_packed_size_func_t)(
    const tqt_conv_params *params);

typedef void (*tqt_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_run_pack_func_t)(
    const void *src, void *dst, const tqt_conv_params *params);

typedef void (*tqt_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_run_dwconv_func_t)(
    const void *packed_input, const void *packed_weight, const void *bias, void *packed_output,
    const tqt_conv_params *params, float clamp_min, float clamp_max);

struct tqt_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_ukernel
{
    // size queries
    tqt_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_get_output_size_func_t get_output_size;
    tqt_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_get_packed_size_func_t get_packed_input_size;
    tqt_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_get_packed_size_func_t get_packed_weight_size;
    // pack / unpack
    tqt_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_run_pack_func_t run_pack_input;
    tqt_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_run_pack_func_t run_pack_weight;
    tqt_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_run_pack_func_t run_unpack_output;
    // compute
    tqt_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_run_dwconv_func_t run_dwconv;
};

#ifdef __cplusplus
}
#endif
