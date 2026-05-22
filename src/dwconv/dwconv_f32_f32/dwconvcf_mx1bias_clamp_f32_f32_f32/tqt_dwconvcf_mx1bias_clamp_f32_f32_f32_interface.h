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

// Generic dwconv (channel-first, NCDHW), Mx1 bias, clamp, FP32, non-packed.
//   input  layout: [IC, ID, IH, IW]   (single batch)
//   weight layout: [OC, 1, KD, KH, KW] (PyTorch dwconv convention)
//   bias   layout: [OC] or NULL
//   output layout: [OC, OD, OH, OW]
// Constraints (asserted at runtime):
//   params->groups == params->input_channel  (dwconv requirement)
//   params->output_channel % params->input_channel == 0
//   PyTorch OC ordering: oc = ic + m * IC,  ic = oc % IC

/// Returns size in bytes of the output buffer (single batch).
typedef size_t (*tqt_dwconvcf_mx1bias_clamp_f32_f32_f32_get_output_size_func_t)(
    const tqt_conv_params *params);

/// Runs the dwconv. Processes ONE batch. Caller loops batch.
typedef void (*tqt_dwconvcf_mx1bias_clamp_f32_f32_f32_run_dwconv_func_t)(
    const void *input, const void *weight, const void *bias, void *output,
    const tqt_conv_params *params, float clamp_min, float clamp_max);

/// Micro-kernel struct: generic (non-packed) cf dwconv.
struct tqt_dwconvcf_mx1bias_clamp_f32_f32_f32_ukernel
{
    tqt_dwconvcf_mx1bias_clamp_f32_f32_f32_get_output_size_func_t get_output_size;
    tqt_dwconvcf_mx1bias_clamp_f32_f32_f32_run_dwconv_func_t run_dwconv;
};

#ifdef __cplusplus
}  // extern "C"
#endif
