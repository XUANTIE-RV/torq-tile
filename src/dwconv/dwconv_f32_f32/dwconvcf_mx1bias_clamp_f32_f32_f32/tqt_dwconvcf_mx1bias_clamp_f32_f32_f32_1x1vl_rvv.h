//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Generic dwconv micro-kernel, channel-first (NCDHW), Mx1 bias, clamp, FP32, non-packed.
/// Tile: 1x1vl (one OH row, one OW column, one vlmax channel chunk per inner step).
///
/// Single-batch processing. Caller loops over batch.

/// Returns size in bytes of the output buffer (single batch).
size_t tqt_get_output_size_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv(
    const tqt_conv_params *params);

/// Runs depthwise convolution.
///
/// @param input   [IC, ID, IH, IW] activation tensor, single batch.
/// @param weight  [OC, 1, KD, KH, KW] PyTorch dwconv weight.
/// @param bias    [OC] or NULL.
/// @param output  [OC, OD, OH, OW] output buffer, single batch.
/// @param params  Convolution parameters; params->groups must equal params->input_channel.
/// @param clamp_min,clamp_max  Output element-wise clamp range.
void tqt_run_dwconv_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv(const void *input,
                                                                 const void *weight,
                                                                 const void *bias, void *output,
                                                                 const tqt_conv_params *params,
                                                                 float clamp_min, float clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif
