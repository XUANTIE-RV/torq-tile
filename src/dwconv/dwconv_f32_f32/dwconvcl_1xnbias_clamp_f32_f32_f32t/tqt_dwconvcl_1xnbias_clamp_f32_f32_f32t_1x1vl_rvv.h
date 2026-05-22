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

/// Generic dwconv micro-kernel, channel-last (NDHWC), 1xN bias, clamp, FP32, non-packed.
/// Tile: 1x1vl (one OH/OW point, vlmax channels per inner step).
size_t tqt_get_output_size_dwconvcl_1xnbias_clamp_f32_f32_f32t_1x1vl_rvv(
    const tqt_conv_params *params);

void tqt_run_dwconv_dwconvcl_1xnbias_clamp_f32_f32_f32t_1x1vl_rvv(const void *input,
                                                                  const void *weight,
                                                                  const void *bias, void *output,
                                                                  const tqt_conv_params *params,
                                                                  float clamp_min, float clamp_max);

#ifdef __cplusplus
}
#endif
