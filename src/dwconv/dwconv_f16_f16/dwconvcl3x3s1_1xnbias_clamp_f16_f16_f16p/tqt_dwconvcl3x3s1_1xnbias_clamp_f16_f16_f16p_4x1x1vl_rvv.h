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

/// Specialized cl dwconv 3x3 stride=1, FP16. Tile: 4x1x1vl.
/// Input/output layout: NHWC (raw, not packed).
/// Weight layout: cl-packed [OC1, KH, KW, OC0] via run_pack_weight.
/// get_oh_step() == 4. Caller must invoke run_dwconv with oh ranges aligned to OH_STEP
/// (final chunk may be shorter; handled internally).

size_t tqt_get_input_size_dwconvcl3x3s1_1xnbias_clamp_f16_f16_f16p_4x1x1vl_rvv(
    const tqt_conv_params *params);
size_t tqt_get_output_size_dwconvcl3x3s1_1xnbias_clamp_f16_f16_f16p_4x1x1vl_rvv(
    const tqt_conv_params *params);
size_t tqt_get_packed_weight_size_dwconvcl3x3s1_1xnbias_clamp_f16_f16_f16p_4x1x1vl_rvv(
    const tqt_conv_params *params);
size_t tqt_get_oh_step_dwconvcl3x3s1_1xnbias_clamp_f16_f16_f16p_4x1x1vl_rvv(void);

void tqt_run_pack_weight_dwconvcl3x3s1_1xnbias_clamp_f16_f16_f16p_4x1x1vl_rvv(
    const void *weight, void *packed_weight, const tqt_conv_params *params);

void tqt_run_dwconv_dwconvcl3x3s1_1xnbias_clamp_f16_f16_f16p_4x1x1vl_rvv(
    size_t oh_start, size_t oh_end, const void *input, const void *packed_weight, const void *bias,
    void *output, const tqt_conv_params *params, float16_t clamp_min, float16_t clamp_max);

#ifdef __cplusplus
}
#endif
