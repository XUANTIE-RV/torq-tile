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

/// Generic packed dwconv (cf family), Mx1 bias, clamp, FP16. Tile: 1x1vl.
size_t tqt_get_output_size_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv(
    const tqt_conv_params *params);
size_t tqt_get_packed_input_size_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv(
    const tqt_conv_params *params);
size_t tqt_get_packed_weight_size_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv(
    const tqt_conv_params *params);

void tqt_run_pack_input_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv(
    const void *input, void *packed_input, const tqt_conv_params *params);
void tqt_run_pack_weight_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv(
    const void *weight, void *packed_weight, const tqt_conv_params *params);
void tqt_run_unpack_output_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv(
    const void *packed_output, void *output, const tqt_conv_params *params);

void tqt_run_dwconv_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv(
    const void *packed_input, const void *packed_weight, const void *bias, void *packed_output,
    const tqt_conv_params *params, float16_t clamp_min, float16_t clamp_max);

#ifdef __cplusplus
}
#endif
