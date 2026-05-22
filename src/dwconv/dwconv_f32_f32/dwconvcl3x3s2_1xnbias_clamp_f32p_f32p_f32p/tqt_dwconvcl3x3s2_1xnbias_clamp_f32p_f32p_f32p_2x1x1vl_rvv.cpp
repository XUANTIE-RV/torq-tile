//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "tqt_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv.h"

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"
#include "src/common/tqt_dwconv_pack_rvv.h"
#include "src/dwconv/dwconv_f32_f32/tqt_dwconv_3x3s2_packed_kernel_f32_rvv.h"

// ----- Specialized constants -----
static const size_t kh = 3;
static const size_t kw = 3;
static const size_t sh = 2;
static const size_t sw = 2;
static const size_t oh_step = 2;

// ----- Size queries -----

size_t tqt_get_output_size_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv(
    const tqt_conv_params *params)
{
    const size_t spatial =
        params->output_shape[0] * params->output_shape[1] * params->output_shape[2];
    return tqt_dwconv_packed_tensor_bytes_f32_rvv(params->output_channel, spatial);
}

size_t tqt_get_packed_input_size_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv(
    const tqt_conv_params *params)
{
    const size_t spatial = params->input_shape[0] * params->input_shape[1] * params->input_shape[2];
    return tqt_dwconv_packed_tensor_bytes_f32_rvv(params->input_channel, spatial);
}

size_t tqt_get_packed_weight_size_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv(
    const tqt_conv_params *params)
{
    const size_t kernel_size =
        params->kernel_shape[0] * params->kernel_shape[1] * params->kernel_shape[2];
    return tqt_dwconv_packed_weight_bytes_f32_rvv(params->output_channel, kernel_size);
}

size_t tqt_get_oh_step_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv(void)
{
    return oh_step;
}

// ----- Pack / Unpack (cl-family) -----

void tqt_run_pack_input_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv(
    const void *input, void *packed_input, const tqt_conv_params *params)
{
    tqt_dwconv_pack_input_cl_f32_rvv((const float *)input, (float *)packed_input, params);
}

void tqt_run_pack_weight_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv(
    const void *weight, void *packed_weight, const tqt_conv_params *params)
{
    tqt_dwconv_pack_weight_cl_f32_rvv((const float *)weight, (float *)packed_weight, params);
}

void tqt_run_unpack_output_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv(
    const void *packed_output, void *output, const tqt_conv_params *params)
{
    tqt_dwconv_unpack_output_cl_f32_rvv((const float *)packed_output, (float *)output, params);
}

// ----- Compute -----

void tqt_run_dwconv_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv(
    size_t oh_start, size_t oh_end, const void *packed_input, const void *packed_weight,
    const void *bias, void *packed_output, const tqt_conv_params *params, float clamp_min,
    float clamp_max)
{
    // Specialization assertions
    TQT_ASSERT(params->kernel_shape[0] == 1 && params->input_shape[0] == 1 &&
               params->output_shape[0] == 1);
    TQT_ASSERT(params->kernel_shape[1] == kh && params->kernel_shape[2] == kw);
    TQT_ASSERT(params->stride[1] == sh && params->stride[2] == sw);
    TQT_ASSERT(params->dilation[0] == 1 && params->dilation[1] == 1 && params->dilation[2] == 1);

    tqt_dwconv_3x3s2_packed_kernel_f32_rvv(oh_start, oh_end, (const float *)packed_input,
                                           (const float *)packed_weight, (const float *)bias,
                                           (float *)packed_output, params, clamp_min, clamp_max);
}
