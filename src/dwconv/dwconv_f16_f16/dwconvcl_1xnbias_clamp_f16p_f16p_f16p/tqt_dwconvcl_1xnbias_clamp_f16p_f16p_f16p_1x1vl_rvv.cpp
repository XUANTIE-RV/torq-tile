//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "tqt_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv.h"

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"
#include "src/common/tqt_dwconv_pack_rvv.h"
#include "src/dwconv/dwconv_f16_f16/tqt_dwconv_generic_packed_kernel_f16_rvv.h"

// ----------------------------------------------------------------------------
// Size queries
// ----------------------------------------------------------------------------

size_t tqt_get_output_size_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv(
    const tqt_conv_params *params)
{
    const size_t spatial =
        params->output_shape[0] * params->output_shape[1] * params->output_shape[2];
    return tqt_dwconv_packed_tensor_bytes_f16_rvv(params->output_channel, spatial);
}

size_t tqt_get_packed_input_size_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv(
    const tqt_conv_params *params)
{
    const size_t spatial = params->input_shape[0] * params->input_shape[1] * params->input_shape[2];
    return tqt_dwconv_packed_tensor_bytes_f16_rvv(params->input_channel, spatial);
}

size_t tqt_get_packed_weight_size_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv(
    const tqt_conv_params *params)
{
    const size_t kernel_size =
        params->kernel_shape[0] * params->kernel_shape[1] * params->kernel_shape[2];
    return tqt_dwconv_packed_weight_bytes_f16_rvv(params->output_channel, kernel_size);
}

// ----------------------------------------------------------------------------
// Pack / Unpack wrappers (cl-family helpers)
// ----------------------------------------------------------------------------

void tqt_run_pack_input_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv(
    const void *input, void *packed_input, const tqt_conv_params *params)
{
    tqt_dwconv_pack_input_cl_f16_rvv((const float16_t *)input, (float16_t *)packed_input, params);
}

void tqt_run_pack_weight_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv(
    const void *weight, void *packed_weight, const tqt_conv_params *params)
{
    tqt_dwconv_pack_weight_cl_f16_rvv((const float16_t *)weight, (float16_t *)packed_weight,
                                      params);
}

void tqt_run_unpack_output_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv(
    const void *packed_output, void *output, const tqt_conv_params *params)
{
    tqt_dwconv_unpack_output_cl_f16_rvv((const float16_t *)packed_output, (float16_t *)output,
                                        params);
}

// ----------------------------------------------------------------------------
// Compute (delegates to shared core)
// ----------------------------------------------------------------------------

void tqt_run_dwconv_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv(
    const void *packed_input, const void *packed_weight, const void *bias, void *packed_output,
    const tqt_conv_params *params, float16_t clamp_min, float16_t clamp_max)
{
    tqt_dwconv_generic_packed_kernel_f16_rvv(
        (const float16_t *)packed_input, (const float16_t *)packed_weight, (const float16_t *)bias,
        (float16_t *)packed_output, params, clamp_min, clamp_max);
}
