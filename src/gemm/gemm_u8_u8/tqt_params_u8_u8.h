//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/// Parameters for u8 x u8 -> f32 dequantization output kernels.
///
/// Used by gemm_mx1bias_1xnbias_clamp_f32_u8{_u8t|p_u8p} kernels.
/// The global dequant scale is typically s_a * s_b from per-tensor
/// asymmetric uint8 quantization of both operands.
struct tqt_u8_u8_clamp_f32_params
{
    float scale;
};

#ifdef __cplusplus
}  // extern "C"
#endif
