//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/// Parameters for mxfp8 (E4M3) x mxfp4 (E2M1) GEMM with bf16 output.
///
/// Carries an optional per-tensor global scale that is applied to the
/// dequantized fp32 accumulator before bias add and bf16 cast in the epilogue:
///
///     D = bf16_cast(clamp(C + (A * B * scale_blocks) * global_scale + bias))
///
/// A run_gemm caller may pass `params == nullptr` to skip the global_scale
/// multiplication entirely (equivalent to `global_scale = 1.0f`). The MX block
/// scales (fp8 E8M0) are always applied inside the mfmacc regardless of params.
struct tqt_mxf8e4_mxf4e2_params
{
    float global_scale;  // = global_scale_a * global_scale_b (per-tensor)
};

#ifdef __cplusplus
}  // extern "C"
#endif
