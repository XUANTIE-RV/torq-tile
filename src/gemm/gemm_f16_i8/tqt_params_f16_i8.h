//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/// Packed quantization parameters for f16 x qsi8c16 GEMM.
struct tqt_f16_qsi8c16_params
{
    const void *qparams_packed;
};

/// Packed quantization parameters for f16 x qsi8c32 GEMM.
struct tqt_f16_qsi8c32_params
{
    const void *qparams_packed;
};

#ifdef __cplusplus
}  // extern "C"
#endif
