//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/// Packed quantization parameters for bf16 x qau4c32 GEMM.
struct tqt_bf16_qau4c32_params
{
    const void *qparams_packed;
};

#ifdef __cplusplus
}  // extern "C"
#endif
