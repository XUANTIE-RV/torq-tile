//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/// Packed quantization parameters for f16 x qau8c32 GEMM.
struct tqt_f16_qau8c32_params
{
    const void *qparams_packed;
};

#ifdef __cplusplus
}  // extern "C"
#endif
