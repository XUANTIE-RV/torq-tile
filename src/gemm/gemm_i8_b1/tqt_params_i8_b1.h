//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Parameters for qsi8d32 A x normalized qsb1c128 B -> f32.
///
/// scale_a is [M, K/32], and scale_b is [N, K/128]. A zero bit represents -1
/// and a one bit represents +1.
struct tqt_qsi8d32_qsb1c128_params
{
    const float *scale_a;
    size_t ldsa;
    const float *scale_b;
    size_t ldsb;
};

#ifdef __cplusplus
}  // extern "C"
#endif
