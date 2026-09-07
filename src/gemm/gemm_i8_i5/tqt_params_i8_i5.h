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

/// Parameters for qsi8d32 A x normalized qsi5c32 B -> f32.
///
/// scale_a and scale_b are [M, K/32] and [N, K/32]. Each signed I5 value is
/// stored as a low nibble plus a one-bit sign plane.
struct tqt_qsi8d32_qsi5c32_params
{
    const float *scale_a;
    size_t ldsa;
    const float *scale_b;
    size_t ldsb;
};

#ifdef __cplusplus
}  // extern "C"
#endif
