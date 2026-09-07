//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "src/gemm/gemm_i8_i8/tqt_kernel_gemm_i8_i8_4x1vl_blockquant_rvv.h"

static inline void tqt_compute_tile_kernel_gemm_qsi8d32_qsi8c32_4x1vl_rvv(
    size_t mr, size_t nr, size_t k, size_t k_idx, float *d, const int8_t *a_packed,
    const int8_t *b_packed, const float *scale_a, size_t ldsa, const float *scale_b, size_t ldsb,
    const float *c, const float *bias, size_t ldc, size_t ldd, float clamp_min, float clamp_max,
    bool apply_clamp)
{
    tqt_compute_tile_kernel_gemm_i8_i8_4x1vl_blockquant_rvv(
        mr, nr, k, k_idx, 32, 32, d, a_packed, b_packed, scale_a, ldsa, scale_b, ldsb, c, bias, ldc,
        ldd, clamp_min, clamp_max, apply_clamp);
}
