//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include <stddef.h>
#include <stdint.h>

/// Shared INT8 IGEMM helpers for the GEMM 4x1vl register layout.
///
/// Register allocation:
///   v0:       Weight vector
///   a0-a3:    Input scalars for up to four spatial positions
///   v8-v15:   Widening multiplication temporaries
///   v16-v31:  INT32 accumulators for up to four spatial positions
template <size_t MR>
static inline void tqt_vmacc_direct_kernel_igemm_i8_i8_4x1vl_rvv(size_t vl, int32_t a0, int32_t a1,
                                                                 int32_t a2, int32_t a3)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile(
        "vsetvli zero, %0, e16, m2, ta, ma\n\t"
        "vsext.vf2 v8, v0\n\t"
        :
        : "r"(vl)
        : "v8", "v9");
    if constexpr (MR >= 1)
        asm volatile("vwmacc.vx v16, %0, v8" : : "r"(a0) : "v16", "v17", "v18", "v19");
    if constexpr (MR >= 2)
        asm volatile("vwmacc.vx v20, %0, v8" : : "r"(a1) : "v20", "v21", "v22", "v23");
    if constexpr (MR >= 3)
        asm volatile("vwmacc.vx v24, %0, v8" : : "r"(a2) : "v24", "v25", "v26", "v27");
    if constexpr (MR >= 4)
        asm volatile("vwmacc.vx v28, %0, v8" : : "r"(a3) : "v28", "v29", "v30", "v31");
}

template <size_t MR, bool CheckBounds, typename LoadWeights>
static inline void tqt_igemmcl_fma_loop_i8_i8_4x1vl_rvv(size_t IC,
                                                        const int8_t *const input_bases[4],
                                                        size_t vl, LoadWeights load_weights,
                                                        int32_t input_zp)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    for (size_t ic = 0; ic < IC; ic++) {
        auto load_input = [&](size_t i) {
            if constexpr (CheckBounds) {
                return input_bases[i] ? ((int32_t)input_bases[i][ic] - input_zp) : int32_t{0};
            } else {
                return ((int32_t)input_bases[i][ic] - input_zp);
            }
        };
        int32_t a0 = 0;
        int32_t a1 = 0;
        int32_t a2 = 0;
        int32_t a3 = 0;
        if constexpr (MR >= 1)
            a0 = load_input(0);
        if constexpr (MR >= 2)
            a1 = load_input(1);
        if constexpr (MR >= 3)
            a2 = load_input(2);
        if constexpr (MR >= 4)
            a3 = load_input(3);

        load_weights();
        tqt_vmacc_direct_kernel_igemm_i8_i8_4x1vl_rvv<MR>(vl, a0, a1, a2, a3);
    }
}
