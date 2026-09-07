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

/// Shared FP32 IGEMM helpers for the GEMM 8x3vl register layout.
///
/// Register allocation:
///   v0-v2:   Weight vectors for up to three output-channel groups
///   a0-a7:   Input scalars for up to eight spatial positions
///   v8-v31:  Accumulators D[mr][nr] = v[8 + mr * 3 + nr]
///
/// Channel-first kernels use the layout with their public MR/NR roles swapped. Channel-last
/// kernels use it directly.
template <size_t MR, size_t NR>
static inline void tqt_vfmacc_direct_kernel_igemm_f32_f32_8x3vl_rvv(size_t vl, float a0, float a1,
                                                                    float a2, float a3, float a4,
                                                                    float a5, float a6, float a7)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(vl) :);

    if constexpr (MR >= 1) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v8, %0, v0" : : "f"(a0) : "v8");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v9, %0, v1" : : "f"(a0) : "v9");
        if constexpr (NR >= 3)
            asm volatile("vfmacc.vf v10, %0, v2" : : "f"(a0) : "v10");
    }
    if constexpr (MR >= 2) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v11, %0, v0" : : "f"(a1) : "v11");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v12, %0, v1" : : "f"(a1) : "v12");
        if constexpr (NR >= 3)
            asm volatile("vfmacc.vf v13, %0, v2" : : "f"(a1) : "v13");
    }
    if constexpr (MR >= 3) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v14, %0, v0" : : "f"(a2) : "v14");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v15, %0, v1" : : "f"(a2) : "v15");
        if constexpr (NR >= 3)
            asm volatile("vfmacc.vf v16, %0, v2" : : "f"(a2) : "v16");
    }
    if constexpr (MR >= 4) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v17, %0, v0" : : "f"(a3) : "v17");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v18, %0, v1" : : "f"(a3) : "v18");
        if constexpr (NR >= 3)
            asm volatile("vfmacc.vf v19, %0, v2" : : "f"(a3) : "v19");
    }
    if constexpr (MR >= 5) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v20, %0, v0" : : "f"(a4) : "v20");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v21, %0, v1" : : "f"(a4) : "v21");
        if constexpr (NR >= 3)
            asm volatile("vfmacc.vf v22, %0, v2" : : "f"(a4) : "v22");
    }
    if constexpr (MR >= 6) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v23, %0, v0" : : "f"(a5) : "v23");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v24, %0, v1" : : "f"(a5) : "v24");
        if constexpr (NR >= 3)
            asm volatile("vfmacc.vf v25, %0, v2" : : "f"(a5) : "v25");
    }
    if constexpr (MR >= 7) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v26, %0, v0" : : "f"(a6) : "v26");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v27, %0, v1" : : "f"(a6) : "v27");
        if constexpr (NR >= 3)
            asm volatile("vfmacc.vf v28, %0, v2" : : "f"(a6) : "v28");
    }
    if constexpr (MR >= 8) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v29, %0, v0" : : "f"(a7) : "v29");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v30, %0, v1" : : "f"(a7) : "v30");
        if constexpr (NR >= 3)
            asm volatile("vfmacc.vf v31, %0, v2" : : "f"(a7) : "v31");
    }
}

template <size_t MR, size_t NR, bool CheckBounds, typename LoadWeights>
static inline void tqt_igemmcl_fma_loop_f32_f32_8x3vl_rvv(size_t IC,
                                                          const float *const input_bases[8],
                                                          size_t vl, LoadWeights load_weights)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    for (size_t ic = 0; ic < IC; ic++) {
        auto load_input = [&](size_t i) {
            if constexpr (CheckBounds) {
                return input_bases[i] ? input_bases[i][ic] : 0.0f;
            } else {
                return input_bases[i][ic];
            }
        };
        float a0 = 0.0f;
        float a1 = 0.0f;
        float a2 = 0.0f;
        float a3 = 0.0f;
        float a4 = 0.0f;
        float a5 = 0.0f;
        float a6 = 0.0f;
        float a7 = 0.0f;
        if constexpr (MR >= 1)
            a0 = load_input(0);
        if constexpr (MR >= 2)
            a1 = load_input(1);
        if constexpr (MR >= 3)
            a2 = load_input(2);
        if constexpr (MR >= 4)
            a3 = load_input(3);
        if constexpr (MR >= 5)
            a4 = load_input(4);
        if constexpr (MR >= 6)
            a5 = load_input(5);
        if constexpr (MR >= 7)
            a6 = load_input(6);
        if constexpr (MR >= 8)
            a7 = load_input(7);

        load_weights();
        tqt_vfmacc_direct_kernel_igemm_f32_f32_8x3vl_rvv<MR, NR>(vl, a0, a1, a2, a3, a4, a5, a6,
                                                                 a7);
    }
}
