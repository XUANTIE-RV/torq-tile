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

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_f32_f32/tqt_kernel_gemm_f32_f32_8x3vl_rvv.h"

/// Igemmcf 3vlx8 kernel — additional functions for strided C/D access.
///
/// Register allocation (reuses GEMM 8x3vl layout with M/N roles swapped):
///   v0-v2:   Weight vectors A[0-2] (MR OC groups, LMUL=1)
///   ft0-ft7: Input scalars B[0-7] (NR spatial positions)
///   v8-v31:  Accumulator matrix D[nr][mr] = v[8 + nr*3 + mr]
///            nr in [0, NR-1] (spatial), mr in [0, MR-1] (OC group)
///
/// Template parameters:
///   MR = number of OC vector groups (1-3)
///   NR = number of scalar spatial positions (1-8)
///
/// Reused from tqt_kernel_gemm_f32_f32_8x3vl_rvv.h (call with swapped <NR, MR>):
///   tqt_init_zero_kernel_gemm<NR, MR>      -- zero accumulators
///   tqt_load_a_kernel_gemm<NR>             -- load NR input scalars into ft0..ft(NR-1)
///   tqt_load_b_kernel_gemm<MR>             -- load MR packed weight vectors into v0..v(MR-1)
///   tqt_load_bt_kernel_gemm<MR>            -- load MR unpacked weight vectors (strided)
///   tqt_vfmacc_kernel_gemm<NR, MR>         -- 24 FMA ops
///   tqt_add_1xnbias_kernel_gemm<NR, MR>    -- per-OC bias (vector load + broadcast add)
///   tqt_clamp_kernel_gemm<NR, MR>          -- clamp all accumulators

/// Initialize accumulator registers from C matrix with strided loads.
/// C layout: [OC, OH*OW] (NCHW), consecutive OC values are stride_c apart.
/// For each accumulator acc[nr][mr] = v[8 + nr*3 + mr]:
///   base = C + nr + mr * vlmax * stride_c
///   vlse32.v v{reg}, (base), stride_c * sizeof(float)
template <size_t MR, size_t NR>
static inline void tqt_init_c_strided_kernel_igemmcf_f32_f32_3vlx8_rvv(size_t vl, const float *C,
                                                                       size_t stride_c,
                                                                       size_t vlmax)
{
    static_assert(MR >= 1 && MR <= 3, "MR must be in range [1, 3]");
    static_assert(NR >= 1 && NR <= 8, "NR must be in range [1, 8]");

    size_t stride_bytes = stride_c * sizeof(float);
    asm volatile("vsetvli zero, %0, e32, m1, ta, ma\n\t" : : "r"(vl) :);

    // nr=0: v8, v9, v10
    if constexpr (NR >= 1) {
        const float *c_nr_0 = C + 0;
        if constexpr (MR >= 1)
            asm volatile("vlse32.v v8, (%0), %1"
                         :
                         : "r"(c_nr_0 + 0 * vlmax * stride_c), "r"(stride_bytes)
                         : "v8", "memory");
        if constexpr (MR >= 2)
            asm volatile("vlse32.v v9, (%0), %1"
                         :
                         : "r"(c_nr_0 + 1 * vlmax * stride_c), "r"(stride_bytes)
                         : "v9", "memory");
        if constexpr (MR >= 3)
            asm volatile("vlse32.v v10, (%0), %1"
                         :
                         : "r"(c_nr_0 + 2 * vlmax * stride_c), "r"(stride_bytes)
                         : "v10", "memory");
    }
    // nr=1: v11, v12, v13
    if constexpr (NR >= 2) {
        const float *c_nr_1 = C + 1;
        if constexpr (MR >= 1)
            asm volatile("vlse32.v v11, (%0), %1"
                         :
                         : "r"(c_nr_1 + 0 * vlmax * stride_c), "r"(stride_bytes)
                         : "v11", "memory");
        if constexpr (MR >= 2)
            asm volatile("vlse32.v v12, (%0), %1"
                         :
                         : "r"(c_nr_1 + 1 * vlmax * stride_c), "r"(stride_bytes)
                         : "v12", "memory");
        if constexpr (MR >= 3)
            asm volatile("vlse32.v v13, (%0), %1"
                         :
                         : "r"(c_nr_1 + 2 * vlmax * stride_c), "r"(stride_bytes)
                         : "v13", "memory");
    }
    // nr=2: v14, v15, v16
    if constexpr (NR >= 3) {
        const float *c_nr_2 = C + 2;
        if constexpr (MR >= 1)
            asm volatile("vlse32.v v14, (%0), %1"
                         :
                         : "r"(c_nr_2 + 0 * vlmax * stride_c), "r"(stride_bytes)
                         : "v14", "memory");
        if constexpr (MR >= 2)
            asm volatile("vlse32.v v15, (%0), %1"
                         :
                         : "r"(c_nr_2 + 1 * vlmax * stride_c), "r"(stride_bytes)
                         : "v15", "memory");
        if constexpr (MR >= 3)
            asm volatile("vlse32.v v16, (%0), %1"
                         :
                         : "r"(c_nr_2 + 2 * vlmax * stride_c), "r"(stride_bytes)
                         : "v16", "memory");
    }
    // nr=3: v17, v18, v19
    if constexpr (NR >= 4) {
        const float *c_nr_3 = C + 3;
        if constexpr (MR >= 1)
            asm volatile("vlse32.v v17, (%0), %1"
                         :
                         : "r"(c_nr_3 + 0 * vlmax * stride_c), "r"(stride_bytes)
                         : "v17", "memory");
        if constexpr (MR >= 2)
            asm volatile("vlse32.v v18, (%0), %1"
                         :
                         : "r"(c_nr_3 + 1 * vlmax * stride_c), "r"(stride_bytes)
                         : "v18", "memory");
        if constexpr (MR >= 3)
            asm volatile("vlse32.v v19, (%0), %1"
                         :
                         : "r"(c_nr_3 + 2 * vlmax * stride_c), "r"(stride_bytes)
                         : "v19", "memory");
    }
    // nr=4: v20, v21, v22
    if constexpr (NR >= 5) {
        const float *c_nr_4 = C + 4;
        if constexpr (MR >= 1)
            asm volatile("vlse32.v v20, (%0), %1"
                         :
                         : "r"(c_nr_4 + 0 * vlmax * stride_c), "r"(stride_bytes)
                         : "v20", "memory");
        if constexpr (MR >= 2)
            asm volatile("vlse32.v v21, (%0), %1"
                         :
                         : "r"(c_nr_4 + 1 * vlmax * stride_c), "r"(stride_bytes)
                         : "v21", "memory");
        if constexpr (MR >= 3)
            asm volatile("vlse32.v v22, (%0), %1"
                         :
                         : "r"(c_nr_4 + 2 * vlmax * stride_c), "r"(stride_bytes)
                         : "v22", "memory");
    }
    // nr=5: v23, v24, v25
    if constexpr (NR >= 6) {
        const float *c_nr_5 = C + 5;
        if constexpr (MR >= 1)
            asm volatile("vlse32.v v23, (%0), %1"
                         :
                         : "r"(c_nr_5 + 0 * vlmax * stride_c), "r"(stride_bytes)
                         : "v23", "memory");
        if constexpr (MR >= 2)
            asm volatile("vlse32.v v24, (%0), %1"
                         :
                         : "r"(c_nr_5 + 1 * vlmax * stride_c), "r"(stride_bytes)
                         : "v24", "memory");
        if constexpr (MR >= 3)
            asm volatile("vlse32.v v25, (%0), %1"
                         :
                         : "r"(c_nr_5 + 2 * vlmax * stride_c), "r"(stride_bytes)
                         : "v25", "memory");
    }
    // nr=6: v26, v27, v28
    if constexpr (NR >= 7) {
        const float *c_nr_6 = C + 6;
        if constexpr (MR >= 1)
            asm volatile("vlse32.v v26, (%0), %1"
                         :
                         : "r"(c_nr_6 + 0 * vlmax * stride_c), "r"(stride_bytes)
                         : "v26", "memory");
        if constexpr (MR >= 2)
            asm volatile("vlse32.v v27, (%0), %1"
                         :
                         : "r"(c_nr_6 + 1 * vlmax * stride_c), "r"(stride_bytes)
                         : "v27", "memory");
        if constexpr (MR >= 3)
            asm volatile("vlse32.v v28, (%0), %1"
                         :
                         : "r"(c_nr_6 + 2 * vlmax * stride_c), "r"(stride_bytes)
                         : "v28", "memory");
    }
    // nr=7: v29, v30, v31
    if constexpr (NR >= 8) {
        const float *c_nr_7 = C + 7;
        if constexpr (MR >= 1)
            asm volatile("vlse32.v v29, (%0), %1"
                         :
                         : "r"(c_nr_7 + 0 * vlmax * stride_c), "r"(stride_bytes)
                         : "v29", "memory");
        if constexpr (MR >= 2)
            asm volatile("vlse32.v v30, (%0), %1"
                         :
                         : "r"(c_nr_7 + 1 * vlmax * stride_c), "r"(stride_bytes)
                         : "v30", "memory");
        if constexpr (MR >= 3)
            asm volatile("vlse32.v v31, (%0), %1"
                         :
                         : "r"(c_nr_7 + 2 * vlmax * stride_c), "r"(stride_bytes)
                         : "v31", "memory");
    }
}

/// Store accumulator registers to D matrix with strided stores.
/// D layout: [OC, OH*OW] (NCHW), consecutive OC values are stride_d apart.
/// For each accumulator acc[nr][mr] = v[8 + nr*3 + mr]:
///   base = D + nr + mr * vlmax * stride_d
///   vsse32.v v{reg}, (base), stride_d * sizeof(float)
template <size_t MR, size_t NR>
static inline void tqt_store_strided_kernel_igemmcf_f32_f32_3vlx8_rvv(size_t vl, float *D,
                                                                      size_t stride_d, size_t vlmax)
{
    static_assert(MR >= 1 && MR <= 3, "MR must be in range [1, 3]");
    static_assert(NR >= 1 && NR <= 8, "NR must be in range [1, 8]");

    size_t stride_bytes = stride_d * sizeof(float);
    asm volatile("vsetvli zero, %0, e32, m1, ta, ma\n\t" : : "r"(vl) :);

    // nr=0: v8, v9, v10
    if constexpr (NR >= 1) {
        float *d_nr_0 = D + 0;
        if constexpr (MR >= 1)
            asm volatile("vsse32.v v8, (%0), %1"
                         :
                         : "r"(d_nr_0 + 0 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
        if constexpr (MR >= 2)
            asm volatile("vsse32.v v9, (%0), %1"
                         :
                         : "r"(d_nr_0 + 1 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
        if constexpr (MR >= 3)
            asm volatile("vsse32.v v10, (%0), %1"
                         :
                         : "r"(d_nr_0 + 2 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
    }
    // nr=1: v11, v12, v13
    if constexpr (NR >= 2) {
        float *d_nr_1 = D + 1;
        if constexpr (MR >= 1)
            asm volatile("vsse32.v v11, (%0), %1"
                         :
                         : "r"(d_nr_1 + 0 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
        if constexpr (MR >= 2)
            asm volatile("vsse32.v v12, (%0), %1"
                         :
                         : "r"(d_nr_1 + 1 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
        if constexpr (MR >= 3)
            asm volatile("vsse32.v v13, (%0), %1"
                         :
                         : "r"(d_nr_1 + 2 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
    }
    // nr=2: v14, v15, v16
    if constexpr (NR >= 3) {
        float *d_nr_2 = D + 2;
        if constexpr (MR >= 1)
            asm volatile("vsse32.v v14, (%0), %1"
                         :
                         : "r"(d_nr_2 + 0 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
        if constexpr (MR >= 2)
            asm volatile("vsse32.v v15, (%0), %1"
                         :
                         : "r"(d_nr_2 + 1 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
        if constexpr (MR >= 3)
            asm volatile("vsse32.v v16, (%0), %1"
                         :
                         : "r"(d_nr_2 + 2 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
    }
    // nr=3: v17, v18, v19
    if constexpr (NR >= 4) {
        float *d_nr_3 = D + 3;
        if constexpr (MR >= 1)
            asm volatile("vsse32.v v17, (%0), %1"
                         :
                         : "r"(d_nr_3 + 0 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
        if constexpr (MR >= 2)
            asm volatile("vsse32.v v18, (%0), %1"
                         :
                         : "r"(d_nr_3 + 1 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
        if constexpr (MR >= 3)
            asm volatile("vsse32.v v19, (%0), %1"
                         :
                         : "r"(d_nr_3 + 2 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
    }
    // nr=4: v20, v21, v22
    if constexpr (NR >= 5) {
        float *d_nr_4 = D + 4;
        if constexpr (MR >= 1)
            asm volatile("vsse32.v v20, (%0), %1"
                         :
                         : "r"(d_nr_4 + 0 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
        if constexpr (MR >= 2)
            asm volatile("vsse32.v v21, (%0), %1"
                         :
                         : "r"(d_nr_4 + 1 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
        if constexpr (MR >= 3)
            asm volatile("vsse32.v v22, (%0), %1"
                         :
                         : "r"(d_nr_4 + 2 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
    }
    // nr=5: v23, v24, v25
    if constexpr (NR >= 6) {
        float *d_nr_5 = D + 5;
        if constexpr (MR >= 1)
            asm volatile("vsse32.v v23, (%0), %1"
                         :
                         : "r"(d_nr_5 + 0 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
        if constexpr (MR >= 2)
            asm volatile("vsse32.v v24, (%0), %1"
                         :
                         : "r"(d_nr_5 + 1 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
        if constexpr (MR >= 3)
            asm volatile("vsse32.v v25, (%0), %1"
                         :
                         : "r"(d_nr_5 + 2 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
    }
    // nr=6: v26, v27, v28
    if constexpr (NR >= 7) {
        float *d_nr_6 = D + 6;
        if constexpr (MR >= 1)
            asm volatile("vsse32.v v26, (%0), %1"
                         :
                         : "r"(d_nr_6 + 0 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
        if constexpr (MR >= 2)
            asm volatile("vsse32.v v27, (%0), %1"
                         :
                         : "r"(d_nr_6 + 1 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
        if constexpr (MR >= 3)
            asm volatile("vsse32.v v28, (%0), %1"
                         :
                         : "r"(d_nr_6 + 2 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
    }
    // nr=7: v29, v30, v31
    if constexpr (NR >= 8) {
        float *d_nr_7 = D + 7;
        if constexpr (MR >= 1)
            asm volatile("vsse32.v v29, (%0), %1"
                         :
                         : "r"(d_nr_7 + 0 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
        if constexpr (MR >= 2)
            asm volatile("vsse32.v v30, (%0), %1"
                         :
                         : "r"(d_nr_7 + 1 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
        if constexpr (MR >= 3)
            asm volatile("vsse32.v v31, (%0), %1"
                         :
                         : "r"(d_nr_7 + 2 * vlmax * stride_d), "r"(stride_bytes)
                         : "memory");
    }
}
