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

#include "src/common/tqt_common.h"

// ============================================================================
// Shared int8-output requantize epilogue for the 4x1vl W4A8 GEMM kernels.
//
// Consumes the f32 dequantized accumulators left in v16/v20/v24/v28 (e32, m4)
// by tqt_dequant_acc_to_f32_kernel_gemm_i8_i4_4x1vl_rvv (after the optional
// f32 C / 1xN bias adds), and requantizes each row to int8:
//
//   D_i8 = clamp( round(D_real * inv_scale_d) + zero_point_d,
//                 clamp_min, clamp_max )                       (-> int8)
//
// inv_scale_d = 1.0f / scale_d is precomputed on the host so the hot path uses
// vfmul.vf (not a divide). zero_point_d / clamp_min / clamp_max arrive as int32
// scalars (the int8 clamp bounds are sign-extended at the call site so the
// vmax.vx / vmin.vx integer clamp operates in i32). Rounding uses the dynamic
// rounding mode (RNE by default), matching a std::nearbyintf golden reference.
//
// Register usage: writes the int8 result to v4 (row0) / v5 (row1) / v6 (row2) /
// v7 (row3) and uses v12/v13 as the e16/m2 narrow intermediate. By the time
// this runs, v4-v7 (rsum_b) and v8-v11 (scale_b) are dead scratch, so nothing
// live is clobbered. The narrow dance strictly walks e32/m4 -> e16/m2 -> e8/m1
// and each per-row block re-establishes e32/m4 at its top.
// ============================================================================

template <size_t MR>
static inline void tqt_requantize_f32_to_i8_kernel_gemm_i8_i4_4x1vl_rvv(
    size_t vl, float inv_scale_d, int32_t zero_point_d, int32_t clamp_min, int32_t clamp_max,
    int8_t *D, size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    if constexpr (MR >= 1) {
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
            "vfmul.vf v16, v16, %[inv]\n\t"  // D_real * (1/scale_d)
            "vfcvt.x.f.v v16, v16\n\t"       // f32 -> i32 (round-to-nearest)
            "vadd.vx  v16, v16, %[z]\n\t"    // + zero_point_d
            "vmax.vx  v16, v16, %[lo]\n\t"   // clamp low  (i32 domain)
            "vmin.vx  v16, v16, %[hi]\n\t"   // clamp high (i32 domain)
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vnclip.wx v12, v16, zero\n\t"  // i32 -> i16 (signed saturate)
            "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
            "vnclip.wx v4, v12, zero\n\t"  // i16 -> i8 (signed saturate)
            "vse8.v   v4, (%[d])\n\t"
            :
            : [vl] "r"(vl), [inv] "f"(inv_scale_d), [z] "r"(zero_point_d), [lo] "r"(clamp_min),
              [hi] "r"(clamp_max), [d] "r"(D + 0 * stride_row)
            : "v4", "v12", "v13", "v16", "v17", "v18", "v19", "memory");
    }
    if constexpr (MR >= 2) {
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
            "vfmul.vf v20, v20, %[inv]\n\t"
            "vfcvt.x.f.v v20, v20\n\t"
            "vadd.vx  v20, v20, %[z]\n\t"
            "vmax.vx  v20, v20, %[lo]\n\t"
            "vmin.vx  v20, v20, %[hi]\n\t"
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vnclip.wx v12, v20, zero\n\t"
            "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
            "vnclip.wx v5, v12, zero\n\t"
            "vse8.v   v5, (%[d])\n\t"
            :
            : [vl] "r"(vl), [inv] "f"(inv_scale_d), [z] "r"(zero_point_d), [lo] "r"(clamp_min),
              [hi] "r"(clamp_max), [d] "r"(D + 1 * stride_row)
            : "v5", "v12", "v13", "v20", "v21", "v22", "v23", "memory");
    }
    if constexpr (MR >= 3) {
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
            "vfmul.vf v24, v24, %[inv]\n\t"
            "vfcvt.x.f.v v24, v24\n\t"
            "vadd.vx  v24, v24, %[z]\n\t"
            "vmax.vx  v24, v24, %[lo]\n\t"
            "vmin.vx  v24, v24, %[hi]\n\t"
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vnclip.wx v12, v24, zero\n\t"
            "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
            "vnclip.wx v6, v12, zero\n\t"
            "vse8.v   v6, (%[d])\n\t"
            :
            : [vl] "r"(vl), [inv] "f"(inv_scale_d), [z] "r"(zero_point_d), [lo] "r"(clamp_min),
              [hi] "r"(clamp_max), [d] "r"(D + 2 * stride_row)
            : "v6", "v12", "v13", "v24", "v25", "v26", "v27", "memory");
    }
    if constexpr (MR >= 4) {
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
            "vfmul.vf v28, v28, %[inv]\n\t"
            "vfcvt.x.f.v v28, v28\n\t"
            "vadd.vx  v28, v28, %[z]\n\t"
            "vmax.vx  v28, v28, %[lo]\n\t"
            "vmin.vx  v28, v28, %[hi]\n\t"
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vnclip.wx v12, v28, zero\n\t"
            "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
            "vnclip.wx v7, v12, zero\n\t"
            "vse8.v   v7, (%[d])\n\t"
            :
            : [vl] "r"(vl), [inv] "f"(inv_scale_d), [z] "r"(zero_point_d), [lo] "r"(clamp_min),
              [hi] "r"(clamp_max), [d] "r"(D + 3 * stride_row)
            : "v7", "v12", "v13", "v28", "v29", "v30", "v31", "memory");
    }
}
