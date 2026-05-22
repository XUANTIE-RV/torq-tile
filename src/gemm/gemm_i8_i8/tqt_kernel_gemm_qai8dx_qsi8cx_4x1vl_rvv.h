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
#include "src/gemm/gemm_i8_i8/tqt_kernel_gemm_i8_i8_4x1vl_rvv.h"

/// Inner kernel for qai8dx (A) x qsi8cx (B) -> f32 / f16 (D).
///
/// The K-loop differs from tqt_kernel_gemm_i8_i8_4x1vl_rvv.h: we use SEW=16
/// vwmacc.vx so that the per-row LHS zero-point correction (q_a - z_a, fits i16
/// but not i8) can be applied directly via a scalar add before the MAC.
///
/// Register allocation (K-loop phase, i.e. init_zero + load_b/load_bt + widen_b
/// + vwmacc_zp):
///   a0-a3:   per-row corrected LHS scalar (q_a - z_a)
///   v0:      raw int8 RHS row (lmul=1, e8)
///   v8-v9:   widened int16 RHS row (lmul=2, e16)
///   v16-v31: int32 accumulator matrix D[0..3][0] (4 rows x lmul=4 each)
///
/// Register allocation (dequant tail phase, i.e. dequant_fp32 / dequant_fp16):
///   v0-v3:   fp32 bias broadcast vector (lmul=4) in dequant_fp32 — REUSES
///            the v0-v3 group that previously held the raw RHS row; the K-loop
///            has finished by this point so the reuse is safe.
///   v2-v3:   fp16 bias broadcast vector (lmul=2) in dequant_fp16 (subset of
///            the same v0-v3 group reuse).
///   v4-v7:   fp32 scale_b vector (lmul=4) loaded once at the start of dequant.
///   v12-v15: scratch fp32 sigma = scale_b * scale_a[m] (lmul=4).
///   v16-v31: same accumulator group, converted in-place from i32 to fp.
///
/// The accumulator init, A-pointer setup and B loads are byte-for-byte identical
/// to the default i8/i8 4x1vl kernel; the wrappers below forward to that shared
/// implementation. Only widen_b / vwmacc_zp / dequant_* are specific to this
/// strategy.

/// Initialize accumulator registers (int32) to zero. Forwards to the shared
/// implementation in tqt_kernel_gemm_i8_i8_4x1vl_rvv.h.
template <size_t MR>
static inline void tqt_init_zero_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv(size_t vl)
{
    tqt_init_zero_kernel_gemm_i8_i8_4x1vl_rvv<MR>(vl);
}

/// Get pointers to A matrix elements for each row. Forwards to the shared
/// implementation in tqt_kernel_gemm_i8_i8_4x1vl_rvv.h.
template <size_t MR>
static inline void tqt_get_a_ptr_gemm_qai8dx_qsi8cx_4x1vl_rvv(const int8_t *A, size_t stride_row,
                                                              const int8_t *a_ptrs[MR])
{
    tqt_get_a_ptr_gemm_i8_i8_4x1vl_rvv<MR>(A, stride_row, a_ptrs);
}

/// Load B (int8) with unit stride into v0 (used by ap_bp variant). Forwards to
/// the shared implementation in tqt_kernel_gemm_i8_i8_4x1vl_rvv.h.
static inline void tqt_load_b_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv(size_t vl, const int8_t *B)
{
    tqt_load_b_kernel_gemm_i8_i8_4x1vl_rvv(vl, B);
}

/// Load B (int8) with strided access into v0 (used by a_bt variant where B is
/// [N, K]). Forwards to the shared implementation in
/// tqt_kernel_gemm_i8_i8_4x1vl_rvv.h.
static inline void tqt_load_bt_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv(size_t vl, const int8_t *B,
                                                                   size_t stride_row)
{
    tqt_load_bt_kernel_gemm_i8_i8_4x1vl_rvv(vl, B, stride_row);
}

/// Sign-extend B from int8 (v0) to int16 (v8-v9) for SEW=16 vwmacc.
static inline void tqt_widen_b_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv(size_t vl)
{
    asm volatile(
        "vsetvli zero, %0, e16, m2, ta, ma\n\t"
        "vsext.vf2 v8, v0\n\t"
        :
        : "r"(vl)
        : "v8", "v9");
}

/// SEW=16 widening multiply-accumulate with per-row zero-point correction.
/// For each row m, loads q_a from a_ptrs[m], applies (q_a + neg_zp_a[m]) in scalar GPR,
/// then vwmacc.vx into the row's i32 accumulator using the widened B in v8-v9.
template <size_t MR>
static inline void tqt_vwmacc_zp_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv(size_t vl,
                                                                     const int8_t *const a_ptrs[MR],
                                                                     const int32_t neg_zp_a[MR])
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e16, m2, ta, ma\n\t" : : "r"(vl) :);

    if constexpr (MR >= 1) {
        asm volatile(
            "lb a0, 0(%0)\n\t"
            "add a0, a0, %1\n\t"
            "vwmacc.vx v16, a0, v8\n\t"
            :
            : "r"(a_ptrs[0]), "r"(neg_zp_a[0])
            : "a0", "v16", "v17", "v18", "v19", "memory");
    }
    if constexpr (MR >= 2) {
        asm volatile(
            "lb a1, 0(%0)\n\t"
            "add a1, a1, %1\n\t"
            "vwmacc.vx v20, a1, v8\n\t"
            :
            : "r"(a_ptrs[1]), "r"(neg_zp_a[1])
            : "a1", "v20", "v21", "v22", "v23", "memory");
    }
    if constexpr (MR >= 3) {
        asm volatile(
            "lb a2, 0(%0)\n\t"
            "add a2, a2, %1\n\t"
            "vwmacc.vx v24, a2, v8\n\t"
            :
            : "r"(a_ptrs[2]), "r"(neg_zp_a[2])
            : "a2", "v24", "v25", "v26", "v27", "memory");
    }
    if constexpr (MR >= 4) {
        asm volatile(
            "lb a3, 0(%0)\n\t"
            "add a3, a3, %1\n\t"
            "vwmacc.vx v28, a3, v8\n\t"
            :
            : "r"(a_ptrs[3]), "r"(neg_zp_a[3])
            : "a3", "v28", "v29", "v30", "v31", "memory");
    }
}

/// Dequantize int32 accumulator into fp32 output, add bias, clamp, store.
/// Acc registers (v16/v20/v24/v28 base, m4) are consumed and store to D row by row.
/// scale_a[m] is per-row (M scales), scale_b is a per-channel f32 vector of length vl.
template <size_t MR>
static inline void tqt_dequant_fp32_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv(
    size_t vl, const float scale_a[MR], const float *scale_b, const float *bias, float clamp_min,
    float clamp_max, float *D, size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile(
        "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
        "vle32.v v4, (%[sb])\n\t"
        :
        : [vl] "r"(vl), [sb] "r"(scale_b)
        : "v4", "v5", "v6", "v7", "memory");

    if (bias) {
        asm volatile("vle32.v v0, (%[bs])\n\t"
                     :
                     : [bs] "r"(bias)
                     : "v0", "v1", "v2", "v3", "memory");
    } else {
        asm volatile("vmv.v.i v0, 0\n\t" : : : "v0", "v1", "v2", "v3");
    }

    if constexpr (MR >= 1) {
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
            "vfcvt.f.x.v v16, v16\n\t"
            "vfmul.vf v12, v4, %[sa]\n\t"
            "vfmul.vv v16, v16, v12\n\t"
            "vfadd.vv v16, v16, v0\n\t"
            "vfmax.vf v16, v16, %[cmin]\n\t"
            "vfmin.vf v16, v16, %[cmax]\n\t"
            "vse32.v v16, (%[d])\n\t"
            :
            : [vl] "r"(vl), [sa] "f"(scale_a[0]), [cmin] "f"(clamp_min), [cmax] "f"(clamp_max),
              [d] "r"(D + 0 * stride_row)
            : "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "memory");
    }
    if constexpr (MR >= 2) {
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
            "vfcvt.f.x.v v20, v20\n\t"
            "vfmul.vf v12, v4, %[sa]\n\t"
            "vfmul.vv v20, v20, v12\n\t"
            "vfadd.vv v20, v20, v0\n\t"
            "vfmax.vf v20, v20, %[cmin]\n\t"
            "vfmin.vf v20, v20, %[cmax]\n\t"
            "vse32.v v20, (%[d])\n\t"
            :
            : [vl] "r"(vl), [sa] "f"(scale_a[1]), [cmin] "f"(clamp_min), [cmax] "f"(clamp_max),
              [d] "r"(D + 1 * stride_row)
            : "v12", "v13", "v14", "v15", "v20", "v21", "v22", "v23", "memory");
    }
    if constexpr (MR >= 3) {
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
            "vfcvt.f.x.v v24, v24\n\t"
            "vfmul.vf v12, v4, %[sa]\n\t"
            "vfmul.vv v24, v24, v12\n\t"
            "vfadd.vv v24, v24, v0\n\t"
            "vfmax.vf v24, v24, %[cmin]\n\t"
            "vfmin.vf v24, v24, %[cmax]\n\t"
            "vse32.v v24, (%[d])\n\t"
            :
            : [vl] "r"(vl), [sa] "f"(scale_a[2]), [cmin] "f"(clamp_min), [cmax] "f"(clamp_max),
              [d] "r"(D + 2 * stride_row)
            : "v12", "v13", "v14", "v15", "v24", "v25", "v26", "v27", "memory");
    }
    if constexpr (MR >= 4) {
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
            "vfcvt.f.x.v v28, v28\n\t"
            "vfmul.vf v12, v4, %[sa]\n\t"
            "vfmul.vv v28, v28, v12\n\t"
            "vfadd.vv v28, v28, v0\n\t"
            "vfmax.vf v28, v28, %[cmin]\n\t"
            "vfmin.vf v28, v28, %[cmax]\n\t"
            "vse32.v v28, (%[d])\n\t"
            :
            : [vl] "r"(vl), [sa] "f"(scale_a[3]), [cmin] "f"(clamp_min), [cmax] "f"(clamp_max),
              [d] "r"(D + 3 * stride_row)
            : "v12", "v13", "v14", "v15", "v28", "v29", "v30", "v31", "memory");
    }
}

#if defined(__riscv_zfh) && defined(__riscv_zvfh)
/// Dequantize int32 accumulator into fp16 output: dequant in fp32, narrow,
/// then add fp16 bias and clamp in fp16 domain (consistent with existing f16 kernels).
template <size_t MR>
static inline void tqt_dequant_fp16_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv(
    size_t vl, const float scale_a[MR], const float *scale_b, const float16_t *bias,
    float16_t clamp_min, float16_t clamp_max, float16_t *D, size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile(
        "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
        "vle32.v v4, (%[sb])\n\t"
        :
        : [vl] "r"(vl), [sb] "r"(scale_b)
        : "v4", "v5", "v6", "v7", "memory");

    if (bias) {
        asm volatile(
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vle16.v v2, (%[bs])\n\t"
            :
            : [vl] "r"(vl), [bs] "r"(bias)
            : "v2", "v3", "memory");
    } else {
        asm volatile(
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vmv.v.i v2, 0\n\t"
            :
            : [vl] "r"(vl)
            : "v2", "v3");
    }

    if constexpr (MR >= 1) {
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
            "vfcvt.f.x.v v16, v16\n\t"
            "vfmul.vf v12, v4, %[sa]\n\t"
            "vfmul.vv v16, v16, v12\n\t"
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vfncvt.f.f.w v0, v16\n\t"
            "vfadd.vv v0, v0, v2\n\t"
            "vfmax.vf v0, v0, %[cmin]\n\t"
            "vfmin.vf v0, v0, %[cmax]\n\t"
            "vse16.v v0, (%[d])\n\t"
            :
            : [vl] "r"(vl), [sa] "f"(scale_a[0]), [cmin] "f"(clamp_min), [cmax] "f"(clamp_max),
              [d] "r"(D + 0 * stride_row)
            : "v0", "v1", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "memory");
    }
    if constexpr (MR >= 2) {
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
            "vfcvt.f.x.v v20, v20\n\t"
            "vfmul.vf v12, v4, %[sa]\n\t"
            "vfmul.vv v20, v20, v12\n\t"
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vfncvt.f.f.w v0, v20\n\t"
            "vfadd.vv v0, v0, v2\n\t"
            "vfmax.vf v0, v0, %[cmin]\n\t"
            "vfmin.vf v0, v0, %[cmax]\n\t"
            "vse16.v v0, (%[d])\n\t"
            :
            : [vl] "r"(vl), [sa] "f"(scale_a[1]), [cmin] "f"(clamp_min), [cmax] "f"(clamp_max),
              [d] "r"(D + 1 * stride_row)
            : "v0", "v1", "v12", "v13", "v14", "v15", "v20", "v21", "v22", "v23", "memory");
    }
    if constexpr (MR >= 3) {
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
            "vfcvt.f.x.v v24, v24\n\t"
            "vfmul.vf v12, v4, %[sa]\n\t"
            "vfmul.vv v24, v24, v12\n\t"
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vfncvt.f.f.w v0, v24\n\t"
            "vfadd.vv v0, v0, v2\n\t"
            "vfmax.vf v0, v0, %[cmin]\n\t"
            "vfmin.vf v0, v0, %[cmax]\n\t"
            "vse16.v v0, (%[d])\n\t"
            :
            : [vl] "r"(vl), [sa] "f"(scale_a[2]), [cmin] "f"(clamp_min), [cmax] "f"(clamp_max),
              [d] "r"(D + 2 * stride_row)
            : "v0", "v1", "v12", "v13", "v14", "v15", "v24", "v25", "v26", "v27", "memory");
    }
    if constexpr (MR >= 4) {
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
            "vfcvt.f.x.v v28, v28\n\t"
            "vfmul.vf v12, v4, %[sa]\n\t"
            "vfmul.vv v28, v28, v12\n\t"
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vfncvt.f.f.w v0, v28\n\t"
            "vfadd.vv v0, v0, v2\n\t"
            "vfmax.vf v0, v0, %[cmin]\n\t"
            "vfmin.vf v0, v0, %[cmax]\n\t"
            "vse16.v v0, (%[d])\n\t"
            :
            : [vl] "r"(vl), [sa] "f"(scale_a[3]), [cmin] "f"(clamp_min), [cmax] "f"(clamp_max),
              [d] "r"(D + 3 * stride_row)
            : "v0", "v1", "v12", "v13", "v14", "v15", "v28", "v29", "v30", "v31", "memory");
    }
}
#endif  // __riscv_zfh && __riscv_zvfh
