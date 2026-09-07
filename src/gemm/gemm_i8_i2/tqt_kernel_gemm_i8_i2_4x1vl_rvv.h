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

// Inner kernel for the gemm_i8_i2 family on RVV.
//
// Accumulator: int32 (m4) per row. The K loop runs the integer dot product
//   iacc[m][n] += q_A[m][k] * dec_B[n][k]
// where dec_B is the int8 representation of the 2-bit B weight (range [-2, 1]).
// Zero-point correction `iacc -= zp_a[m] * sum_b[n]` and the final scale/bias/clamp
// stage are applied in the K-loop epilogue. See gemm_1xnbias_clamp_f32_qai8dx*qsi2cx*
// kernel-bases for the public entry points.
//
// Register allocation:
//   a0-a3:    A row scalars
//   v0:       raw packed-byte loaded from B (e8 m1)
//   v1:       decoded int8 value for the current K slot (e8 m1)
//   v8-v9:    sign-extended i16 view of v1 consumed by vwmacc (e16 m2)
//   v16-v31:  int32 accumulators D[0..3] (m4 each)
//   v0-v7:    f32 epilogue temporaries (after K loop ends)

/// Initialize accumulator registers (int32) to zero.
template <size_t MR>
static inline void tqt_init_zero_kernel_gemm_i8_i2_4x1vl_rvv(size_t vl)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    if constexpr (MR >= 1)
        asm volatile("vmv.v.i v16, 0\n\t" : : : "v16", "v17", "v18", "v19");
    if constexpr (MR >= 2)
        asm volatile("vmv.v.i v20, 0\n\t" : : : "v20", "v21", "v22", "v23");
    if constexpr (MR >= 3)
        asm volatile("vmv.v.i v24, 0\n\t" : : : "v24", "v25", "v26", "v27");
    if constexpr (MR >= 4)
        asm volatile("vmv.v.i v28, 0\n\t" : : : "v28", "v29", "v30", "v31");
}

/// Get pointers to A matrix elements for each row.
template <size_t MR>
static inline void tqt_get_a_ptr_gemm_i8_i2_4x1vl_rvv(const int8_t *A, size_t stride_row,
                                                      const int8_t *a_ptrs[MR])
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    if constexpr (MR >= 1)
        a_ptrs[0] = A + 0 * stride_row;
    if constexpr (MR >= 2)
        a_ptrs[1] = A + 1 * stride_row;
    if constexpr (MR >= 3)
        a_ptrs[2] = A + 2 * stride_row;
    if constexpr (MR >= 4)
        a_ptrs[3] = A + 3 * stride_row;
}

/// Load one byte per N-column into v0 (e8 m1) from packed B (non-transposed / contiguous).
/// Used by the ap_bp variant where B is laid out as [N/vl, K/4, vl] uint8.
static inline void tqt_load_b_byte_kernel_gemm_i8_i2_4x1vl_rvv(size_t vl, const uint8_t *B)
{
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vle8.v v0, (%0)" : : "r"(B) : "v0", "memory");
}

/// Load one byte per N-column into v0 (e8 m1) from B[n, k/4] (transposed layout).
/// Used by the a_bt variant where B is laid out as [N, K/4] uint8 row-major.
/// stride_row is in bytes (= K/4).
static inline void tqt_load_bt_byte_kernel_gemm_i8_i2_4x1vl_rvv(size_t vl, const uint8_t *B,
                                                                size_t stride_row)
{
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vlse8.v v0, (%0), %1" : : "r"(B), "r"(stride_row) : "v0", "memory");
}

/// Decode one 2-bit slot from the byte in v0 into v1 as int8 ∈ [-2, 1].
///
/// Default LUT path: equivalent to `(u2 - 2)` because `lut_default = {-2,-1,0,1}`.
/// SLOT must be a compile-time constant in [0, 3].
/// SET_VL may be false only when the caller already configured e8 m1 with the same vl.
template <size_t SLOT, bool SET_VL = true>
static inline void tqt_decode_b_default_lut_kernel_gemm_i8_i2_4x1vl_rvv(size_t vl)
{
    static_assert(SLOT <= 3, "SLOT must be in range [0, 3]");

    if constexpr (SET_VL)
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t" : : "r"(vl) :);

    if constexpr (SLOT == 0) {
        asm volatile(
            "vand.vi v1, v0, 0x3\n\t"
            "vadd.vi v1, v1, -2\n\t"
            :
            :
            : "v1");
    } else {
        asm volatile(
            "vsrl.vi v1, v0, %c[s]\n\t"
            "vand.vi v1, v1, 0x3\n\t"
            "vadd.vi v1, v1, -2\n\t"
            :
            : [s] "i"(SLOT * 2)
            : "v1");
    }
}

/// Decode one 2-bit slot from the byte in v0 into v1 as int8 via a vrgather table.
///
/// `v_lut` should be preloaded by the caller with the 4-entry LUT in its first four lanes.
/// LUT_VREG must match the register that holds the LUT vector (e.g. "v2").
/// SET_VL may be false only when the caller already configured e8 m1 with the same vl.
template <size_t SLOT, bool SET_VL = true>
static inline void tqt_decode_b_custom_lut_kernel_gemm_i8_i2_4x1vl_rvv(size_t vl)
{
    static_assert(SLOT <= 3, "SLOT must be in range [0, 3]");

    if constexpr (SET_VL)
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t" : : "r"(vl) :);

    if constexpr (SLOT == 0) {
        asm volatile(
            "vand.vi v3, v0, 0x3\n\t"
            "vrgather.vv v1, v2, v3\n\t"
            :
            :
            : "v1", "v3");
    } else {
        asm volatile(
            "vsrl.vi v3, v0, %c[s]\n\t"
            "vand.vi v3, v3, 0x3\n\t"
            "vrgather.vv v1, v2, v3\n\t"
            :
            : [s] "i"(SLOT * 2)
            : "v1", "v3");
    }
}

/// Load a 4-entry LUT into v2 (e8 m1).
///
/// Only the first 4 lanes of v2 need to be populated because all `vrgather.vv` indices are
/// guaranteed to be in [0, 3] by the preceding `vand.vi 0x3` mask.
static inline void tqt_load_custom_lut_kernel_gemm_i8_i2_4x1vl_rvv(size_t vl, const int8_t *lut)
{
    const size_t lut_entries = 4;
    asm volatile(
        "vsetvli zero, %[lut_entries], e8, m1, ta, ma\n\t"
        "vle8.v v2, (%[lut])\n\t"
        "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
        :
        : [lut_entries] "r"(lut_entries), [lut] "r"(lut), [vl] "r"(vl)
        : "v2", "memory");
}

/// Widening multiply-accumulate: i32_acc[r] += q_A[r] * dec_B.
///
/// Sign-extends the decoded int8 B vector in v1 to i16 in v8/v9 once, then issues
/// `vwmacc.vx v_acc, a_r, v8` per row to accumulate (i8 × i16 → i32) into
/// v16/v20/v24/v28. Reads 1 byte per row from `a_ptrs[r]`. Assumes v1 holds the
/// decoded int8 B vector for the current K position.
template <size_t MR>
static inline void tqt_vmacc_kernel_gemm_i8_i2_4x1vl_rvv(size_t vl, const int8_t *const a_ptrs[MR])
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e16, m2, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vsext.vf2 v8, v1" : : : "v8", "v9");

    if constexpr (MR >= 1) {
        asm volatile("lb a0, 0(%0)" : : "r"(a_ptrs[0]) : "a0", "memory");
        asm volatile("vwmacc.vx v16, a0, v8" : : : "v16", "v17", "v18", "v19");
    }
    if constexpr (MR >= 2) {
        asm volatile("lb a1, 0(%0)" : : "r"(a_ptrs[1]) : "a1", "memory");
        asm volatile("vwmacc.vx v20, a1, v8" : : : "v20", "v21", "v22", "v23");
    }
    if constexpr (MR >= 3) {
        asm volatile("lb a2, 0(%0)" : : "r"(a_ptrs[2]) : "a2", "memory");
        asm volatile("vwmacc.vx v24, a2, v8" : : : "v24", "v25", "v26", "v27");
    }
    if constexpr (MR >= 4) {
        asm volatile("lb a3, 0(%0)" : : "r"(a_ptrs[3]) : "a3", "memory");
        asm volatile("vwmacc.vx v28, a3, v8" : : : "v28", "v29", "v30", "v31");
    }
}

/// Apply zero-point correction to the int32 accumulators:
///   iacc[r][:] += (-zero_point_a[r]) * sum_b[:]
/// `sum_b_ptr` must point to vl int32 values for the current N-tile.
template <size_t MR>
static inline void tqt_apply_zp_correction_kernel_gemm_i8_i2_4x1vl_rvv(size_t vl,
                                                                       const int32_t *sum_b_ptr,
                                                                       const int32_t *neg_zp_a)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vle32.v v8, (%0)" : : "r"(sum_b_ptr) : "v8", "v9", "v10", "v11", "memory");

    if constexpr (MR >= 1)
        asm volatile("vmacc.vx v16, %0, v8" : : "r"(neg_zp_a[0]) : "v16", "v17", "v18", "v19");
    if constexpr (MR >= 2)
        asm volatile("vmacc.vx v20, %0, v8" : : "r"(neg_zp_a[1]) : "v20", "v21", "v22", "v23");
    if constexpr (MR >= 3)
        asm volatile("vmacc.vx v24, %0, v8" : : "r"(neg_zp_a[2]) : "v24", "v25", "v26", "v27");
    if constexpr (MR >= 4)
        asm volatile("vmacc.vx v28, %0, v8" : : "r"(neg_zp_a[3]) : "v28", "v29", "v30", "v31");
}

/// Convert int32 accumulators to fp32 and apply combined per-row × per-channel scale.
///   acc_f32[r][n] = (float)iacc[r][n] * scale_a[r] * scale_b[n]
/// Result lives in v16/v20/v24/v28 as fp32 (m4).
template <size_t MR>
static inline void tqt_dequant_to_f32_kernel_gemm_i8_i2_4x1vl_rvv(size_t vl, const float *scale_a,
                                                                  const float *scale_b_ptr)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vle32.v v0, (%0)" : : "r"(scale_b_ptr) : "v0", "v1", "v2", "v3", "memory");

    if constexpr (MR >= 1) {
        asm volatile("vfcvt.f.x.v v16, v16" : : : "v16", "v17", "v18", "v19");
        asm volatile("vfmul.vf v16, v16, %0" : : "f"(scale_a[0]) : "v16", "v17", "v18", "v19");
        asm volatile("vfmul.vv v16, v16, v0" : : : "v16", "v17", "v18", "v19");
    }
    if constexpr (MR >= 2) {
        asm volatile("vfcvt.f.x.v v20, v20" : : : "v20", "v21", "v22", "v23");
        asm volatile("vfmul.vf v20, v20, %0" : : "f"(scale_a[1]) : "v20", "v21", "v22", "v23");
        asm volatile("vfmul.vv v20, v20, v0" : : : "v20", "v21", "v22", "v23");
    }
    if constexpr (MR >= 3) {
        asm volatile("vfcvt.f.x.v v24, v24" : : : "v24", "v25", "v26", "v27");
        asm volatile("vfmul.vf v24, v24, %0" : : "f"(scale_a[2]) : "v24", "v25", "v26", "v27");
        asm volatile("vfmul.vv v24, v24, v0" : : : "v24", "v25", "v26", "v27");
    }
    if constexpr (MR >= 4) {
        asm volatile("vfcvt.f.x.v v28, v28" : : : "v28", "v29", "v30", "v31");
        asm volatile("vfmul.vf v28, v28, %0" : : "f"(scale_a[3]) : "v28", "v29", "v30", "v31");
        asm volatile("vfmul.vv v28, v28, v0" : : : "v28", "v29", "v30", "v31");
    }
}

/// Add 1xN fp32 bias to the fp32 accumulators.
template <size_t MR>
static inline void tqt_add_1xnbias_f32_kernel_gemm_i8_i2_4x1vl_rvv(size_t vl, const float *bias)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vle32.v v4, (%0)" : : "r"(bias) : "v4", "v5", "v6", "v7", "memory");

    if constexpr (MR >= 1)
        asm volatile("vfadd.vv v16, v16, v4" : : : "v16", "v17", "v18", "v19");
    if constexpr (MR >= 2)
        asm volatile("vfadd.vv v20, v20, v4" : : : "v20", "v21", "v22", "v23");
    if constexpr (MR >= 3)
        asm volatile("vfadd.vv v24, v24, v4" : : : "v24", "v25", "v26", "v27");
    if constexpr (MR >= 4)
        asm volatile("vfadd.vv v28, v28, v4" : : : "v28", "v29", "v30", "v31");
}

/// Clamp fp32 accumulators to [min_val, max_val].
template <size_t MR>
static inline void tqt_clamp_f32_kernel_gemm_i8_i2_4x1vl_rvv(size_t vl, float min_val,
                                                             float max_val)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    if constexpr (MR >= 1) {
        asm volatile("vfmax.vf v16, v16, %0" : : "f"(min_val) : "v16", "v17", "v18", "v19");
        asm volatile("vfmin.vf v16, v16, %0" : : "f"(max_val) : "v16", "v17", "v18", "v19");
    }
    if constexpr (MR >= 2) {
        asm volatile("vfmax.vf v20, v20, %0" : : "f"(min_val) : "v20", "v21", "v22", "v23");
        asm volatile("vfmin.vf v20, v20, %0" : : "f"(max_val) : "v20", "v21", "v22", "v23");
    }
    if constexpr (MR >= 3) {
        asm volatile("vfmax.vf v24, v24, %0" : : "f"(min_val) : "v24", "v25", "v26", "v27");
        asm volatile("vfmin.vf v24, v24, %0" : : "f"(max_val) : "v24", "v25", "v26", "v27");
    }
    if constexpr (MR >= 4) {
        asm volatile("vfmax.vf v28, v28, %0" : : "f"(min_val) : "v28", "v29", "v30", "v31");
        asm volatile("vfmin.vf v28, v28, %0" : : "f"(max_val) : "v28", "v29", "v30", "v31");
    }
}

/// Store fp32 accumulators (v16/v20/v24/v28) to the destination.
template <size_t MR>
static inline void tqt_store_f32_kernel_gemm_i8_i2_4x1vl_rvv(size_t vl, float *D, size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    if constexpr (MR >= 1)
        asm volatile("vse32.v v16, (%0)" : : "r"(D + 0 * stride_row) : "memory");
    if constexpr (MR >= 2)
        asm volatile("vse32.v v20, (%0)" : : "r"(D + 1 * stride_row) : "memory");
    if constexpr (MR >= 3)
        asm volatile("vse32.v v24, (%0)" : : "r"(D + 2 * stride_row) : "memory");
    if constexpr (MR >= 4)
        asm volatile("vse32.v v28, (%0)" : : "r"(D + 3 * stride_row) : "memory");
}
