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

/// Register allocation for 4x1vl W4A8 GEMM kernel:
///   v0:      B lo nibbles (signed i8, m1) — current K iteration's even k
///   v1:      B hi nibbles (signed i8, m1) — current K iteration's odd k
///   v2:      B raw byte load destination (uint8, m1) — overwritten each K iter
///   v3:      reserved temporary
///   v4-v7:   B column-sum accumulator (i32, m4) — running rsum_b for the
///            current N tile, accumulated inside the K loop and consumed by
///            tqt_apply_zp_correction_kernel
///   v8-v15:  Widening multiply temporaries (i16, m2)
///   v16-v19: Accumulator row 0 (i32, m4)
///   v20-v23: Accumulator row 1 (i32, m4)
///   v24-v27: Accumulator row 2 (i32, m4)
///   v28-v31: Accumulator row 3 (i32, m4)
///
/// Internal accumulator is i32. The K loop body folds the B nibble sum into
/// v4-v7 in lockstep with the main vwmul/vwadd, so there is no pre-pass over
/// B and no scratch buffer for rsum_b. After the K loop, the epilogue:
///   1. Subtracts zp_a[m] * v_rsum_b (cross-quant correction; v_rsum_b lives
///      in v4-v7).
///   2. Multiplies by scale_a[m] * scale_b[n] (dequant to f32).
///   3. Adds bias[n] (in fp32 space, broadcast across rows).
///   4. Adds C[m, n] (in fp32 space, optional).
///   5. Clamps in fp32 space.
///   6. Narrows to f16 or bf16 (or stores as f32) and stores to D.

// ============================================================================
// Initialize accumulators to zero (i32, m4)
// ============================================================================

template <size_t MR>
static inline void tqt_init_zero_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl)
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

// ============================================================================
// Dequantize B for transposed (qsi4cxt) layout.
//
// Loads `vl` bytes via stride-load (one byte per N row), splits each byte into
// two signed nibbles, and stores them as signed i8 vectors:
//   v0 = B[n_idx:n_idx+vl, k+0]  (signed in [-8, 7])
//   v1 = B[n_idx:n_idx+vl, k+1]  (signed in [-8, 7])
//
// Encoding: each byte = ((b_hi+8) << 4) | (b_lo+8), so we extract nibbles,
// zero-extend to wider, subtract 8 to recover signed values, and narrow back
// to i8 (since values fit in [-8, 7]).
// ============================================================================

static inline void tqt_dequant_bt_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl, const uint8_t *b_col,
                                                              size_t b_byte_stride)
{
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vlse8.v v2, (%0), %1" : : "r"(b_col), "r"(b_byte_stride) : "v2", "memory");

    // Extract nibbles into v0 (lo) and v1 (hi), then convert to signed [-8, 7].
    // Use vand/vsrl for unsigned extraction, then vadd with -8 (5-bit immediate).
    asm volatile(
        "vand.vi v0, v2, 0xf\n\t"
        "vsrl.vi v1, v2, 4\n\t"
        "vadd.vi v0, v0, -8\n\t"
        "vadd.vi v1, v1, -8\n\t"
        :
        :
        : "v0", "v1");
}

// ============================================================================
// Dequantize B for packed (qsi4cxp) layout.
//
// Same as dequant_bt but contiguous load (no stride). Used when the packed
// layout has placed `vl` bytes of N-tile data contiguously for each k-pair.
// ============================================================================

static inline void tqt_dequant_b_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl, const uint8_t *b_col)
{
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vle8.v v2, (%0)" : : "r"(b_col) : "v2", "memory");

    asm volatile(
        "vand.vi v0, v2, 0xf\n\t"
        "vsrl.vi v1, v2, 4\n\t"
        "vadd.vi v0, v0, -8\n\t"
        "vadd.vi v1, v1, -8\n\t"
        :
        :
        : "v0", "v1");
}

// ============================================================================
// Initialize the running B column-sum accumulator (v4-v7, i32 m4) to zero.
// Must be called once per N tile, before entering the K loop.
// ============================================================================

static inline void tqt_init_rsum_b_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl)
{
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vmv.v.i v4, 0\n\t" : : : "v4", "v5", "v6", "v7");
}

// ============================================================================
// Fold the current K iteration's B nibble pair into the column-sum accumulator.
//
// Reads the just-dequantized signed-nibble vectors v0 (lo, k+0) and v1 (hi,
// k+1), both i8/m1. Computes (v0 + v1) widened to i16/m2 in v8-v9, then
// sign-extends and adds into the i32/m4 running sum in v4-v7.
//
// Must be called once per K-pair iteration, after the corresponding
// tqt_dequant_b{,t}_kernel and any time before v0/v1 are overwritten.
// ============================================================================

static inline void tqt_accum_rsum_b_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl)
{
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vwadd.vv v8, v0, v1\n\t" : : : "v8", "v9");

    asm volatile("vsetvli zero, %0, e16, m2, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vwadd.wv v4, v4, v8\n\t" : : : "v4", "v5", "v6", "v7");
}

// ============================================================================
// FMA with lo nibble (k+0): a_scalars × v0 → i16 → i32 acc
// ============================================================================

template <size_t MR>
static inline void tqt_vmacc_lo_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl,
                                                            const int8_t *const a_ptrs[MR])
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t" : : "r"(vl) :);
    if constexpr (MR >= 1) {
        asm volatile("lb a0, 0(%0)" : : "r"(a_ptrs[0]) : "a0", "memory");
        asm volatile("vwmul.vx v8, v0, a0" : : : "v8", "v9");
    }
    if constexpr (MR >= 2) {
        asm volatile("lb a1, 0(%0)" : : "r"(a_ptrs[1]) : "a1", "memory");
        asm volatile("vwmul.vx v10, v0, a1" : : : "v10", "v11");
    }
    if constexpr (MR >= 3) {
        asm volatile("lb a2, 0(%0)" : : "r"(a_ptrs[2]) : "a2", "memory");
        asm volatile("vwmul.vx v12, v0, a2" : : : "v12", "v13");
    }
    if constexpr (MR >= 4) {
        asm volatile("lb a3, 0(%0)" : : "r"(a_ptrs[3]) : "a3", "memory");
        asm volatile("vwmul.vx v14, v0, a3" : : : "v14", "v15");
    }

    asm volatile("vsetvli zero, %0, e16, m2, ta, ma\n\t" : : "r"(vl) :);
    if constexpr (MR >= 1)
        asm volatile("vwadd.wv v16, v16, v8" : : : "v16", "v17", "v18", "v19");
    if constexpr (MR >= 2)
        asm volatile("vwadd.wv v20, v20, v10" : : : "v20", "v21", "v22", "v23");
    if constexpr (MR >= 3)
        asm volatile("vwadd.wv v24, v24, v12" : : : "v24", "v25", "v26", "v27");
    if constexpr (MR >= 4)
        asm volatile("vwadd.wv v28, v28, v14" : : : "v28", "v29", "v30", "v31");
}

// ============================================================================
// FMA with hi nibble (k+1): a_scalars × v1 → i16 → i32 acc
// ============================================================================

template <size_t MR>
static inline void tqt_vmacc_hi_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl,
                                                            const int8_t *const a_ptrs[MR])
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t" : : "r"(vl) :);
    if constexpr (MR >= 1) {
        asm volatile("lb a0, 0(%0)" : : "r"(a_ptrs[0]) : "a0", "memory");
        asm volatile("vwmul.vx v8, v1, a0" : : : "v8", "v9");
    }
    if constexpr (MR >= 2) {
        asm volatile("lb a1, 0(%0)" : : "r"(a_ptrs[1]) : "a1", "memory");
        asm volatile("vwmul.vx v10, v1, a1" : : : "v10", "v11");
    }
    if constexpr (MR >= 3) {
        asm volatile("lb a2, 0(%0)" : : "r"(a_ptrs[2]) : "a2", "memory");
        asm volatile("vwmul.vx v12, v1, a2" : : : "v12", "v13");
    }
    if constexpr (MR >= 4) {
        asm volatile("lb a3, 0(%0)" : : "r"(a_ptrs[3]) : "a3", "memory");
        asm volatile("vwmul.vx v14, v1, a3" : : : "v14", "v15");
    }

    asm volatile("vsetvli zero, %0, e16, m2, ta, ma\n\t" : : "r"(vl) :);
    if constexpr (MR >= 1)
        asm volatile("vwadd.wv v16, v16, v8" : : : "v16", "v17", "v18", "v19");
    if constexpr (MR >= 2)
        asm volatile("vwadd.wv v20, v20, v10" : : : "v20", "v21", "v22", "v23");
    if constexpr (MR >= 3)
        asm volatile("vwadd.wv v24, v24, v12" : : : "v24", "v25", "v26", "v27");
    if constexpr (MR >= 4)
        asm volatile("vwadd.wv v28, v28, v14" : : : "v28", "v29", "v30", "v31");
}

// ============================================================================
// Apply zero-point-A correction: acc -= zp_a[m] * v_rsum_b
//
// Consumes the running B column-sum accumulator in v4-v7 (i32, m4), which was
// folded inside the K loop by tqt_accum_rsum_b_kernel. Multiplies by zp_a
// scalar per row (v8-v11 tmp) and subtracts from each accumulator row.
// ============================================================================

template <size_t MR>
static inline void tqt_apply_zp_correction_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl,
                                                                       const int32_t *zp_a_arr)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    if constexpr (MR >= 1) {
        // acc[0] -= zp_a[0] * v_rsum_b
        asm volatile("vmul.vx v8, v4, %0" : : "r"(zp_a_arr[0]) : "v8", "v9", "v10", "v11");
        asm volatile("vsub.vv v16, v16, v8" : : : "v16", "v17", "v18", "v19");
    }
    if constexpr (MR >= 2) {
        asm volatile("vmul.vx v8, v4, %0" : : "r"(zp_a_arr[1]) : "v8", "v9", "v10", "v11");
        asm volatile("vsub.vv v20, v20, v8" : : : "v20", "v21", "v22", "v23");
    }
    if constexpr (MR >= 3) {
        asm volatile("vmul.vx v8, v4, %0" : : "r"(zp_a_arr[2]) : "v8", "v9", "v10", "v11");
        asm volatile("vsub.vv v24, v24, v8" : : : "v24", "v25", "v26", "v27");
    }
    if constexpr (MR >= 4) {
        asm volatile("vmul.vx v8, v4, %0" : : "r"(zp_a_arr[3]) : "v8", "v9", "v10", "v11");
        asm volatile("vsub.vv v28, v28, v8" : : : "v28", "v29", "v30", "v31");
    }
}

// ============================================================================
// Dequantize accumulators to f32: acc → fp32 × scale_a[m] × scale_b[n]
//
// Internal accumulators are i32 in v16/v20/v24/v28 (m4). After this helper,
// each accumulator group holds the f32 dequantized result, ready for bias and
// clamp. Uses vfcvt.f.x.v (i32 → f32) and vfmul.
//
// scale_b is loaded once into v8 (f32, m4). scale_a is broadcast per row.
// ============================================================================

template <size_t MR>
static inline void tqt_dequant_acc_to_f32_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl,
                                                                      const float *scale_a_arr,
                                                                      const float *scale_b)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vle32.v v8, (%0)" : : "r"(scale_b) : "v8", "v9", "v10", "v11", "memory");

    if constexpr (MR >= 1) {
        asm volatile("vfcvt.f.x.v v16, v16" : : : "v16", "v17", "v18", "v19");
        asm volatile("vfmul.vf v16, v16, %0" : : "f"(scale_a_arr[0]) : "v16", "v17", "v18", "v19");
        asm volatile("vfmul.vv v16, v16, v8" : : : "v16", "v17", "v18", "v19");
    }
    if constexpr (MR >= 2) {
        asm volatile("vfcvt.f.x.v v20, v20" : : : "v20", "v21", "v22", "v23");
        asm volatile("vfmul.vf v20, v20, %0" : : "f"(scale_a_arr[1]) : "v20", "v21", "v22", "v23");
        asm volatile("vfmul.vv v20, v20, v8" : : : "v20", "v21", "v22", "v23");
    }
    if constexpr (MR >= 3) {
        asm volatile("vfcvt.f.x.v v24, v24" : : : "v24", "v25", "v26", "v27");
        asm volatile("vfmul.vf v24, v24, %0" : : "f"(scale_a_arr[2]) : "v24", "v25", "v26", "v27");
        asm volatile("vfmul.vv v24, v24, v8" : : : "v24", "v25", "v26", "v27");
    }
    if constexpr (MR >= 4) {
        asm volatile("vfcvt.f.x.v v28, v28" : : : "v28", "v29", "v30", "v31");
        asm volatile("vfmul.vf v28, v28, %0" : : "f"(scale_a_arr[3]) : "v28", "v29", "v30", "v31");
        asm volatile("vfmul.vv v28, v28, v8" : : : "v28", "v29", "v30", "v31");
    }
}

// ============================================================================
// Add f32 bias (1xN broadcast) to f32 accumulators.
// ============================================================================

template <size_t MR>
static inline void tqt_add_1xnbias_f32_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl, const float *BIAS)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vle32.v v8, (%0)" : : "r"(BIAS) : "v8", "v9", "v10", "v11", "memory");

    if constexpr (MR >= 1)
        asm volatile("vfadd.vv v16, v16, v8" : : : "v16", "v17", "v18", "v19");
    if constexpr (MR >= 2)
        asm volatile("vfadd.vv v20, v20, v8" : : : "v20", "v21", "v22", "v23");
    if constexpr (MR >= 3)
        asm volatile("vfadd.vv v24, v24, v8" : : : "v24", "v25", "v26", "v27");
    if constexpr (MR >= 4)
        asm volatile("vfadd.vv v28, v28, v8" : : : "v28", "v29", "v30", "v31");
}

// ============================================================================
// Add f32 C matrix (per-cell) to f32 accumulators.
//
// stride_row is in float (fp32) elements.
// ============================================================================

template <size_t MR>
static inline void tqt_add_c_f32_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl, const float *C,
                                                             size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    if constexpr (MR >= 1) {
        asm volatile("vle32.v v8, (%0)"
                     :
                     : "r"(C + 0 * stride_row)
                     : "v8", "v9", "v10", "v11", "memory");
        asm volatile("vfadd.vv v16, v16, v8" : : : "v16", "v17", "v18", "v19");
    }
    if constexpr (MR >= 2) {
        asm volatile("vle32.v v8, (%0)"
                     :
                     : "r"(C + 1 * stride_row)
                     : "v8", "v9", "v10", "v11", "memory");
        asm volatile("vfadd.vv v20, v20, v8" : : : "v20", "v21", "v22", "v23");
    }
    if constexpr (MR >= 3) {
        asm volatile("vle32.v v8, (%0)"
                     :
                     : "r"(C + 2 * stride_row)
                     : "v8", "v9", "v10", "v11", "memory");
        asm volatile("vfadd.vv v24, v24, v8" : : : "v24", "v25", "v26", "v27");
    }
    if constexpr (MR >= 4) {
        asm volatile("vle32.v v8, (%0)"
                     :
                     : "r"(C + 3 * stride_row)
                     : "v8", "v9", "v10", "v11", "memory");
        asm volatile("vfadd.vv v28, v28, v8" : : : "v28", "v29", "v30", "v31");
    }
}

// ============================================================================
// Clamp f32 accumulators with f32 min/max scalars.
// ============================================================================

template <size_t MR>
static inline void tqt_clamp_f32_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl, float min_val,
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

// ============================================================================
// Store f32 accumulators to D matrix (f32 output).
// stride_row is in float (fp32) elements.
// ============================================================================

template <size_t MR>
static inline void tqt_store_f32_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl, float *D, size_t stride_row)
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

// ============================================================================
// Narrow f32 accumulators to f16 and store.
// stride_row is in float16_t elements.
// Uses vfncvt.f.f.w (f32 → f16). After narrow, results are in v8/v10/v12/v14
// (f16, m2). Note: requires zfh/zvfh extensions.
// ============================================================================

#if defined(__riscv_zfh) && defined(__riscv_zvfh)
template <size_t MR>
static inline void tqt_store_narrow_f16_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl, float16_t *D,
                                                                    size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    if constexpr (MR >= 1) {
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl) :);
        asm volatile("vfncvt.f.f.w v8, v16" : : : "v8", "v9");
        asm volatile("vse16.v v8, (%0)" : : "r"(D + 0 * stride_row) : "memory");
    }
    if constexpr (MR >= 2) {
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl) :);
        asm volatile("vfncvt.f.f.w v10, v20" : : : "v10", "v11");
        asm volatile("vse16.v v10, (%0)" : : "r"(D + 1 * stride_row) : "memory");
    }
    if constexpr (MR >= 3) {
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl) :);
        asm volatile("vfncvt.f.f.w v12, v24" : : : "v12", "v13");
        asm volatile("vse16.v v12, (%0)" : : "r"(D + 2 * stride_row) : "memory");
    }
    if constexpr (MR >= 4) {
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl) :);
        asm volatile("vfncvt.f.f.w v14, v28" : : : "v14", "v15");
        asm volatile("vse16.v v14, (%0)" : : "r"(D + 3 * stride_row) : "memory");
    }
}

// ============================================================================
// Add f16 bias (1xN broadcast) to f32 accumulators (with widening).
// Uses vle16 (e16, m2) → vfwcvt.f.f.v → f32 (e32, m4) → vfadd.
// ============================================================================

template <size_t MR>
static inline void tqt_add_1xnbias_f16_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl, const float16_t *BIAS)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl) :);
    asm volatile("vle16.v v8, (%0)" : : "r"(BIAS) : "v8", "v9", "memory");
    asm volatile("vfwcvt.f.f.v v12, v8" : : : "v12", "v13", "v14", "v15");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl) :);
    if constexpr (MR >= 1)
        asm volatile("vfadd.vv v16, v16, v12" : : : "v16", "v17", "v18", "v19");
    if constexpr (MR >= 2)
        asm volatile("vfadd.vv v20, v20, v12" : : : "v20", "v21", "v22", "v23");
    if constexpr (MR >= 3)
        asm volatile("vfadd.vv v24, v24, v12" : : : "v24", "v25", "v26", "v27");
    if constexpr (MR >= 4)
        asm volatile("vfadd.vv v28, v28, v12" : : : "v28", "v29", "v30", "v31");
}

// ============================================================================
// Add f16 C matrix (per-cell) to f32 accumulators (with widening).
// stride_row is in float16_t elements.
// ============================================================================

template <size_t MR>
static inline void tqt_add_c_f16_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl, const float16_t *C,
                                                             size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    if constexpr (MR >= 1) {
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl) :);
        asm volatile("vle16.v v8, (%0)" : : "r"(C + 0 * stride_row) : "v8", "v9", "memory");
        asm volatile("vfwcvt.f.f.v v12, v8" : : : "v12", "v13", "v14", "v15");
        asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl) :);
        asm volatile("vfadd.vv v16, v16, v12" : : : "v16", "v17", "v18", "v19");
    }
    if constexpr (MR >= 2) {
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl) :);
        asm volatile("vle16.v v8, (%0)" : : "r"(C + 1 * stride_row) : "v8", "v9", "memory");
        asm volatile("vfwcvt.f.f.v v12, v8" : : : "v12", "v13", "v14", "v15");
        asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl) :);
        asm volatile("vfadd.vv v20, v20, v12" : : : "v20", "v21", "v22", "v23");
    }
    if constexpr (MR >= 3) {
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl) :);
        asm volatile("vle16.v v8, (%0)" : : "r"(C + 2 * stride_row) : "v8", "v9", "memory");
        asm volatile("vfwcvt.f.f.v v12, v8" : : : "v12", "v13", "v14", "v15");
        asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl) :);
        asm volatile("vfadd.vv v24, v24, v12" : : : "v24", "v25", "v26", "v27");
    }
    if constexpr (MR >= 4) {
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl) :);
        asm volatile("vle16.v v8, (%0)" : : "r"(C + 3 * stride_row) : "v8", "v9", "memory");
        asm volatile("vfwcvt.f.f.v v12, v8" : : : "v12", "v13", "v14", "v15");
        asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl) :);
        asm volatile("vfadd.vv v28, v28, v12" : : : "v28", "v29", "v30", "v31");
    }
}
#endif  // zfh + zvfh

// ============================================================================
// BF16 epilogue helpers (D = bf16). Mirrors the f16 block above; the only
// substantive change is replacing vfncvt.f.f.w with vfncvtbf16.f.f.w for
// narrow, and vfwcvt.f.f.v with vfwcvtbf16.f.f.v for widen. Acc layout
// (e32/m4 in v16/v20/v24/v28), narrow result layout (e16/m2 in
// v8/v10/v12/v14) and the bf16 vector temp slot for widen (v8 e16/m2) are
// identical.
// ============================================================================

#if defined(__riscv_zvfbfwma)
template <size_t MR>
static inline void tqt_store_narrow_bf16_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl, bfloat16_t *D,
                                                                     size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    if constexpr (MR >= 1) {
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl) :);
        asm volatile("vfncvtbf16.f.f.w v8, v16" : : : "v8", "v9");
        asm volatile("vse16.v v8, (%0)" : : "r"(D + 0 * stride_row) : "memory");
    }
    if constexpr (MR >= 2) {
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl) :);
        asm volatile("vfncvtbf16.f.f.w v10, v20" : : : "v10", "v11");
        asm volatile("vse16.v v10, (%0)" : : "r"(D + 1 * stride_row) : "memory");
    }
    if constexpr (MR >= 3) {
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl) :);
        asm volatile("vfncvtbf16.f.f.w v12, v24" : : : "v12", "v13");
        asm volatile("vse16.v v12, (%0)" : : "r"(D + 2 * stride_row) : "memory");
    }
    if constexpr (MR >= 4) {
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl) :);
        asm volatile("vfncvtbf16.f.f.w v14, v28" : : : "v14", "v15");
        asm volatile("vse16.v v14, (%0)" : : "r"(D + 3 * stride_row) : "memory");
    }
}

// ============================================================================
// Add bf16 bias (1xN broadcast) to f32 accumulators (with widening).
// Uses vle16 (e16, m2) -> vfwcvtbf16.f.f.v -> f32 (e32, m4) -> vfadd.
// ============================================================================

template <size_t MR>
static inline void tqt_add_1xnbias_bf16_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl,
                                                                    const bfloat16_t *BIAS)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl) :);
    asm volatile("vle16.v v8, (%0)" : : "r"(BIAS) : "v8", "v9", "memory");
    asm volatile("vfwcvtbf16.f.f.v v12, v8" : : : "v12", "v13", "v14", "v15");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl) :);
    if constexpr (MR >= 1)
        asm volatile("vfadd.vv v16, v16, v12" : : : "v16", "v17", "v18", "v19");
    if constexpr (MR >= 2)
        asm volatile("vfadd.vv v20, v20, v12" : : : "v20", "v21", "v22", "v23");
    if constexpr (MR >= 3)
        asm volatile("vfadd.vv v24, v24, v12" : : : "v24", "v25", "v26", "v27");
    if constexpr (MR >= 4)
        asm volatile("vfadd.vv v28, v28, v12" : : : "v28", "v29", "v30", "v31");
}

// ============================================================================
// Add bf16 C matrix (per-cell) to f32 accumulators (with widening).
// stride_row is in bfloat16_t elements.
// ============================================================================

template <size_t MR>
static inline void tqt_add_c_bf16_kernel_gemm_i8_i4_4x1vl_rvv(size_t vl, const bfloat16_t *C,
                                                              size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    if constexpr (MR >= 1) {
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl) :);
        asm volatile("vle16.v v8, (%0)" : : "r"(C + 0 * stride_row) : "v8", "v9", "memory");
        asm volatile("vfwcvtbf16.f.f.v v12, v8" : : : "v12", "v13", "v14", "v15");
        asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl) :);
        asm volatile("vfadd.vv v16, v16, v12" : : : "v16", "v17", "v18", "v19");
    }
    if constexpr (MR >= 2) {
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl) :);
        asm volatile("vle16.v v8, (%0)" : : "r"(C + 1 * stride_row) : "v8", "v9", "memory");
        asm volatile("vfwcvtbf16.f.f.v v12, v8" : : : "v12", "v13", "v14", "v15");
        asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl) :);
        asm volatile("vfadd.vv v20, v20, v12" : : : "v20", "v21", "v22", "v23");
    }
    if constexpr (MR >= 3) {
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl) :);
        asm volatile("vle16.v v8, (%0)" : : "r"(C + 2 * stride_row) : "v8", "v9", "memory");
        asm volatile("vfwcvtbf16.f.f.v v12, v8" : : : "v12", "v13", "v14", "v15");
        asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl) :);
        asm volatile("vfadd.vv v24, v24, v12" : : : "v24", "v25", "v26", "v27");
    }
    if constexpr (MR >= 4) {
        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(vl) :);
        asm volatile("vle16.v v8, (%0)" : : "r"(C + 3 * stride_row) : "v8", "v9", "memory");
        asm volatile("vfwcvtbf16.f.f.v v12, v8" : : : "v12", "v13", "v14", "v15");
        asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl) :);
        asm volatile("vfadd.vv v28, v28, v12" : : : "v28", "v29", "v30", "v31");
    }
}
#endif  // zvfbfwma
