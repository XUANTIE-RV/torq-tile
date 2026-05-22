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

/// Register allocation for 8x3vl GEMM kernel:
///   a0-a3:   A[0-3]
///   v0:      B matrix columns (int8 lmul=1)
///   v8-v15:  Temp int16 (int16 lmul=2)
///   v16-v31: Accumulator matrix D[0-3][0] (int32 lmul=4)
///   v4-v7:   Narrowed int8 results after requantize (int8 lmul=1)

/// Initialize accumulator registers (int32) to zero
template <size_t MR>
static inline void tqt_init_zero_kernel_gemm_i8_i8_4x1vl_rvv(size_t vl)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    // Row 0: v16-v19 (m4)
    if constexpr (MR >= 1)
        asm volatile("vmv.v.i v16, 0\n\t" : : : "v16", "v17", "v18", "v19");
    // Row 1: v20-v23 (m4)
    if constexpr (MR >= 2)
        asm volatile("vmv.v.i v20, 0\n\t" : : : "v20", "v21", "v22", "v23");
    // Row 2: v24-v27 (m4)
    if constexpr (MR >= 3)
        asm volatile("vmv.v.i v24, 0\n\t" : : : "v24", "v25", "v26", "v27");
    // Row 3: v28-v31 (m4)
    if constexpr (MR >= 4)
        asm volatile("vmv.v.i v28, 0\n\t" : : : "v28", "v29", "v30", "v31");
}

/// Initialize accumulator registers from C matrix (int32)
template <size_t MR>
static inline void tqt_init_c_kernel_gemm_i8_i8_4x1vl_rvv(size_t vl, const int32_t *C,
                                                          size_t stride_row, size_t stride_col)
{
    TQT_UNUSED(stride_col);
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    // Row 0: v16-v19
    if constexpr (MR >= 1)
        asm volatile("vle32.v v16, (%0)"
                     :
                     : "r"(C + 0 * stride_row)
                     : "v16", "v17", "v18", "v19", "memory");
    // Row 1: v20-v23
    if constexpr (MR >= 2)
        asm volatile("vle32.v v20, (%0)"
                     :
                     : "r"(C + 1 * stride_row)
                     : "v20", "v21", "v22", "v23", "memory");
    // Row 2: v24-v27
    if constexpr (MR >= 3)
        asm volatile("vle32.v v24, (%0)"
                     :
                     : "r"(C + 2 * stride_row)
                     : "v24", "v25", "v26", "v27", "memory");
    // Row 3: v28-v31
    if constexpr (MR >= 4)
        asm volatile("vle32.v v28, (%0)"
                     :
                     : "r"(C + 3 * stride_row)
                     : "v28", "v29", "v30", "v31", "memory");
}

/// Get pointers to A matrix elements for each row
/// Stores pointers into the provided array for later use by vmacc kernel
template <size_t MR>
static inline void tqt_get_a_ptr_gemm_i8_i8_4x1vl_rvv(const int8_t *A, size_t stride_row,
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

/// Load B matrix elements (non-transposed)
static inline void tqt_load_b_kernel_gemm_i8_i8_4x1vl_rvv(size_t vl, const int8_t *B)
{
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vle8.v v0, (%0)" : : "r"(B) : "v0", "memory");
}

/// Load B matrix elements (transposed)
static inline void tqt_load_bt_kernel_gemm_i8_i8_4x1vl_rvv(size_t vl, const int8_t *B,
                                                           size_t stride_row)
{
    size_t stride_bytes = stride_row * sizeof(int8_t);
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vlse8.v v0, (%0), %1" : : "r"(B), "r"(stride_bytes) : "v0", "memory");
}

/// Perform widening multiply-accumulate operation
/// Loads A scalars from provided pointers into a0-a3
/// Assumes B vector is already loaded in v0
template <size_t MR>
static inline void tqt_vmacc_kernel_gemm_i8_i8_4x1vl_rvv(size_t vl, const int8_t *const a_ptrs[MR])
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    // Load all A scalars into a0-a3
    // Widening multiply: int8 * int8 -> int16
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t" : : "r"(vl) :);
    if constexpr (MR >= 1)
        asm volatile("lb a0, 0(%0)" : : "r"(a_ptrs[0]) : "a0", "memory");
    asm volatile("vwmul.vx v8, v0, a0" : : : "v8", "v9");
    if constexpr (MR >= 2)
        asm volatile("lb a1, 0(%0)" : : "r"(a_ptrs[1]) : "a1", "memory");
    asm volatile("vwmul.vx v10, v0, a1" : : : "v10", "v11");
    if constexpr (MR >= 3)
        asm volatile("lb a2, 0(%0)" : : "r"(a_ptrs[2]) : "a2", "memory");
    asm volatile("vwmul.vx v12, v0, a2" : : : "v12", "v13");
    if constexpr (MR >= 4)
        asm volatile("lb a3, 0(%0)" : : "r"(a_ptrs[3]) : "a3", "memory");
    asm volatile("vwmul.vx v14, v0, a3" : : : "v14", "v15");

    // Widening add int16 -> int32 accumulator
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

/// Add 1xN bias to accumulator registers
template <size_t MR>
static inline void tqt_add_1xnbias_kernel_gemm_i8_i8_4x1vl_rvv(size_t vl, const int32_t *BIAS)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    // Load bias into v0-v3 (m4)
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vle32.v v0, (%0)" : : "r"(BIAS) : "v0", "v1", "v2", "v3", "memory");

    // Row 0
    if constexpr (MR >= 1)
        asm volatile("vadd.vv v16, v16, v0" : : : "v16", "v17", "v18", "v19");
    // Row 1
    if constexpr (MR >= 2)
        asm volatile("vadd.vv v20, v20, v0" : : : "v20", "v21", "v22", "v23");
    // Row 2
    if constexpr (MR >= 3)
        asm volatile("vadd.vv v24, v24, v0" : : : "v24", "v25", "v26", "v27");
    // Row 3
    if constexpr (MR >= 4)
        asm volatile("vadd.vv v28, v28, v0" : : : "v28", "v29", "v30", "v31");
}

/// Add Mx1 bias to accumulator registers
template <size_t MR>
static inline void tqt_add_mx1bias_kernel_gemm_i8_i8_4x1vl_rvv(size_t vl, const int32_t *BIAS)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);
    // Row 0
    if constexpr (MR >= 1)
        asm volatile("vadd.vx v16, v16, %0" : : "r"(BIAS[0]) : "v16", "v17", "v18", "v19");
    // Row 1
    if constexpr (MR >= 2)
        asm volatile("vadd.vx v20, v20, %0" : : "r"(BIAS[1]) : "v20", "v21", "v22", "v23");
    // Row 2
    if constexpr (MR >= 3)
        asm volatile("vadd.vx v24, v24, %0" : : "r"(BIAS[2]) : "v24", "v25", "v26", "v27");
    // Row 3
    if constexpr (MR >= 4)
        asm volatile("vadd.vx v28, v28, %0" : : "r"(BIAS[3]) : "v28", "v29", "v30", "v31");
}

/// Apply clamp (min/max) to int32 accumulator registers
template <size_t MR>
static inline void tqt_clamp_i32_kernel_gemm_i8_i8_4x1vl_rvv(size_t vl, int32_t min_val,
                                                             int32_t max_val)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    // Row 0
    if constexpr (MR >= 1) {
        asm volatile("vmax.vx v16, v16, %0" : : "r"(min_val) : "v16", "v17", "v18", "v19");
        asm volatile("vmin.vx v16, v16, %0" : : "r"(max_val) : "v16", "v17", "v18", "v19");
    }
    // Row 1
    if constexpr (MR >= 2) {
        asm volatile("vmax.vx v20, v20, %0" : : "r"(min_val) : "v20", "v21", "v22", "v23");
        asm volatile("vmin.vx v20, v20, %0" : : "r"(max_val) : "v20", "v21", "v22", "v23");
    }
    // Row 2
    if constexpr (MR >= 3) {
        asm volatile("vmax.vx v24, v24, %0" : : "r"(min_val) : "v24", "v25", "v26", "v27");
        asm volatile("vmin.vx v24, v24, %0" : : "r"(max_val) : "v24", "v25", "v26", "v27");
    }
    // Row 3
    if constexpr (MR >= 4) {
        asm volatile("vmax.vx v28, v28, %0" : : "r"(min_val) : "v28", "v29", "v30", "v31");
        asm volatile("vmin.vx v28, v28, %0" : : "r"(max_val) : "v28", "v29", "v30", "v31");
    }
}

/// Apply clamp (min/max) to int8 accumulator registers (v4-v7, m1)
/// Must be called after requantize which narrows results to int8
template <size_t MR>
static inline void tqt_clamp_i8_kernel_gemm_i8_i8_4x1vl_rvv(size_t vl, int8_t min_val,
                                                            int8_t max_val)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t" : : "r"(vl) :);

    // Row 0: v4 (m1)
    if constexpr (MR >= 1) {
        asm volatile("vmax.vx v4, v4, %0" : : "r"(min_val) : "v4");
        asm volatile("vmin.vx v4, v4, %0" : : "r"(max_val) : "v4");
    }
    // Row 1: v5 (m1)
    if constexpr (MR >= 2) {
        asm volatile("vmax.vx v5, v5, %0" : : "r"(min_val) : "v5");
        asm volatile("vmin.vx v5, v5, %0" : : "r"(max_val) : "v5");
    }
    // Row 2: v6 (m1)
    if constexpr (MR >= 3) {
        asm volatile("vmax.vx v6, v6, %0" : : "r"(min_val) : "v6");
        asm volatile("vmin.vx v6, v6, %0" : : "r"(max_val) : "v6");
    }
    // Row 3: v7 (m1)
    if constexpr (MR >= 4) {
        asm volatile("vmax.vx v7, v7, %0" : : "r"(min_val) : "v7");
        asm volatile("vmin.vx v7, v7, %0" : : "r"(max_val) : "v7");
    }
}

/// Requantize int32 to int8 with per-row (Mx1) quantization parameters
/// After this function, results are narrowed to int8 in v4-v7 (m1)
template <size_t MR>
static inline void tqt_requantize_mx1_kernel_gemm_i8_i8_4x1vl_rvv(size_t vl, const int32_t *mult,
                                                                  const int32_t *shift, int32_t zp)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    // Row 0: requantize + narrow int32 -> int16 -> int8
    if constexpr (MR >= 1) {
        asm volatile(
            "vmulh.vx v16, v16, %[m]\n\t"
            "vsra.vx v16, v16, %[s]\n\t"
            "vadd.vx v16, v16, %[z]\n\t"
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vnclip.wx v8, v16, zero\n\t"
            "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
            "vnclip.wx v4, v8, zero\n\t"
            :
            : [m] "r"(mult[0]), [s] "r"(shift[0]), [z] "r"(zp), [vl] "r"(vl)
            : "v4", "v8", "v9", "v16", "v17", "v18", "v19");
    }
    // Row 1
    if constexpr (MR >= 2) {
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
            "vmulh.vx v20, v20, %[m]\n\t"
            "vsra.vx v20, v20, %[s]\n\t"
            "vadd.vx v20, v20, %[z]\n\t"
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vnclip.wx v10, v20, zero\n\t"
            "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
            "vnclip.wx v5, v10, zero\n\t"
            :
            : [m] "r"(mult[1]), [s] "r"(shift[1]), [z] "r"(zp), [vl] "r"(vl)
            : "v5", "v10", "v11", "v20", "v21", "v22", "v23");
    }
    // Row 2
    if constexpr (MR >= 3) {
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
            "vmulh.vx v24, v24, %[m]\n\t"
            "vsra.vx v24, v24, %[s]\n\t"
            "vadd.vx v24, v24, %[z]\n\t"
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vnclip.wx v12, v24, zero\n\t"
            "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
            "vnclip.wx v6, v12, zero\n\t"
            :
            : [m] "r"(mult[2]), [s] "r"(shift[2]), [z] "r"(zp), [vl] "r"(vl)
            : "v6", "v12", "v13", "v24", "v25", "v26", "v27");
    }
    // Row 3
    if constexpr (MR >= 4) {
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
            "vmulh.vx v28, v28, %[m]\n\t"
            "vsra.vx v28, v28, %[s]\n\t"
            "vadd.vx v28, v28, %[z]\n\t"
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vnclip.wx v14, v28, zero\n\t"
            "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
            "vnclip.wx v7, v14, zero\n\t"
            :
            : [m] "r"(mult[3]), [s] "r"(shift[3]), [z] "r"(zp), [vl] "r"(vl)
            : "v7", "v14", "v15", "v28", "v29", "v30", "v31");
    }
}

/// Requantize int32 to int8 with per-column (1xN) quantization parameters
/// After this function, results are narrowed to int8 in v4-v7 (m1)
template <size_t MR>
static inline void tqt_requantize_1xn_kernel_gemm_i8_i8_4x1vl_rvv(size_t vl, const int32_t *mult,
                                                                  const int32_t *shift, int32_t zp)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    // Load mult and shift vectors into v0-v3 (m4) and v8-v11 (m4)
    asm volatile(
        "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
        "vle32.v v0, (%[mult])\n\t"
        "vle32.v v8, (%[shift])\n\t"
        :
        : [vl] "r"(vl), [mult] "r"(mult), [shift] "r"(shift)
        : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "memory");

    // Row 0: requantize + narrow int32 -> int16 -> int8
    if constexpr (MR >= 1) {
        asm volatile(
            "vmulh.vv v16, v16, v0\n\t"
            "vsra.vv v16, v16, v8\n\t"
            "vadd.vx v16, v16, %[z]\n\t"
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vnclip.wx v12, v16, zero\n\t"
            "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
            "vnclip.wx v4, v12, zero\n\t"
            :
            : [z] "r"(zp), [vl] "r"(vl)
            : "v4", "v12", "v13", "v16", "v17", "v18", "v19");
    }
    // Row 1
    if constexpr (MR >= 2) {
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
            "vmulh.vv v20, v20, v0\n\t"
            "vsra.vv v20, v20, v8\n\t"
            "vadd.vx v20, v20, %[z]\n\t"
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vnclip.wx v12, v20, zero\n\t"
            "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
            "vnclip.wx v5, v12, zero\n\t"
            :
            : [z] "r"(zp), [vl] "r"(vl)
            : "v5", "v12", "v13", "v20", "v21", "v22", "v23");
    }
    // Row 2
    if constexpr (MR >= 3) {
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
            "vmulh.vv v24, v24, v0\n\t"
            "vsra.vv v24, v24, v8\n\t"
            "vadd.vx v24, v24, %[z]\n\t"
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vnclip.wx v12, v24, zero\n\t"
            "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
            "vnclip.wx v6, v12, zero\n\t"
            :
            : [z] "r"(zp), [vl] "r"(vl)
            : "v6", "v12", "v13", "v24", "v25", "v26", "v27");
    }
    // Row 3
    if constexpr (MR >= 4) {
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"
            "vmulh.vv v28, v28, v0\n\t"
            "vsra.vv v28, v28, v8\n\t"
            "vadd.vx v28, v28, %[z]\n\t"
            "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
            "vnclip.wx v14, v28, zero\n\t"
            "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
            "vnclip.wx v7, v14, zero\n\t"
            :
            : [z] "r"(zp), [vl] "r"(vl)
            : "v7", "v14", "v15", "v28", "v29", "v30", "v31");
    }
}

/// Store int32 accumulator registers to D matrix
template <size_t MR>
static inline void tqt_store_i32_kernel_gemm_i8_i8_4x1vl_rvv(size_t vl, int32_t *D,
                                                             size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    // Row 0
    if constexpr (MR >= 1)
        asm volatile("vse32.v v16, (%0)" : : "r"(D + 0 * stride_row) : "memory");
    // Row 1
    if constexpr (MR >= 2)
        asm volatile("vse32.v v20, (%0)" : : "r"(D + 1 * stride_row) : "memory");
    // Row 2
    if constexpr (MR >= 3)
        asm volatile("vse32.v v24, (%0)" : : "r"(D + 2 * stride_row) : "memory");
    // Row 3
    if constexpr (MR >= 4)
        asm volatile("vse32.v v28, (%0)" : : "r"(D + 3 * stride_row) : "memory");
}

/// Store int8 accumulator registers (v4-v7, m1) to D matrix
/// Must be called after requantize (which narrows to int8) and clamp_i8
template <size_t MR>
static inline void tqt_store_i8_kernel_gemm_i8_i8_4x1vl_rvv(size_t vl, int8_t *D, size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t" : : "r"(vl) :);

    if constexpr (MR >= 1)
        asm volatile("vse8.v v4, (%0)" : : "r"(D + 0 * stride_row) : "memory");
    if constexpr (MR >= 2)
        asm volatile("vse8.v v5, (%0)" : : "r"(D + 1 * stride_row) : "memory");
    if constexpr (MR >= 3)
        asm volatile("vse8.v v6, (%0)" : : "r"(D + 2 * stride_row) : "memory");
    if constexpr (MR >= 4)
        asm volatile("vse8.v v7, (%0)" : : "r"(D + 3 * stride_row) : "memory");
}
