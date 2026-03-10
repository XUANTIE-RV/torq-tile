//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif  // Architectural features check.

#include <stddef.h>

#include "src/common/tqt_common.h"

/// Register allocation for 8x3vl GEMM kernel:
///   ft0-ft7: A[0-7]
///   v0-v2:   B matrix columns (3 vector registers)
///   v8-v31:  Accumulator matrix D[0-7][0-2] (24 vector registers)

/// Macro to initialize a single vector register to zero
#define TQT_VZERO_FP16(reg_num) asm volatile("vmv.v.i v" #reg_num ", 0\n\t" : : : "v" #reg_num);

/// Initialize accumulator registers to zero
template <size_t MR, size_t NR>
static inline void tqt_init_zero_kernel_gemm_f16_f16_f16_rvv(size_t vl)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma\n\t" : : "r"(vl) :);

    // Row 0: v8, v9, v10
    if constexpr (MR >= 1) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP16(8);
        if constexpr (NR >= 2)
            TQT_VZERO_FP16(9);
        if constexpr (NR >= 3)
            TQT_VZERO_FP16(10);
    }
    // Row 1: v11, v12, v13
    if constexpr (MR >= 2) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP16(11);
        if constexpr (NR >= 2)
            TQT_VZERO_FP16(12);
        if constexpr (NR >= 3)
            TQT_VZERO_FP16(13);
    }
    // Row 2: v14, v15, v16
    if constexpr (MR >= 3) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP16(14);
        if constexpr (NR >= 2)
            TQT_VZERO_FP16(15);
        if constexpr (NR >= 3)
            TQT_VZERO_FP16(16);
    }
    // Row 3: v17, v18, v19
    if constexpr (MR >= 4) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP16(17);
        if constexpr (NR >= 2)
            TQT_VZERO_FP16(18);
        if constexpr (NR >= 3)
            TQT_VZERO_FP16(19);
    }
    // Row 4: v20, v21, v22
    if constexpr (MR >= 5) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP16(20);
        if constexpr (NR >= 2)
            TQT_VZERO_FP16(21);
        if constexpr (NR >= 3)
            TQT_VZERO_FP16(22);
    }
    // Row 5: v23, v24, v25
    if constexpr (MR >= 6) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP16(23);
        if constexpr (NR >= 2)
            TQT_VZERO_FP16(24);
        if constexpr (NR >= 3)
            TQT_VZERO_FP16(25);
    }
    // Row 6: v26, v27, v28
    if constexpr (MR >= 7) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP16(26);
        if constexpr (NR >= 2)
            TQT_VZERO_FP16(27);
        if constexpr (NR >= 3)
            TQT_VZERO_FP16(28);
    }
    // Row 7: v29, v30, v31
    if constexpr (MR >= 8) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP16(29);
        if constexpr (NR >= 2)
            TQT_VZERO_FP16(30);
        if constexpr (NR >= 3)
            TQT_VZERO_FP16(31);
    }
}

/// Initialize accumulator registers from C matrix
template <size_t MR, size_t NR>
static inline void tqt_init_c_kernel_gemm_f16_f16_f16_rvv(size_t vl, const float16_t *C,
                                                          size_t stride_row, size_t stride_col)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma\n\t" : : "r"(vl) :);

    // Row 0
    if constexpr (MR >= 1) {
        const float16_t *c_row_0 = C + 0 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vle16.v v8, (%0)" : : "r"(c_row_0 + 0 * stride_col) : "v8", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle16.v v9, (%0)" : : "r"(c_row_0 + 1 * stride_col) : "v9", "memory");
        if constexpr (NR >= 3)
            asm volatile("vle16.v v10, (%0)" : : "r"(c_row_0 + 2 * stride_col) : "v10", "memory");
    }
    // Row 1
    if constexpr (MR >= 2) {
        const float16_t *c_row_1 = C + 1 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vle16.v v11, (%0)" : : "r"(c_row_1 + 0 * stride_col) : "v11", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle16.v v12, (%0)" : : "r"(c_row_1 + 1 * stride_col) : "v12", "memory");
        if constexpr (NR >= 3)
            asm volatile("vle16.v v13, (%0)" : : "r"(c_row_1 + 2 * stride_col) : "v13", "memory");
    }
    // Row 2
    if constexpr (MR >= 3) {
        const float16_t *c_row_2 = C + 2 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vle16.v v14, (%0)" : : "r"(c_row_2 + 0 * stride_col) : "v14", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle16.v v15, (%0)" : : "r"(c_row_2 + 1 * stride_col) : "v15", "memory");
        if constexpr (NR >= 3)
            asm volatile("vle16.v v16, (%0)" : : "r"(c_row_2 + 2 * stride_col) : "v16", "memory");
    }
    // Row 3
    if constexpr (MR >= 4) {
        const float16_t *c_row_3 = C + 3 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vle16.v v17, (%0)" : : "r"(c_row_3 + 0 * stride_col) : "v17", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle16.v v18, (%0)" : : "r"(c_row_3 + 1 * stride_col) : "v18", "memory");
        if constexpr (NR >= 3)
            asm volatile("vle16.v v19, (%0)" : : "r"(c_row_3 + 2 * stride_col) : "v19", "memory");
    }
    // Row 4
    if constexpr (MR >= 5) {
        const float16_t *c_row_4 = C + 4 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vle16.v v20, (%0)" : : "r"(c_row_4 + 0 * stride_col) : "v20", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle16.v v21, (%0)" : : "r"(c_row_4 + 1 * stride_col) : "v21", "memory");
        if constexpr (NR >= 3)
            asm volatile("vle16.v v22, (%0)" : : "r"(c_row_4 + 2 * stride_col) : "v22", "memory");
    }
    // Row 5
    if constexpr (MR >= 6) {
        const float16_t *c_row_5 = C + 5 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vle16.v v23, (%0)" : : "r"(c_row_5 + 0 * stride_col) : "v23", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle16.v v24, (%0)" : : "r"(c_row_5 + 1 * stride_col) : "v24", "memory");
        if constexpr (NR >= 3)
            asm volatile("vle16.v v25, (%0)" : : "r"(c_row_5 + 2 * stride_col) : "v25", "memory");
    }
    // Row 6
    if constexpr (MR >= 7) {
        const float16_t *c_row_6 = C + 6 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vle16.v v26, (%0)" : : "r"(c_row_6 + 0 * stride_col) : "v26", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle16.v v27, (%0)" : : "r"(c_row_6 + 1 * stride_col) : "v27", "memory");
        if constexpr (NR >= 3)
            asm volatile("vle16.v v28, (%0)" : : "r"(c_row_6 + 2 * stride_col) : "v28", "memory");
    }
    // Row 7
    if constexpr (MR >= 8) {
        const float16_t *c_row_7 = C + 7 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vle16.v v29, (%0)" : : "r"(c_row_7 + 0 * stride_col) : "v29", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle16.v v30, (%0)" : : "r"(c_row_7 + 1 * stride_col) : "v30", "memory");
        if constexpr (NR >= 3)
            asm volatile("vle16.v v31, (%0)" : : "r"(c_row_7 + 2 * stride_col) : "v31", "memory");
    }
}

/// Load A matrix elements (scalars) into floating-point registers
template <size_t MR>
static inline void tqt_load_a_kernel_gemm_f16_f16_f16_rvv(const float16_t *A, size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");

    if constexpr (MR >= 1)
        asm volatile("flh ft0, 0(%0)" : : "r"(A + 0 * stride_row) : "ft0", "memory");
    if constexpr (MR >= 2)
        asm volatile("flh ft1, 0(%0)" : : "r"(A + 1 * stride_row) : "ft1", "memory");
    if constexpr (MR >= 3)
        asm volatile("flh ft2, 0(%0)" : : "r"(A + 2 * stride_row) : "ft2", "memory");
    if constexpr (MR >= 4)
        asm volatile("flh ft3, 0(%0)" : : "r"(A + 3 * stride_row) : "ft3", "memory");
    if constexpr (MR >= 5)
        asm volatile("flh ft4, 0(%0)" : : "r"(A + 4 * stride_row) : "ft4", "memory");
    if constexpr (MR >= 6)
        asm volatile("flh ft5, 0(%0)" : : "r"(A + 5 * stride_row) : "ft5", "memory");
    if constexpr (MR >= 7)
        asm volatile("flh ft6, 0(%0)" : : "r"(A + 6 * stride_row) : "ft6", "memory");
    if constexpr (MR >= 8)
        asm volatile("flh ft7, 0(%0)" : : "r"(A + 7 * stride_row) : "ft7", "memory");
}

/// Load B matrix columns (vectors) into vector registers
template <size_t NR>
static inline void tqt_load_b_kernel_gemm_f16_f16_f16_rvv(size_t vl, const float16_t *B,
                                                          size_t stride_col)
{
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    if constexpr (NR >= 1)
        asm volatile("vle16.v v0, (%0)" : : "r"(B + 0 * stride_col) : "v0", "memory");
    if constexpr (NR >= 2)
        asm volatile("vle16.v v1, (%0)" : : "r"(B + 1 * stride_col) : "v1", "memory");
    if constexpr (NR >= 3)
        asm volatile("vle16.v v2, (%0)" : : "r"(B + 2 * stride_col) : "v2", "memory");
}

/// Load transposed B matrix columns (vectors) into vector registers
template <size_t NR>
static inline void tqt_load_bt_kernel_gemm_f16_f16_f16_rvv(size_t vl, const float16_t *B,
                                                           size_t stride_row)
{
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    size_t stride_bytes = stride_row * sizeof(float16_t);
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    if constexpr (NR >= 1)
        asm volatile("vlse16.v v0, (%0), %1"
                     :
                     : "r"(B + 0 * vl * stride_row), "r"(stride_bytes)
                     : "v0", "memory");
    if constexpr (NR >= 2)
        asm volatile("vlse16.v v1, (%0), %1"
                     :
                     : "r"(B + 1 * vl * stride_row), "r"(stride_bytes)
                     : "v1", "memory");
    if constexpr (NR >= 3)
        asm volatile("vlse16.v v2, (%0), %1"
                     :
                     : "r"(B + 2 * vl * stride_row), "r"(stride_bytes)
                     : "v2", "memory");
}

/// Perform VFMACC operations
template <size_t MR, size_t NR>
static inline void tqt_vfmacc_kernel_gemm_f16_f16_f16_rvv(size_t vl)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    // Row 0
    if constexpr (MR >= 1) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v8, ft0, v0" : : : "v8");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v9, ft0, v1" : : : "v9");
        if constexpr (NR >= 3)
            asm volatile("vfmacc.vf v10, ft0, v2" : : : "v10");
    }
    // Row 1
    if constexpr (MR >= 2) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v11, ft1, v0" : : : "v11");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v12, ft1, v1" : : : "v12");
        if constexpr (NR >= 3)
            asm volatile("vfmacc.vf v13, ft1, v2" : : : "v13");
    }
    // Row 2
    if constexpr (MR >= 3) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v14, ft2, v0" : : : "v14");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v15, ft2, v1" : : : "v15");
        if constexpr (NR >= 3)
            asm volatile("vfmacc.vf v16, ft2, v2" : : : "v16");
    }
    // Row 3
    if constexpr (MR >= 4) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v17, ft3, v0" : : : "v17");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v18, ft3, v1" : : : "v18");
        if constexpr (NR >= 3)
            asm volatile("vfmacc.vf v19, ft3, v2" : : : "v19");
    }
    // Row 4
    if constexpr (MR >= 5) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v20, ft4, v0" : : : "v20");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v21, ft4, v1" : : : "v21");
        if constexpr (NR >= 3)
            asm volatile("vfmacc.vf v22, ft4, v2" : : : "v22");
    }
    // Row 5
    if constexpr (MR >= 6) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v23, ft5, v0" : : : "v23");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v24, ft5, v1" : : : "v24");
        if constexpr (NR >= 3)
            asm volatile("vfmacc.vf v25, ft5, v2" : : : "v25");
    }
    // Row 6
    if constexpr (MR >= 7) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v26, ft6, v0" : : : "v26");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v27, ft6, v1" : : : "v27");
        if constexpr (NR >= 3)
            asm volatile("vfmacc.vf v28, ft6, v2" : : : "v28");
    }
    // Row 7
    if constexpr (MR >= 8) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v29, ft7, v0" : : : "v29");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v30, ft7, v1" : : : "v30");
        if constexpr (NR >= 3)
            asm volatile("vfmacc.vf v31, ft7, v2" : : : "v31");
    }
}

/// Add 1xn bias to accumulator registers
template <size_t MR, size_t NR>
static inline void tqt_add_1xnbias_kernel_gemm_f16_f16_f16_rvv(size_t vl, const float16_t *BIAS)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    if constexpr (NR >= 1)
        asm volatile("vle16.v v3, (%0)" : : "r"(BIAS + 0 * vl) : "v3", "memory");
    if constexpr (NR >= 2)
        asm volatile("vle16.v v4, (%0)" : : "r"(BIAS + 1 * vl) : "v4", "memory");
    if constexpr (NR >= 3)
        asm volatile("vle16.v v5, (%0)" : : "r"(BIAS + 2 * vl) : "v5", "memory");

    // Row 0
    if constexpr (MR >= 1) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v8, v8, v3" : : : "v8");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v9, v9, v4" : : : "v9");
        if constexpr (NR >= 3)
            asm volatile("vfadd.vv v10, v10, v5" : : : "v10");
    }
    // Row 1
    if constexpr (MR >= 2) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v11, v11, v3" : : : "v11");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v12, v12, v4" : : : "v12");
        if constexpr (NR >= 3)
            asm volatile("vfadd.vv v13, v13, v5" : : : "v13");
    }
    // Row 2
    if constexpr (MR >= 3) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v14, v14, v3" : : : "v14");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v15, v15, v4" : : : "v15");
        if constexpr (NR >= 3)
            asm volatile("vfadd.vv v16, v16, v5" : : : "v16");
    }
    // Row 3
    if constexpr (MR >= 4) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v17, v17, v3" : : : "v17");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v18, v18, v4" : : : "v18");
        if constexpr (NR >= 3)
            asm volatile("vfadd.vv v19, v19, v5" : : : "v19");
    }
    // Row 4
    if constexpr (MR >= 5) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v20, v20, v3" : : : "v20");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v21, v21, v4" : : : "v21");
        if constexpr (NR >= 3)
            asm volatile("vfadd.vv v22, v22, v5" : : : "v22");
    }
    // Row 5
    if constexpr (MR >= 6) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v23, v23, v3" : : : "v23");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v24, v24, v4" : : : "v24");
        if constexpr (NR >= 3)
            asm volatile("vfadd.vv v25, v25, v5" : : : "v25");
    }
    // Row 6
    if constexpr (MR >= 7) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v26, v26, v3" : : : "v26");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v27, v27, v4" : : : "v27");
        if constexpr (NR >= 3)
            asm volatile("vfadd.vv v28, v28, v5" : : : "v28");
    }
    // Row 7
    if constexpr (MR >= 8) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v29, v29, v3" : : : "v29");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v30, v30, v4" : : : "v30");
        if constexpr (NR >= 3)
            asm volatile("vfadd.vv v31, v31, v5" : : : "v31");
    }
}

/// Add mx1 bias to accumulator registers
template <size_t MR, size_t NR>
static inline void tqt_add_mx1bias_kernel_gemm_f16_f16_f16_rvv(size_t vl, const float16_t *BIAS)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    // Row 0
    if constexpr (MR >= 1) {
        asm volatile("flh ft0, 0(%0)" : : "r"(BIAS + 0) : "ft0", "memory");
        if constexpr (NR >= 1)
            asm volatile("vfadd.vf v8, v8, ft0" : : : "v8");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vf v9, v9, ft0" : : : "v9");
        if constexpr (NR >= 3)
            asm volatile("vfadd.vf v10, v10, ft0" : : : "v10");
    }
    // Row 1
    if constexpr (MR >= 2) {
        asm volatile("flh ft1, 0(%0)" : : "r"(BIAS + 1) : "ft1", "memory");
        if constexpr (NR >= 1)
            asm volatile("vfadd.vf v11, v11, ft1" : : : "v11");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vf v12, v12, ft1" : : : "v12");
        if constexpr (NR >= 3)
            asm volatile("vfadd.vf v13, v13, ft1" : : : "v13");
    }
    // Row 2
    if constexpr (MR >= 3) {
        asm volatile("flh ft2, 0(%0)" : : "r"(BIAS + 2) : "ft2", "memory");
        if constexpr (NR >= 1)
            asm volatile("vfadd.vf v14, v14, ft2" : : : "v14");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vf v15, v15, ft2" : : : "v15");
        if constexpr (NR >= 3)
            asm volatile("vfadd.vf v16, v16, ft2" : : : "v16");
    }
    // Row 3
    if constexpr (MR >= 4) {
        asm volatile("flh ft3, 0(%0)" : : "r"(BIAS + 3) : "ft3", "memory");
        if constexpr (NR >= 1)
            asm volatile("vfadd.vf v17, v17, ft3" : : : "v17");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vf v18, v18, ft3" : : : "v18");
        if constexpr (NR >= 3)
            asm volatile("vfadd.vf v19, v19, ft3" : : : "v19");
    }
    // Row 4
    if constexpr (MR >= 5) {
        asm volatile("flh ft4, 0(%0)" : : "r"(BIAS + 4) : "ft4", "memory");
        if constexpr (NR >= 1)
            asm volatile("vfadd.vf v20, v20, ft4" : : : "v20");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vf v21, v21, ft4" : : : "v21");
        if constexpr (NR >= 3)
            asm volatile("vfadd.vf v22, v22, ft4" : : : "v22");
    }
    // Row 5
    if constexpr (MR >= 6) {
        asm volatile("flh ft5, 0(%0)" : : "r"(BIAS + 5) : "ft5", "memory");
        if constexpr (NR >= 1)
            asm volatile("vfadd.vf v23, v23, ft5" : : : "v23");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vf v24, v24, ft5" : : : "v24");
        if constexpr (NR >= 3)
            asm volatile("vfadd.vf v25, v25, ft5" : : : "v25");
    }
    // Row 6
    if constexpr (MR >= 7) {
        asm volatile("flh ft6, 0(%0)" : : "r"(BIAS + 6) : "ft6", "memory");
        if constexpr (NR >= 1)
            asm volatile("vfadd.vf v26, v26, ft6" : : : "v26");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vf v27, v27, ft6" : : : "v27");
        if constexpr (NR >= 3)
            asm volatile("vfadd.vf v28, v28, ft6" : : : "v28");
    }
    // Row 7
    if constexpr (MR >= 8) {
        asm volatile("flh ft7, 0(%0)" : : "r"(BIAS + 7) : "ft7", "memory");
        if constexpr (NR >= 1)
            asm volatile("vfadd.vf v29, v29, ft7" : : : "v29");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vf v30, v30, ft7" : : : "v30");
        if constexpr (NR >= 3)
            asm volatile("vfadd.vf v31, v31, ft7" : : : "v31");
    }
}

/// Apply clamp (min/max) to accumulator registers
template <size_t MR, size_t NR>
static inline void tqt_clamp_kernel_gemm_f16_f16_f16_rvv(size_t vl, float16_t min, float16_t max)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    asm volatile(
        "flh ft8, 0(%0)\n\t"
        "flh ft9, 0(%1)\n\t"
        :
        : "r"(&min), "r"(&max)
        : "ft8", "ft9", "memory");

    // Row 0
    if constexpr (MR >= 1) {
        if constexpr (NR >= 1) {
            asm volatile("vfmax.vf v8, v8, ft8" : : : "v8");
            asm volatile("vfmin.vf v8, v8, ft9" : : : "v8");
        }
        if constexpr (NR >= 2) {
            asm volatile("vfmax.vf v9, v9, ft8" : : : "v9");
            asm volatile("vfmin.vf v9, v9, ft9" : : : "v9");
        }
        if constexpr (NR >= 3) {
            asm volatile("vfmax.vf v10, v10, ft8" : : : "v10");
            asm volatile("vfmin.vf v10, v10, ft9" : : : "v10");
        }
    }
    // Row 1
    if constexpr (MR >= 2) {
        if constexpr (NR >= 1) {
            asm volatile("vfmax.vf v11, v11, ft8" : : : "v11");
            asm volatile("vfmin.vf v11, v11, ft9" : : : "v11");
        }
        if constexpr (NR >= 2) {
            asm volatile("vfmax.vf v12, v12, ft8" : : : "v12");
            asm volatile("vfmin.vf v12, v12, ft9" : : : "v12");
        }
        if constexpr (NR >= 3) {
            asm volatile("vfmax.vf v13, v13, ft8" : : : "v13");
            asm volatile("vfmin.vf v13, v13, ft9" : : : "v13");
        }
    }
    // Row 2
    if constexpr (MR >= 3) {
        if constexpr (NR >= 1) {
            asm volatile("vfmax.vf v14, v14, ft8" : : : "v14");
            asm volatile("vfmin.vf v14, v14, ft9" : : : "v14");
        }
        if constexpr (NR >= 2) {
            asm volatile("vfmax.vf v15, v15, ft8" : : : "v15");
            asm volatile("vfmin.vf v15, v15, ft9" : : : "v15");
        }
        if constexpr (NR >= 3) {
            asm volatile("vfmax.vf v16, v16, ft8" : : : "v16");
            asm volatile("vfmin.vf v16, v16, ft9" : : : "v16");
        }
    }
    // Row 3
    if constexpr (MR >= 4) {
        if constexpr (NR >= 1) {
            asm volatile("vfmax.vf v17, v17, ft8" : : : "v17");
            asm volatile("vfmin.vf v17, v17, ft9" : : : "v17");
        }
        if constexpr (NR >= 2) {
            asm volatile("vfmax.vf v18, v18, ft8" : : : "v18");
            asm volatile("vfmin.vf v18, v18, ft9" : : : "v18");
        }
        if constexpr (NR >= 3) {
            asm volatile("vfmax.vf v19, v19, ft8" : : : "v19");
            asm volatile("vfmin.vf v19, v19, ft9" : : : "v19");
        }
    }
    // Row 4
    if constexpr (MR >= 5) {
        if constexpr (NR >= 1) {
            asm volatile("vfmax.vf v20, v20, ft8" : : : "v20");
            asm volatile("vfmin.vf v20, v20, ft9" : : : "v20");
        }
        if constexpr (NR >= 2) {
            asm volatile("vfmax.vf v21, v21, ft8" : : : "v21");
            asm volatile("vfmin.vf v21, v21, ft9" : : : "v21");
        }
        if constexpr (NR >= 3) {
            asm volatile("vfmax.vf v22, v22, ft8" : : : "v22");
            asm volatile("vfmin.vf v22, v22, ft9" : : : "v22");
        }
    }
    // Row 5
    if constexpr (MR >= 6) {
        if constexpr (NR >= 1) {
            asm volatile("vfmax.vf v23, v23, ft8" : : : "v23");
            asm volatile("vfmin.vf v23, v23, ft9" : : : "v23");
        }
        if constexpr (NR >= 2) {
            asm volatile("vfmax.vf v24, v24, ft8" : : : "v24");
            asm volatile("vfmin.vf v24, v24, ft9" : : : "v24");
        }
        if constexpr (NR >= 3) {
            asm volatile("vfmax.vf v25, v25, ft8" : : : "v25");
            asm volatile("vfmin.vf v25, v25, ft9" : : : "v25");
        }
    }
    // Row 6
    if constexpr (MR >= 7) {
        if constexpr (NR >= 1) {
            asm volatile("vfmax.vf v26, v26, ft8" : : : "v26");
            asm volatile("vfmin.vf v26, v26, ft9" : : : "v26");
        }
        if constexpr (NR >= 2) {
            asm volatile("vfmax.vf v27, v27, ft8" : : : "v27");
            asm volatile("vfmin.vf v27, v27, ft9" : : : "v27");
        }
        if constexpr (NR >= 3) {
            asm volatile("vfmax.vf v28, v28, ft8" : : : "v28");
            asm volatile("vfmin.vf v28, v28, ft9" : : : "v28");
        }
    }
    // Row 7
    if constexpr (MR >= 8) {
        if constexpr (NR >= 1) {
            asm volatile("vfmax.vf v29, v29, ft8" : : : "v29");
            asm volatile("vfmin.vf v29, v29, ft9" : : : "v29");
        }
        if constexpr (NR >= 2) {
            asm volatile("vfmax.vf v30, v30, ft8" : : : "v30");
            asm volatile("vfmin.vf v30, v30, ft9" : : : "v30");
        }
        if constexpr (NR >= 3) {
            asm volatile("vfmax.vf v31, v31, ft8" : : : "v31");
            asm volatile("vfmin.vf v31, v31, ft9" : : : "v31");
        }
    }
}

/// Store accumulator registers to D matrix
template <size_t MR, size_t NR>
static inline void tqt_store_kernel_gemm_f16_f16_f16_rvv(size_t vl, float16_t *D, size_t stride_row,
                                                         size_t stride_col)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    if constexpr (MR >= 1) {
        float16_t *d_row_0 = D + 0 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse16.v v8, (%0)" : : "r"(d_row_0 + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse16.v v9, (%0)" : : "r"(d_row_0 + 1 * stride_col) : "memory");
        if constexpr (NR >= 3)
            asm volatile("vse16.v v10, (%0)" : : "r"(d_row_0 + 2 * stride_col) : "memory");
    }
    if constexpr (MR >= 2) {
        float16_t *d_row_1 = D + 1 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse16.v v11, (%0)" : : "r"(d_row_1 + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse16.v v12, (%0)" : : "r"(d_row_1 + 1 * stride_col) : "memory");
        if constexpr (NR >= 3)
            asm volatile("vse16.v v13, (%0)" : : "r"(d_row_1 + 2 * stride_col) : "memory");
    }
    if constexpr (MR >= 3) {
        float16_t *d_row_2 = D + 2 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse16.v v14, (%0)" : : "r"(d_row_2 + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse16.v v15, (%0)" : : "r"(d_row_2 + 1 * stride_col) : "memory");
        if constexpr (NR >= 3)
            asm volatile("vse16.v v16, (%0)" : : "r"(d_row_2 + 2 * stride_col) : "memory");
    }
    if constexpr (MR >= 4) {
        float16_t *d_row_3 = D + 3 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse16.v v17, (%0)" : : "r"(d_row_3 + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse16.v v18, (%0)" : : "r"(d_row_3 + 1 * stride_col) : "memory");
        if constexpr (NR >= 3)
            asm volatile("vse16.v v19, (%0)" : : "r"(d_row_3 + 2 * stride_col) : "memory");
    }
    if constexpr (MR >= 5) {
        float16_t *d_row_4 = D + 4 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse16.v v20, (%0)" : : "r"(d_row_4 + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse16.v v21, (%0)" : : "r"(d_row_4 + 1 * stride_col) : "memory");
        if constexpr (NR >= 3)
            asm volatile("vse16.v v22, (%0)" : : "r"(d_row_4 + 2 * stride_col) : "memory");
    }
    if constexpr (MR >= 6) {
        float16_t *d_row_5 = D + 5 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse16.v v23, (%0)" : : "r"(d_row_5 + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse16.v v24, (%0)" : : "r"(d_row_5 + 1 * stride_col) : "memory");
        if constexpr (NR >= 3)
            asm volatile("vse16.v v25, (%0)" : : "r"(d_row_5 + 2 * stride_col) : "memory");
    }
    if constexpr (MR >= 7) {
        float16_t *d_row_6 = D + 6 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse16.v v26, (%0)" : : "r"(d_row_6 + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse16.v v27, (%0)" : : "r"(d_row_6 + 1 * stride_col) : "memory");
        if constexpr (NR >= 3)
            asm volatile("vse16.v v28, (%0)" : : "r"(d_row_6 + 2 * stride_col) : "memory");
    }
    if constexpr (MR >= 8) {
        float16_t *d_row_7 = D + 7 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse16.v v29, (%0)" : : "r"(d_row_7 + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse16.v v30, (%0)" : : "r"(d_row_7 + 1 * stride_col) : "memory");
        if constexpr (NR >= 3)
            asm volatile("vse16.v v31, (%0)" : : "r"(d_row_7 + 2 * stride_col) : "memory");
    }
}
