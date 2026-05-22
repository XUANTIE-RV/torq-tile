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

/// Register allocation for 8x2vl W4A16 GEMM kernel:
///   ft0-ft7:  A[0-7] scalars
///   v0-v1:    dequantized B col 0 (lo / hi nibble)
///   v2-v3:    dequantized B col 1 (lo / hi nibble)
///   v4-v6:    dequant temporaries (mf2: raw load / lo nibble / hi nibble)
///   v8-v9:    scale vectors (col 0 / col 1)
///   v10-v11:  bias vectors (epilogue)
///   v16-v31:  accumulator matrix D[0-7][0-1] (16 vector registers)

// ============================================================================
// Macros
// ============================================================================

#define TQT_VZERO_FP16_W4A16(reg_num) \
    asm volatile("vmv.v.i v" #reg_num ", 0\n\t" : : : "v" #reg_num);

// ============================================================================
// Initialize accumulators to zero
// ============================================================================

template <size_t MR, size_t NR>
static inline void tqt_init_zero_kernel_gemm_f16_i4_8x2vl_rvv(size_t vl)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 2, "NR must be in range [1, 2]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma\n\t" : : "r"(vl) :);

    if constexpr (MR >= 1) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP16_W4A16(16);
        if constexpr (NR >= 2)
            TQT_VZERO_FP16_W4A16(17);
    }
    if constexpr (MR >= 2) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP16_W4A16(18);
        if constexpr (NR >= 2)
            TQT_VZERO_FP16_W4A16(19);
    }
    if constexpr (MR >= 3) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP16_W4A16(20);
        if constexpr (NR >= 2)
            TQT_VZERO_FP16_W4A16(21);
    }
    if constexpr (MR >= 4) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP16_W4A16(22);
        if constexpr (NR >= 2)
            TQT_VZERO_FP16_W4A16(23);
    }
    if constexpr (MR >= 5) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP16_W4A16(24);
        if constexpr (NR >= 2)
            TQT_VZERO_FP16_W4A16(25);
    }
    if constexpr (MR >= 6) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP16_W4A16(26);
        if constexpr (NR >= 2)
            TQT_VZERO_FP16_W4A16(27);
    }
    if constexpr (MR >= 7) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP16_W4A16(28);
        if constexpr (NR >= 2)
            TQT_VZERO_FP16_W4A16(29);
    }
    if constexpr (MR >= 8) {
        if constexpr (NR >= 1)
            TQT_VZERO_FP16_W4A16(30);
        if constexpr (NR >= 2)
            TQT_VZERO_FP16_W4A16(31);
    }
}

// ============================================================================
// Initialize accumulators from C matrix
// ============================================================================

template <size_t MR, size_t NR>
static inline void tqt_init_c_kernel_gemm_f16_i4_8x2vl_rvv(size_t vl, const float16_t *C,
                                                           size_t stride_row, size_t stride_col)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 2, "NR must be in range [1, 2]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma\n\t" : : "r"(vl) :);

    if constexpr (MR >= 1) {
        const float16_t *c_row = C + 0 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vle16.v v16, (%0)" : : "r"(c_row + 0 * stride_col) : "v16", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle16.v v17, (%0)" : : "r"(c_row + 1 * stride_col) : "v17", "memory");
    }
    if constexpr (MR >= 2) {
        const float16_t *c_row = C + 1 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vle16.v v18, (%0)" : : "r"(c_row + 0 * stride_col) : "v18", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle16.v v19, (%0)" : : "r"(c_row + 1 * stride_col) : "v19", "memory");
    }
    if constexpr (MR >= 3) {
        const float16_t *c_row = C + 2 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vle16.v v20, (%0)" : : "r"(c_row + 0 * stride_col) : "v20", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle16.v v21, (%0)" : : "r"(c_row + 1 * stride_col) : "v21", "memory");
    }
    if constexpr (MR >= 4) {
        const float16_t *c_row = C + 3 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vle16.v v22, (%0)" : : "r"(c_row + 0 * stride_col) : "v22", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle16.v v23, (%0)" : : "r"(c_row + 1 * stride_col) : "v23", "memory");
    }
    if constexpr (MR >= 5) {
        const float16_t *c_row = C + 4 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vle16.v v24, (%0)" : : "r"(c_row + 0 * stride_col) : "v24", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle16.v v25, (%0)" : : "r"(c_row + 1 * stride_col) : "v25", "memory");
    }
    if constexpr (MR >= 6) {
        const float16_t *c_row = C + 5 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vle16.v v26, (%0)" : : "r"(c_row + 0 * stride_col) : "v26", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle16.v v27, (%0)" : : "r"(c_row + 1 * stride_col) : "v27", "memory");
    }
    if constexpr (MR >= 7) {
        const float16_t *c_row = C + 6 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vle16.v v28, (%0)" : : "r"(c_row + 0 * stride_col) : "v28", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle16.v v29, (%0)" : : "r"(c_row + 1 * stride_col) : "v29", "memory");
    }
    if constexpr (MR >= 8) {
        const float16_t *c_row = C + 7 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vle16.v v30, (%0)" : : "r"(c_row + 0 * stride_col) : "v30", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle16.v v31, (%0)" : : "r"(c_row + 1 * stride_col) : "v31", "memory");
    }
}

// ============================================================================
// Load scale vectors via stride load along N dimension → v8, v9
// ============================================================================

template <size_t NR>
static inline void tqt_load_scale_kernel_gemm_f16_i4_8x2vl_rvv(size_t vl,
                                                               const float16_t *scale_col0,
                                                               const float16_t *scale_col1,
                                                               size_t scale_byte_stride)
{
    static_assert(NR >= 1 && NR <= 2, "NR must be in range [1, 2]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    if constexpr (NR >= 1)
        asm volatile("vlse16.v v8, (%0), %1"
                     :
                     : "r"(scale_col0), "r"(scale_byte_stride)
                     : "v8", "memory");
    if constexpr (NR >= 2)
        asm volatile("vlse16.v v9, (%0), %1"
                     :
                     : "r"(scale_col1), "r"(scale_byte_stride)
                     : "v9", "memory");
}

// ============================================================================
// Dequantize B: stride load int4 bytes, split nibbles, convert to f16, apply scale
// Results: v0/v1 (col 0 lo/hi), v2/v3 (col 1 lo/hi)
// Temporaries: v4-v6 (mf2)
// Scale in: v8 (col 0), v9 (col 1)
// ============================================================================

template <size_t NR>
static inline void tqt_dequant_bt_kernel_gemm_f16_i4_8x2vl_rvv(size_t vl, const uint8_t *b_col0,
                                                               const uint8_t *b_col1,
                                                               size_t b_byte_stride)
{
    static_assert(NR >= 1 && NR <= 2, "NR must be in range [1, 2]");

    // --- Col 0 ---
    if constexpr (NR >= 1) {
        // Load raw bytes (e8, mf2)
        asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" : : "r"(vl) :);
        asm volatile("vlse8.v v4, (%0), %1" : : "r"(b_col0), "r"(b_byte_stride) : "v4", "memory");
        // Split nibbles
        asm volatile(
            "vand.vi v5, v4, 0xf\n\t"  // lo nibble → v5
            "vsrl.vi v6, v4, 4\n\t"    // hi nibble → v6
            :
            :
            : "v5", "v6");

        // Widen to u16, subtract ZP, convert to f16, multiply scale
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        asm volatile(
            "vzext.vf2 v0, v5\n\t"  // u8 → u16
            "vzext.vf2 v1, v6\n\t"
            "vadd.vi v0, v0, -8\n\t"  // unsigned → signed (ZP=8)
            "vadd.vi v1, v1, -8\n\t"
            "vfcvt.f.x.v v0, v0\n\t"  // i16 → f16
            "vfcvt.f.x.v v1, v1\n\t"
            "vfmul.vv v0, v0, v8\n\t"  // × scale col 0
            "vfmul.vv v1, v1, v8\n\t"
            :
            :
            : "v0", "v1");
    }

    // --- Col 1 ---
    if constexpr (NR >= 2) {
        asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" : : "r"(vl) :);
        asm volatile("vlse8.v v4, (%0), %1" : : "r"(b_col1), "r"(b_byte_stride) : "v4", "memory");
        asm volatile(
            "vand.vi v5, v4, 0xf\n\t"
            "vsrl.vi v6, v4, 4\n\t"
            :
            :
            : "v5", "v6");

        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        asm volatile(
            "vzext.vf2 v2, v5\n\t"
            "vzext.vf2 v3, v6\n\t"
            "vadd.vi v2, v2, -8\n\t"
            "vadd.vi v3, v3, -8\n\t"
            "vfcvt.f.x.v v2, v2\n\t"
            "vfcvt.f.x.v v3, v3\n\t"
            "vfmul.vv v2, v2, v9\n\t"  // × scale col 1
            "vfmul.vv v3, v3, v9\n\t"
            :
            :
            : "v2", "v3");
    }
}

// ============================================================================
// Dequantize B (packed): contiguous load int4 bytes, split nibbles, convert to f16, apply scale
// For packed B layout where data is already contiguous in memory.
// Results: v0/v1 (col 0 lo/hi), v2/v3 (col 1 lo/hi)
// Temporaries: v4-v6 (mf2)
// Scale in: v8 (col 0), v9 (col 1)
// ============================================================================

template <size_t NR>
static inline void tqt_dequant_b_kernel_gemm_f16_i4_8x2vl_rvv(size_t vl, const uint8_t *b_col0,
                                                              const uint8_t *b_col1)
{
    static_assert(NR >= 1 && NR <= 2, "NR must be in range [1, 2]");

    // --- Col 0 ---
    if constexpr (NR >= 1) {
        // Load raw bytes (e8, mf2) - contiguous load for packed B
        asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" : : "r"(vl) :);
        asm volatile("vle8.v v4, (%0)" : : "r"(b_col0) : "v4", "memory");
        // Split nibbles
        asm volatile(
            "vand.vi v5, v4, 0xf\n\t"  // lo nibble → v5
            "vsrl.vi v6, v4, 4\n\t"    // hi nibble → v6
            :
            :
            : "v5", "v6");

        // Widen to u16, subtract ZP, convert to f16, multiply scale
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        asm volatile(
            "vzext.vf2 v0, v5\n\t"  // u8 → u16
            "vzext.vf2 v1, v6\n\t"
            "vadd.vi v0, v0, -8\n\t"  // unsigned → signed (ZP=8)
            "vadd.vi v1, v1, -8\n\t"
            "vfcvt.f.x.v v0, v0\n\t"  // i16 → f16
            "vfcvt.f.x.v v1, v1\n\t"
            "vfmul.vv v0, v0, v8\n\t"  // × scale col 0
            "vfmul.vv v1, v1, v8\n\t"
            :
            :
            : "v0", "v1");
    }

    // --- Col 1 ---
    if constexpr (NR >= 2) {
        asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" : : "r"(vl) :);
        asm volatile("vle8.v v4, (%0)" : : "r"(b_col1) : "v4", "memory");
        asm volatile(
            "vand.vi v5, v4, 0xf\n\t"
            "vsrl.vi v6, v4, 4\n\t"
            :
            :
            : "v5", "v6");

        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        asm volatile(
            "vzext.vf2 v2, v5\n\t"
            "vzext.vf2 v3, v6\n\t"
            "vadd.vi v2, v2, -8\n\t"
            "vadd.vi v3, v3, -8\n\t"
            "vfcvt.f.x.v v2, v2\n\t"
            "vfcvt.f.x.v v3, v3\n\t"
            "vfmul.vv v2, v2, v9\n\t"  // × scale col 1
            "vfmul.vv v3, v3, v9\n\t"
            :
            :
            : "v2", "v3");
    }
}

// ============================================================================
// Load A scalars into ft0-ft7
// ============================================================================

template <size_t MR>
static inline void tqt_load_a_kernel_gemm_f16_i4_8x2vl_rvv(const float16_t *A, size_t stride_row)
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

// ============================================================================
// FMA with lo nibble (ki+0): v0=col0, v2=col1, ft0-ft7=A scalars
// Accumulators: v16-v31
// ============================================================================

template <size_t MR, size_t NR>
static inline void tqt_vfmacc_lo_kernel_gemm_f16_i4_8x2vl_rvv()
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 2, "NR must be in range [1, 2]");

    if constexpr (MR >= 1) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v16, ft0, v0" ::: "v16");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v17, ft0, v2" ::: "v17");
    }
    if constexpr (MR >= 2) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v18, ft1, v0" ::: "v18");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v19, ft1, v2" ::: "v19");
    }
    if constexpr (MR >= 3) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v20, ft2, v0" ::: "v20");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v21, ft2, v2" ::: "v21");
    }
    if constexpr (MR >= 4) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v22, ft3, v0" ::: "v22");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v23, ft3, v2" ::: "v23");
    }
    if constexpr (MR >= 5) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v24, ft4, v0" ::: "v24");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v25, ft4, v2" ::: "v25");
    }
    if constexpr (MR >= 6) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v26, ft5, v0" ::: "v26");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v27, ft5, v2" ::: "v27");
    }
    if constexpr (MR >= 7) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v28, ft6, v0" ::: "v28");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v29, ft6, v2" ::: "v29");
    }
    if constexpr (MR >= 8) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v30, ft7, v0" ::: "v30");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v31, ft7, v2" ::: "v31");
    }
}

// ============================================================================
// FMA with hi nibble (ki+1): v1=col0, v3=col1, ft0-ft7=A scalars
// ============================================================================

template <size_t MR, size_t NR>
static inline void tqt_vfmacc_hi_kernel_gemm_f16_i4_8x2vl_rvv()
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 2, "NR must be in range [1, 2]");

    if constexpr (MR >= 1) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v16, ft0, v1" ::: "v16");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v17, ft0, v3" ::: "v17");
    }
    if constexpr (MR >= 2) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v18, ft1, v1" ::: "v18");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v19, ft1, v3" ::: "v19");
    }
    if constexpr (MR >= 3) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v20, ft2, v1" ::: "v20");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v21, ft2, v3" ::: "v21");
    }
    if constexpr (MR >= 4) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v22, ft3, v1" ::: "v22");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v23, ft3, v3" ::: "v23");
    }
    if constexpr (MR >= 5) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v24, ft4, v1" ::: "v24");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v25, ft4, v3" ::: "v25");
    }
    if constexpr (MR >= 6) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v26, ft5, v1" ::: "v26");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v27, ft5, v3" ::: "v27");
    }
    if constexpr (MR >= 7) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v28, ft6, v1" ::: "v28");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v29, ft6, v3" ::: "v29");
    }
    if constexpr (MR >= 8) {
        if constexpr (NR >= 1)
            asm volatile("vfmacc.vf v30, ft7, v1" ::: "v30");
        if constexpr (NR >= 2)
            asm volatile("vfmacc.vf v31, ft7, v3" ::: "v31");
    }
}

// ============================================================================
// Add 1xN bias: load bias into v10/v11, add to accumulators
// ============================================================================

template <size_t MR, size_t NR>
static inline void tqt_add_1xnbias_kernel_gemm_f16_i4_8x2vl_rvv(size_t vl, const float16_t *BIAS)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 2, "NR must be in range [1, 2]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    if constexpr (NR >= 1)
        asm volatile("vle16.v v10, (%0)" : : "r"(BIAS + 0 * vl) : "v10", "memory");
    if constexpr (NR >= 2)
        asm volatile("vle16.v v11, (%0)" : : "r"(BIAS + 1 * vl) : "v11", "memory");

    if constexpr (MR >= 1) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v16, v16, v10" ::: "v16");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v17, v17, v11" ::: "v17");
    }
    if constexpr (MR >= 2) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v18, v18, v10" ::: "v18");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v19, v19, v11" ::: "v19");
    }
    if constexpr (MR >= 3) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v20, v20, v10" ::: "v20");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v21, v21, v11" ::: "v21");
    }
    if constexpr (MR >= 4) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v22, v22, v10" ::: "v22");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v23, v23, v11" ::: "v23");
    }
    if constexpr (MR >= 5) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v24, v24, v10" ::: "v24");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v25, v25, v11" ::: "v25");
    }
    if constexpr (MR >= 6) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v26, v26, v10" ::: "v26");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v27, v27, v11" ::: "v27");
    }
    if constexpr (MR >= 7) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v28, v28, v10" ::: "v28");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v29, v29, v11" ::: "v29");
    }
    if constexpr (MR >= 8) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v30, v30, v10" ::: "v30");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v31, v31, v11" ::: "v31");
    }
}

// ============================================================================
// Clamp accumulators: vfmax(min) + vfmin(max)
// ============================================================================

template <size_t MR, size_t NR>
static inline void tqt_clamp_kernel_gemm_f16_i4_8x2vl_rvv(size_t vl, float16_t min, float16_t max)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 2, "NR must be in range [1, 2]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    asm volatile(
        "flh ft8, 0(%0)\n\t"
        "flh ft9, 0(%1)\n\t"
        :
        : "r"(&min), "r"(&max)
        : "ft8", "ft9", "memory");

    if constexpr (MR >= 1) {
        if constexpr (NR >= 1) {
            asm volatile("vfmax.vf v16, v16, ft8" ::: "v16");
            asm volatile("vfmin.vf v16, v16, ft9" ::: "v16");
        }
        if constexpr (NR >= 2) {
            asm volatile("vfmax.vf v17, v17, ft8" ::: "v17");
            asm volatile("vfmin.vf v17, v17, ft9" ::: "v17");
        }
    }
    if constexpr (MR >= 2) {
        if constexpr (NR >= 1) {
            asm volatile("vfmax.vf v18, v18, ft8" ::: "v18");
            asm volatile("vfmin.vf v18, v18, ft9" ::: "v18");
        }
        if constexpr (NR >= 2) {
            asm volatile("vfmax.vf v19, v19, ft8" ::: "v19");
            asm volatile("vfmin.vf v19, v19, ft9" ::: "v19");
        }
    }
    if constexpr (MR >= 3) {
        if constexpr (NR >= 1) {
            asm volatile("vfmax.vf v20, v20, ft8" ::: "v20");
            asm volatile("vfmin.vf v20, v20, ft9" ::: "v20");
        }
        if constexpr (NR >= 2) {
            asm volatile("vfmax.vf v21, v21, ft8" ::: "v21");
            asm volatile("vfmin.vf v21, v21, ft9" ::: "v21");
        }
    }
    if constexpr (MR >= 4) {
        if constexpr (NR >= 1) {
            asm volatile("vfmax.vf v22, v22, ft8" ::: "v22");
            asm volatile("vfmin.vf v22, v22, ft9" ::: "v22");
        }
        if constexpr (NR >= 2) {
            asm volatile("vfmax.vf v23, v23, ft8" ::: "v23");
            asm volatile("vfmin.vf v23, v23, ft9" ::: "v23");
        }
    }
    if constexpr (MR >= 5) {
        if constexpr (NR >= 1) {
            asm volatile("vfmax.vf v24, v24, ft8" ::: "v24");
            asm volatile("vfmin.vf v24, v24, ft9" ::: "v24");
        }
        if constexpr (NR >= 2) {
            asm volatile("vfmax.vf v25, v25, ft8" ::: "v25");
            asm volatile("vfmin.vf v25, v25, ft9" ::: "v25");
        }
    }
    if constexpr (MR >= 6) {
        if constexpr (NR >= 1) {
            asm volatile("vfmax.vf v26, v26, ft8" ::: "v26");
            asm volatile("vfmin.vf v26, v26, ft9" ::: "v26");
        }
        if constexpr (NR >= 2) {
            asm volatile("vfmax.vf v27, v27, ft8" ::: "v27");
            asm volatile("vfmin.vf v27, v27, ft9" ::: "v27");
        }
    }
    if constexpr (MR >= 7) {
        if constexpr (NR >= 1) {
            asm volatile("vfmax.vf v28, v28, ft8" ::: "v28");
            asm volatile("vfmin.vf v28, v28, ft9" ::: "v28");
        }
        if constexpr (NR >= 2) {
            asm volatile("vfmax.vf v29, v29, ft8" ::: "v29");
            asm volatile("vfmin.vf v29, v29, ft9" ::: "v29");
        }
    }
    if constexpr (MR >= 8) {
        if constexpr (NR >= 1) {
            asm volatile("vfmax.vf v30, v30, ft8" ::: "v30");
            asm volatile("vfmin.vf v30, v30, ft9" ::: "v30");
        }
        if constexpr (NR >= 2) {
            asm volatile("vfmax.vf v31, v31, ft8" ::: "v31");
            asm volatile("vfmin.vf v31, v31, ft9" ::: "v31");
        }
    }
}

// ============================================================================
// Store accumulators to D matrix
// ============================================================================

template <size_t MR, size_t NR>
static inline void tqt_store_kernel_gemm_f16_i4_8x2vl_rvv(size_t vl, float16_t *D,
                                                          size_t stride_row, size_t stride_col)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 2, "NR must be in range [1, 2]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    if constexpr (MR >= 1) {
        float16_t *d_row = D + 0 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse16.v v16, (%0)" : : "r"(d_row + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse16.v v17, (%0)" : : "r"(d_row + 1 * stride_col) : "memory");
    }
    if constexpr (MR >= 2) {
        float16_t *d_row = D + 1 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse16.v v18, (%0)" : : "r"(d_row + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse16.v v19, (%0)" : : "r"(d_row + 1 * stride_col) : "memory");
    }
    if constexpr (MR >= 3) {
        float16_t *d_row = D + 2 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse16.v v20, (%0)" : : "r"(d_row + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse16.v v21, (%0)" : : "r"(d_row + 1 * stride_col) : "memory");
    }
    if constexpr (MR >= 4) {
        float16_t *d_row = D + 3 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse16.v v22, (%0)" : : "r"(d_row + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse16.v v23, (%0)" : : "r"(d_row + 1 * stride_col) : "memory");
    }
    if constexpr (MR >= 5) {
        float16_t *d_row = D + 4 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse16.v v24, (%0)" : : "r"(d_row + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse16.v v25, (%0)" : : "r"(d_row + 1 * stride_col) : "memory");
    }
    if constexpr (MR >= 6) {
        float16_t *d_row = D + 5 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse16.v v26, (%0)" : : "r"(d_row + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse16.v v27, (%0)" : : "r"(d_row + 1 * stride_col) : "memory");
    }
    if constexpr (MR >= 7) {
        float16_t *d_row = D + 6 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse16.v v28, (%0)" : : "r"(d_row + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse16.v v29, (%0)" : : "r"(d_row + 1 * stride_col) : "memory");
    }
    if constexpr (MR >= 8) {
        float16_t *d_row = D + 7 * stride_row;
        if constexpr (NR >= 1)
            asm volatile("vse16.v v30, (%0)" : : "r"(d_row + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse16.v v31, (%0)" : : "r"(d_row + 1 * stride_col) : "memory");
    }
}

// ============================================================================
// FP32 epilogue helpers for f32-output W4A16 kernel variants.
// Internal accumulation stays in f16 (v16-v31, e16/m1); these helpers wrap
// the f32 inputs/outputs at the boundary so that the f32-output variants can
// reuse all the core f16 helpers above.
// ============================================================================

// Initialize accumulators from fp32 C: vle32 (e32, m2) → vfncvt.f.f.w → v16-v31 (e16, m1).
// stride_row / stride_col are in float (fp32) elements.
template <size_t MR, size_t NR>
static inline void tqt_init_c_f32_kernel_gemm_f16_i4_8x2vl_rvv(size_t vl, const float *C,
                                                               size_t stride_row, size_t stride_col)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 2, "NR must be in range [1, 2]");

    if constexpr (MR >= 1) {
        const float *c_row = C + 0 * stride_row;
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vle32.v v0, (%0)" : : "r"(c_row + 0 * stride_col) : "v0", "v1", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle32.v v2, (%0)" : : "r"(c_row + 1 * stride_col) : "v2", "v3", "memory");
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vfncvt.f.f.w v16, v0" ::: "v16");
        if constexpr (NR >= 2)
            asm volatile("vfncvt.f.f.w v17, v2" ::: "v17");
    }
    if constexpr (MR >= 2) {
        const float *c_row = C + 1 * stride_row;
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vle32.v v0, (%0)" : : "r"(c_row + 0 * stride_col) : "v0", "v1", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle32.v v2, (%0)" : : "r"(c_row + 1 * stride_col) : "v2", "v3", "memory");
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vfncvt.f.f.w v18, v0" ::: "v18");
        if constexpr (NR >= 2)
            asm volatile("vfncvt.f.f.w v19, v2" ::: "v19");
    }
    if constexpr (MR >= 3) {
        const float *c_row = C + 2 * stride_row;
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vle32.v v0, (%0)" : : "r"(c_row + 0 * stride_col) : "v0", "v1", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle32.v v2, (%0)" : : "r"(c_row + 1 * stride_col) : "v2", "v3", "memory");
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vfncvt.f.f.w v20, v0" ::: "v20");
        if constexpr (NR >= 2)
            asm volatile("vfncvt.f.f.w v21, v2" ::: "v21");
    }
    if constexpr (MR >= 4) {
        const float *c_row = C + 3 * stride_row;
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vle32.v v0, (%0)" : : "r"(c_row + 0 * stride_col) : "v0", "v1", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle32.v v2, (%0)" : : "r"(c_row + 1 * stride_col) : "v2", "v3", "memory");
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vfncvt.f.f.w v22, v0" ::: "v22");
        if constexpr (NR >= 2)
            asm volatile("vfncvt.f.f.w v23, v2" ::: "v23");
    }
    if constexpr (MR >= 5) {
        const float *c_row = C + 4 * stride_row;
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vle32.v v0, (%0)" : : "r"(c_row + 0 * stride_col) : "v0", "v1", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle32.v v2, (%0)" : : "r"(c_row + 1 * stride_col) : "v2", "v3", "memory");
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vfncvt.f.f.w v24, v0" ::: "v24");
        if constexpr (NR >= 2)
            asm volatile("vfncvt.f.f.w v25, v2" ::: "v25");
    }
    if constexpr (MR >= 6) {
        const float *c_row = C + 5 * stride_row;
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vle32.v v0, (%0)" : : "r"(c_row + 0 * stride_col) : "v0", "v1", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle32.v v2, (%0)" : : "r"(c_row + 1 * stride_col) : "v2", "v3", "memory");
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vfncvt.f.f.w v26, v0" ::: "v26");
        if constexpr (NR >= 2)
            asm volatile("vfncvt.f.f.w v27, v2" ::: "v27");
    }
    if constexpr (MR >= 7) {
        const float *c_row = C + 6 * stride_row;
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vle32.v v0, (%0)" : : "r"(c_row + 0 * stride_col) : "v0", "v1", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle32.v v2, (%0)" : : "r"(c_row + 1 * stride_col) : "v2", "v3", "memory");
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vfncvt.f.f.w v28, v0" ::: "v28");
        if constexpr (NR >= 2)
            asm volatile("vfncvt.f.f.w v29, v2" ::: "v29");
    }
    if constexpr (MR >= 8) {
        const float *c_row = C + 7 * stride_row;
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vle32.v v0, (%0)" : : "r"(c_row + 0 * stride_col) : "v0", "v1", "memory");
        if constexpr (NR >= 2)
            asm volatile("vle32.v v2, (%0)" : : "r"(c_row + 1 * stride_col) : "v2", "v3", "memory");
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vfncvt.f.f.w v30, v0" ::: "v30");
        if constexpr (NR >= 2)
            asm volatile("vfncvt.f.f.w v31, v2" ::: "v31");
    }
}

// Add 1xN fp32 bias: vle32 (e32, m2) → vfncvt.f.f.w → v10/v11 (e16, m1) → vfadd.vv.
template <size_t MR, size_t NR>
static inline void tqt_add_1xnbias_f32_kernel_gemm_f16_i4_8x2vl_rvv(size_t vl, const float *BIAS)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 2, "NR must be in range [1, 2]");

    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);
    if constexpr (NR >= 1)
        asm volatile("vle32.v v0, (%0)" : : "r"(BIAS + 0 * vl) : "v0", "v1", "memory");
    if constexpr (NR >= 2)
        asm volatile("vle32.v v2, (%0)" : : "r"(BIAS + 1 * vl) : "v2", "v3", "memory");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
    if constexpr (NR >= 1)
        asm volatile("vfncvt.f.f.w v10, v0" ::: "v10");
    if constexpr (NR >= 2)
        asm volatile("vfncvt.f.f.w v11, v2" ::: "v11");

    if constexpr (MR >= 1) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v16, v16, v10" ::: "v16");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v17, v17, v11" ::: "v17");
    }
    if constexpr (MR >= 2) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v18, v18, v10" ::: "v18");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v19, v19, v11" ::: "v19");
    }
    if constexpr (MR >= 3) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v20, v20, v10" ::: "v20");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v21, v21, v11" ::: "v21");
    }
    if constexpr (MR >= 4) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v22, v22, v10" ::: "v22");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v23, v23, v11" ::: "v23");
    }
    if constexpr (MR >= 5) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v24, v24, v10" ::: "v24");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v25, v25, v11" ::: "v25");
    }
    if constexpr (MR >= 6) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v26, v26, v10" ::: "v26");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v27, v27, v11" ::: "v27");
    }
    if constexpr (MR >= 7) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v28, v28, v10" ::: "v28");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v29, v29, v11" ::: "v29");
    }
    if constexpr (MR >= 8) {
        if constexpr (NR >= 1)
            asm volatile("vfadd.vv v30, v30, v10" ::: "v30");
        if constexpr (NR >= 2)
            asm volatile("vfadd.vv v31, v31, v11" ::: "v31");
    }
}

// Clamp using fp32 scalars (cast to f16 once and forward to the f16 clamp helper).
// Internal accumulators are f16, so clamping in f16 is sufficient.
template <size_t MR, size_t NR>
static inline void tqt_clamp_f32_kernel_gemm_f16_i4_8x2vl_rvv(size_t vl, float min_f32,
                                                              float max_f32)
{
    float16_t mn = static_cast<float16_t>(min_f32);
    float16_t mx = static_cast<float16_t>(max_f32);
    tqt_clamp_kernel_gemm_f16_i4_8x2vl_rvv<MR, NR>(vl, mn, mx);
}

// Widen-and-store: v16-v31 (e16, m1) → fp32 (e32, m2) via vfwcvt.f.f.v → vse32.v.
// stride_row / stride_col are in float (fp32) elements.
template <size_t MR, size_t NR>
static inline void tqt_store_widen_f32_kernel_gemm_f16_i4_8x2vl_rvv(size_t vl, float *D,
                                                                    size_t stride_row,
                                                                    size_t stride_col)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 2, "NR must be in range [1, 2]");

    if constexpr (MR >= 1) {
        float *d_row = D + 0 * stride_row;
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vfwcvt.f.f.v v0, v16" ::: "v0", "v1");
        if constexpr (NR >= 2)
            asm volatile("vfwcvt.f.f.v v2, v17" ::: "v2", "v3");
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vse32.v v0, (%0)" : : "r"(d_row + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse32.v v2, (%0)" : : "r"(d_row + 1 * stride_col) : "memory");
    }
    if constexpr (MR >= 2) {
        float *d_row = D + 1 * stride_row;
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vfwcvt.f.f.v v0, v18" ::: "v0", "v1");
        if constexpr (NR >= 2)
            asm volatile("vfwcvt.f.f.v v2, v19" ::: "v2", "v3");
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vse32.v v0, (%0)" : : "r"(d_row + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse32.v v2, (%0)" : : "r"(d_row + 1 * stride_col) : "memory");
    }
    if constexpr (MR >= 3) {
        float *d_row = D + 2 * stride_row;
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vfwcvt.f.f.v v0, v20" ::: "v0", "v1");
        if constexpr (NR >= 2)
            asm volatile("vfwcvt.f.f.v v2, v21" ::: "v2", "v3");
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vse32.v v0, (%0)" : : "r"(d_row + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse32.v v2, (%0)" : : "r"(d_row + 1 * stride_col) : "memory");
    }
    if constexpr (MR >= 4) {
        float *d_row = D + 3 * stride_row;
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vfwcvt.f.f.v v0, v22" ::: "v0", "v1");
        if constexpr (NR >= 2)
            asm volatile("vfwcvt.f.f.v v2, v23" ::: "v2", "v3");
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vse32.v v0, (%0)" : : "r"(d_row + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse32.v v2, (%0)" : : "r"(d_row + 1 * stride_col) : "memory");
    }
    if constexpr (MR >= 5) {
        float *d_row = D + 4 * stride_row;
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vfwcvt.f.f.v v0, v24" ::: "v0", "v1");
        if constexpr (NR >= 2)
            asm volatile("vfwcvt.f.f.v v2, v25" ::: "v2", "v3");
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vse32.v v0, (%0)" : : "r"(d_row + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse32.v v2, (%0)" : : "r"(d_row + 1 * stride_col) : "memory");
    }
    if constexpr (MR >= 6) {
        float *d_row = D + 5 * stride_row;
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vfwcvt.f.f.v v0, v26" ::: "v0", "v1");
        if constexpr (NR >= 2)
            asm volatile("vfwcvt.f.f.v v2, v27" ::: "v2", "v3");
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vse32.v v0, (%0)" : : "r"(d_row + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse32.v v2, (%0)" : : "r"(d_row + 1 * stride_col) : "memory");
    }
    if constexpr (MR >= 7) {
        float *d_row = D + 6 * stride_row;
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vfwcvt.f.f.v v0, v28" ::: "v0", "v1");
        if constexpr (NR >= 2)
            asm volatile("vfwcvt.f.f.v v2, v29" ::: "v2", "v3");
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vse32.v v0, (%0)" : : "r"(d_row + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse32.v v2, (%0)" : : "r"(d_row + 1 * stride_col) : "memory");
    }
    if constexpr (MR >= 8) {
        float *d_row = D + 7 * stride_row;
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vfwcvt.f.f.v v0, v30" ::: "v0", "v1");
        if constexpr (NR >= 2)
            asm volatile("vfwcvt.f.f.v v2, v31" ::: "v2", "v3");
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);
        if constexpr (NR >= 1)
            asm volatile("vse32.v v0, (%0)" : : "r"(d_row + 0 * stride_col) : "memory");
        if constexpr (NR >= 2)
            asm volatile("vse32.v v2, (%0)" : : "r"(d_row + 1 * stride_col) : "memory");
    }
}
