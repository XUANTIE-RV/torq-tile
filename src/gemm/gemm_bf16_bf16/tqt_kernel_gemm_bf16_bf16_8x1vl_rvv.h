//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zvfbfwma)
#error This file must be compiled for riscv, riscv_vector, zvfbfwma.
#endif  // Architectural features check.

#include <stddef.h>

#include "src/common/tqt_common.h"

/// Register allocation for 8x1vl GEMM kernel:
///   ft0-ft7: A[0-7]
///   v0:      B matrix columns (1 vector registers)
///   v8-v23:  Accumulator(fp32) matrix D[0-7][0] (16 vector registers)
///   v24-v31: Narrowed(bf16) matrix D[0-7][0] (8 vector registers)

/// Initialize accumulator registers to zero (fp32, e32m2)
template <size_t MR>
static inline void tqt_init_zero_kernel_gemm_bf16_bf16_8x1vl_rvv(size_t vl)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");

    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);

    if constexpr (MR >= 1) {
        asm volatile("vmv.v.i v8, 0" : : : "v8", "v9");
    }
    if constexpr (MR >= 2) {
        asm volatile("vmv.v.i v10, 0" : : : "v10", "v11");
    }
    if constexpr (MR >= 3) {
        asm volatile("vmv.v.i v12, 0" : : : "v12", "v13");
    }
    if constexpr (MR >= 4) {
        asm volatile("vmv.v.i v14, 0" : : : "v14", "v15");
    }
    if constexpr (MR >= 5) {
        asm volatile("vmv.v.i v16, 0" : : : "v16", "v17");
    }
    if constexpr (MR >= 6) {
        asm volatile("vmv.v.i v18, 0" : : : "v18", "v19");
    }
    if constexpr (MR >= 7) {
        asm volatile("vmv.v.i v20, 0" : : : "v20", "v21");
    }
    if constexpr (MR >= 8) {
        asm volatile("vmv.v.i v22, 0" : : : "v22", "v23");
    }
}

/// Initialize accumulator registers from C matrix (load bf16, widen to fp32)
template <size_t MR>
static inline void tqt_init_c_kernel_gemm_bf16_bf16_8x1vl_rvv(size_t vl, const bfloat16_t *C,
                                                              size_t stride_row, size_t stride_col)
{
    TQT_UNUSED(stride_col);
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    if constexpr (MR >= 1) {
        const bfloat16_t *c_row_0 = C + 0 * stride_row;
        asm volatile("vle16.v v24, (%0)" : : "r"(c_row_0) : "v24", "memory");
        asm volatile("vfwcvtbf16.f.f.v v8, v24" : : : "v8");
    }
    if constexpr (MR >= 2) {
        const bfloat16_t *c_row_1 = C + 1 * stride_row;
        asm volatile("vle16.v v25, (%0)" : : "r"(c_row_1) : "v25", "memory");
        asm volatile("vfwcvtbf16.f.f.v v10, v25" : : : "v10");
    }
    if constexpr (MR >= 3) {
        const bfloat16_t *c_row_2 = C + 2 * stride_row;
        asm volatile("vle16.v v26, (%0)" : : "r"(c_row_2) : "v26", "memory");
        asm volatile("vfwcvtbf16.f.f.v v12, v26" : : : "v12");
    }
    if constexpr (MR >= 4) {
        const bfloat16_t *c_row_3 = C + 3 * stride_row;
        asm volatile("vle16.v v27, (%0)" : : "r"(c_row_3) : "v27", "memory");
        asm volatile("vfwcvtbf16.f.f.v v14, v27" : : : "v14");
    }
    if constexpr (MR >= 5) {
        const bfloat16_t *c_row_4 = C + 4 * stride_row;
        asm volatile("vle16.v v28, (%0)" : : "r"(c_row_4) : "v28", "memory");
        asm volatile("vfwcvtbf16.f.f.v v16, v28" : : : "v16");
    }
    if constexpr (MR >= 6) {
        const bfloat16_t *c_row_5 = C + 5 * stride_row;
        asm volatile("vle16.v v29, (%0)" : : "r"(c_row_5) : "v29", "memory");
        asm volatile("vfwcvtbf16.f.f.v v18, v29" : : : "v18");
    }
    if constexpr (MR >= 7) {
        const bfloat16_t *c_row_6 = C + 6 * stride_row;
        asm volatile("vle16.v v30, (%0)" : : "r"(c_row_6) : "v30", "memory");
        asm volatile("vfwcvtbf16.f.f.v v20, v30" : : : "v20");
    }
    if constexpr (MR >= 8) {
        const bfloat16_t *c_row_7 = C + 7 * stride_row;
        asm volatile("vle16.v v31, (%0)" : : "r"(c_row_7) : "v31", "memory");
        asm volatile("vfwcvtbf16.f.f.v v22, v31" : : : "v22");
    }
}

/// Load A matrix elements (scalars) into floating-point registers
template <size_t MR>
static inline void tqt_load_a_kernel_gemm_bf16_bf16_8x1vl_rvv(const bfloat16_t *A,
                                                              size_t stride_row)
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

/// Load B matrix elements (non-transposed, row-major) into v0
static inline void tqt_load_b_kernel_gemm_bf16_bf16_8x1vl_rvv(size_t vl, const bfloat16_t *B)
{
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
    asm volatile("vle16.v v0, (%0)" : : "r"(B) : "v0", "memory");
}

/// Load B matrix elements (transposed) into v0
static inline void tqt_load_bt_kernel_gemm_bf16_bf16_8x1vl_rvv(size_t vl, const bfloat16_t *B,
                                                               size_t stride_row)
{
    size_t stride_bytes = stride_row * sizeof(bfloat16_t);
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);
    asm volatile("vlse16.v v0, (%0), %1" : : "r"(B), "r"(stride_bytes) : "v0", "memory");
}

/// Perform widening FMA: vfwmaccbf16.vf vd, ftX, v0
template <size_t MR>
static inline void tqt_vfwmacc_kernel_gemm_bf16_bf16_8x1vl_rvv(size_t vl)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    if constexpr (MR >= 1)
        asm volatile("vfwmaccbf16.vf v8, ft0, v0" : : : "v8", "v9");
    if constexpr (MR >= 2)
        asm volatile("vfwmaccbf16.vf v10, ft1, v0" : : : "v10", "v11");
    if constexpr (MR >= 3)
        asm volatile("vfwmaccbf16.vf v12, ft2, v0" : : : "v12", "v13");
    if constexpr (MR >= 4)
        asm volatile("vfwmaccbf16.vf v14, ft3, v0" : : : "v14", "v15");
    if constexpr (MR >= 5)
        asm volatile("vfwmaccbf16.vf v16, ft4, v0" : : : "v16", "v17");
    if constexpr (MR >= 6)
        asm volatile("vfwmaccbf16.vf v18, ft5, v0" : : : "v18", "v19");
    if constexpr (MR >= 7)
        asm volatile("vfwmaccbf16.vf v20, ft6, v0" : : : "v20", "v21");
    if constexpr (MR >= 8)
        asm volatile("vfwmaccbf16.vf v22, ft7, v0" : : : "v22", "v23");
}

/// Add 1xN bias to fp32 registers
template <size_t MR>
static inline void tqt_add_1xnbias_kernel_gemm_bf16_bf16_8x1vl_rvv(size_t vl,
                                                                   const bfloat16_t *BIAS)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");

    asm volatile(
        "vsetvli            zero, %0, e16, m1, ta, ma\n\t"
        "vle16.v            v1, (%1)\n\t"
        "vfwcvtbf16.f.f.v   v0, v1\n\t"
        "vsetvli            zero, %0, e32, m2, ta, ma\n\t"
        :
        : "r"(vl), "r"(BIAS)
        : "v0", "v1", "memory");

    if constexpr (MR >= 1)
        asm volatile("vfadd.vv v8, v8, v0" : : : "v8");
    if constexpr (MR >= 2)
        asm volatile("vfadd.vv v10, v10, v0" : : : "v10");
    if constexpr (MR >= 3)
        asm volatile("vfadd.vv v12, v12, v0" : : : "v12");
    if constexpr (MR >= 4)
        asm volatile("vfadd.vv v14, v14, v0" : : : "v14");
    if constexpr (MR >= 5)
        asm volatile("vfadd.vv v16, v16, v0" : : : "v16");
    if constexpr (MR >= 6)
        asm volatile("vfadd.vv v18, v18, v0" : : : "v18");
    if constexpr (MR >= 7)
        asm volatile("vfadd.vv v20, v20, v0" : : : "v20");
    if constexpr (MR >= 8)
        asm volatile("vfadd.vv v22, v22, v0" : : : "v22");
}

/// Apply clamp (min/max) to fp32 registers
template <size_t MR>
static inline void tqt_clamp_kernel_gemm_bf16_bf16_8x1vl_rvv(size_t vl, bfloat16_t min,
                                                             bfloat16_t max)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");

    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);

    asm volatile(
        "flh ft8, 0(%0)\n\t"
        "flh ft9, 0(%1)\n\t"
        :
        : "r"(&min), "r"(&max)
        : "ft8", "ft9", "memory");

    if constexpr (MR >= 1) {
        asm volatile("vfmax.vf v8, v8, ft8" : : : "v8");
        asm volatile("vfmin.vf v8, v8, ft9" : : : "v8");
    }
    if constexpr (MR >= 2) {
        asm volatile("vfmax.vf v10, v10, ft8" : : : "v10");
        asm volatile("vfmin.vf v10, v10, ft9" : : : "v10");
    }
    if constexpr (MR >= 3) {
        asm volatile("vfmax.vf v12, v12, ft8" : : : "v12");
        asm volatile("vfmin.vf v12, v12, ft9" : : : "v12");
    }
    if constexpr (MR >= 4) {
        asm volatile("vfmax.vf v14, v14, ft8" : : : "v14");
        asm volatile("vfmin.vf v14, v14, ft9" : : : "v14");
    }
    if constexpr (MR >= 5) {
        asm volatile("vfmax.vf v16, v16, ft8" : : : "v16");
        asm volatile("vfmin.vf v16, v16, ft9" : : : "v16");
    }
    if constexpr (MR >= 6) {
        asm volatile("vfmax.vf v18, v18, ft8" : : : "v18");
        asm volatile("vfmin.vf v18, v18, ft9" : : : "v18");
    }
    if constexpr (MR >= 7) {
        asm volatile("vfmax.vf v20, v20, ft8" : : : "v20");
        asm volatile("vfmin.vf v20, v20, ft9" : : : "v20");
    }
    if constexpr (MR >= 8) {
        asm volatile("vfmax.vf v22, v22, ft8" : : : "v22");
        asm volatile("vfmin.vf v22, v22, ft9" : : : "v22");
    }
}

/// Narrow fp32 accumulators back to bf16
template <size_t MR>
static inline void tqt_cvt_fp32_to_bf16_kernel_gemm_bf16_bf16_8x1vl_rvv(size_t vl)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    if constexpr (MR >= 1)
        asm volatile("vfncvtbf16.f.f.w v24, v8" : : : "v24");
    if constexpr (MR >= 2)
        asm volatile("vfncvtbf16.f.f.w v25, v10" : : : "v25");
    if constexpr (MR >= 3)
        asm volatile("vfncvtbf16.f.f.w v26, v12" : : : "v26");
    if constexpr (MR >= 4)
        asm volatile("vfncvtbf16.f.f.w v27, v14" : : : "v27");
    if constexpr (MR >= 5)
        asm volatile("vfncvtbf16.f.f.w v28, v16" : : : "v28");
    if constexpr (MR >= 6)
        asm volatile("vfncvtbf16.f.f.w v29, v18" : : : "v29");
    if constexpr (MR >= 7)
        asm volatile("vfncvtbf16.f.f.w v30, v20" : : : "v30");
    if constexpr (MR >= 8)
        asm volatile("vfncvtbf16.f.f.w v31, v22" : : : "v31");
}

/// Store narrowed bf16 registers to D matrix
template <size_t MR>
static inline void tqt_store_kernel_gemm_bf16_bf16_8x1vl_rvv(size_t vl, bfloat16_t *D,
                                                             size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    if constexpr (MR >= 1)
        asm volatile("vse16.v v24, (%0)" : : "r"(D + 0 * stride_row) : "memory");
    if constexpr (MR >= 2)
        asm volatile("vse16.v v25, (%0)" : : "r"(D + 1 * stride_row) : "memory");
    if constexpr (MR >= 3)
        asm volatile("vse16.v v26, (%0)" : : "r"(D + 2 * stride_row) : "memory");
    if constexpr (MR >= 4)
        asm volatile("vse16.v v27, (%0)" : : "r"(D + 3 * stride_row) : "memory");
    if constexpr (MR >= 5)
        asm volatile("vse16.v v28, (%0)" : : "r"(D + 4 * stride_row) : "memory");
    if constexpr (MR >= 6)
        asm volatile("vse16.v v29, (%0)" : : "r"(D + 5 * stride_row) : "memory");
    if constexpr (MR >= 7)
        asm volatile("vse16.v v30, (%0)" : : "r"(D + 6 * stride_row) : "memory");
    if constexpr (MR >= 8)
        asm volatile("vse16.v v31, (%0)" : : "r"(D + 7 * stride_row) : "memory");
}

// ============================================================================
// D=f16 helpers: load f16 C/bias and widen to fp32 acc; narrow fp32 acc to f16
// ============================================================================

/// Initialize accumulator registers from C matrix (load f16, widen to fp32)
template <size_t MR>
static inline void tqt_init_c_f16_kernel_gemm_bf16_bf16_8x1vl_rvv(size_t vl, const float16_t *C,
                                                                  size_t stride_row,
                                                                  size_t stride_col)
{
    TQT_UNUSED(stride_col);
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    if constexpr (MR >= 1) {
        const float16_t *c_row_0 = C + 0 * stride_row;
        asm volatile("vle16.v v24, (%0)" : : "r"(c_row_0) : "v24", "memory");
        asm volatile("vfwcvt.f.f.v v8, v24" : : : "v8");
    }
    if constexpr (MR >= 2) {
        const float16_t *c_row_1 = C + 1 * stride_row;
        asm volatile("vle16.v v25, (%0)" : : "r"(c_row_1) : "v25", "memory");
        asm volatile("vfwcvt.f.f.v v10, v25" : : : "v10");
    }
    if constexpr (MR >= 3) {
        const float16_t *c_row_2 = C + 2 * stride_row;
        asm volatile("vle16.v v26, (%0)" : : "r"(c_row_2) : "v26", "memory");
        asm volatile("vfwcvt.f.f.v v12, v26" : : : "v12");
    }
    if constexpr (MR >= 4) {
        const float16_t *c_row_3 = C + 3 * stride_row;
        asm volatile("vle16.v v27, (%0)" : : "r"(c_row_3) : "v27", "memory");
        asm volatile("vfwcvt.f.f.v v14, v27" : : : "v14");
    }
    if constexpr (MR >= 5) {
        const float16_t *c_row_4 = C + 4 * stride_row;
        asm volatile("vle16.v v28, (%0)" : : "r"(c_row_4) : "v28", "memory");
        asm volatile("vfwcvt.f.f.v v16, v28" : : : "v16");
    }
    if constexpr (MR >= 6) {
        const float16_t *c_row_5 = C + 5 * stride_row;
        asm volatile("vle16.v v29, (%0)" : : "r"(c_row_5) : "v29", "memory");
        asm volatile("vfwcvt.f.f.v v18, v29" : : : "v18");
    }
    if constexpr (MR >= 7) {
        const float16_t *c_row_6 = C + 6 * stride_row;
        asm volatile("vle16.v v30, (%0)" : : "r"(c_row_6) : "v30", "memory");
        asm volatile("vfwcvt.f.f.v v20, v30" : : : "v20");
    }
    if constexpr (MR >= 8) {
        const float16_t *c_row_7 = C + 7 * stride_row;
        asm volatile("vle16.v v31, (%0)" : : "r"(c_row_7) : "v31", "memory");
        asm volatile("vfwcvt.f.f.v v22, v31" : : : "v22");
    }
}

/// Add 1xN f16 bias (widened to fp32) to fp32 accumulator registers
template <size_t MR>
static inline void tqt_add_1xnbias_f16_kernel_gemm_bf16_bf16_8x1vl_rvv(size_t vl,
                                                                       const float16_t *BIAS)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");

    asm volatile(
        "vsetvli            zero, %0, e16, m1, ta, ma\n\t"
        "vle16.v            v1, (%1)\n\t"
        "vfwcvt.f.f.v       v0, v1\n\t"
        "vsetvli            zero, %0, e32, m2, ta, ma\n\t"
        :
        : "r"(vl), "r"(BIAS)
        : "v0", "v1", "memory");

    if constexpr (MR >= 1)
        asm volatile("vfadd.vv v8, v8, v0" : : : "v8");
    if constexpr (MR >= 2)
        asm volatile("vfadd.vv v10, v10, v0" : : : "v10");
    if constexpr (MR >= 3)
        asm volatile("vfadd.vv v12, v12, v0" : : : "v12");
    if constexpr (MR >= 4)
        asm volatile("vfadd.vv v14, v14, v0" : : : "v14");
    if constexpr (MR >= 5)
        asm volatile("vfadd.vv v16, v16, v0" : : : "v16");
    if constexpr (MR >= 6)
        asm volatile("vfadd.vv v18, v18, v0" : : : "v18");
    if constexpr (MR >= 7)
        asm volatile("vfadd.vv v20, v20, v0" : : : "v20");
    if constexpr (MR >= 8)
        asm volatile("vfadd.vv v22, v22, v0" : : : "v22");
}

/// Narrow fp32 accumulators back to f16 (IEEE half precision)
template <size_t MR>
static inline void tqt_cvt_fp32_to_f16_kernel_gemm_bf16_bf16_8x1vl_rvv(size_t vl)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    if constexpr (MR >= 1)
        asm volatile("vfncvt.f.f.w v24, v8" : : : "v24");
    if constexpr (MR >= 2)
        asm volatile("vfncvt.f.f.w v25, v10" : : : "v25");
    if constexpr (MR >= 3)
        asm volatile("vfncvt.f.f.w v26, v12" : : : "v26");
    if constexpr (MR >= 4)
        asm volatile("vfncvt.f.f.w v27, v14" : : : "v27");
    if constexpr (MR >= 5)
        asm volatile("vfncvt.f.f.w v28, v16" : : : "v28");
    if constexpr (MR >= 6)
        asm volatile("vfncvt.f.f.w v29, v18" : : : "v29");
    if constexpr (MR >= 7)
        asm volatile("vfncvt.f.f.w v30, v20" : : : "v30");
    if constexpr (MR >= 8)
        asm volatile("vfncvt.f.f.w v31, v22" : : : "v31");
}

/// Store narrowed f16 registers to D matrix
template <size_t MR>
static inline void tqt_store_f16_kernel_gemm_bf16_bf16_8x1vl_rvv(size_t vl, float16_t *D,
                                                                 size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    if constexpr (MR >= 1)
        asm volatile("vse16.v v24, (%0)" : : "r"(D + 0 * stride_row) : "memory");
    if constexpr (MR >= 2)
        asm volatile("vse16.v v25, (%0)" : : "r"(D + 1 * stride_row) : "memory");
    if constexpr (MR >= 3)
        asm volatile("vse16.v v26, (%0)" : : "r"(D + 2 * stride_row) : "memory");
    if constexpr (MR >= 4)
        asm volatile("vse16.v v27, (%0)" : : "r"(D + 3 * stride_row) : "memory");
    if constexpr (MR >= 5)
        asm volatile("vse16.v v28, (%0)" : : "r"(D + 4 * stride_row) : "memory");
    if constexpr (MR >= 6)
        asm volatile("vse16.v v29, (%0)" : : "r"(D + 5 * stride_row) : "memory");
    if constexpr (MR >= 7)
        asm volatile("vse16.v v30, (%0)" : : "r"(D + 6 * stride_row) : "memory");
    if constexpr (MR >= 8)
        asm volatile("vse16.v v31, (%0)" : : "r"(D + 7 * stride_row) : "memory");
}

// ============================================================================
// D=f32 helpers: directly load/store fp32 C/bias/D from/to fp32 acc registers
// ============================================================================

/// Initialize accumulator registers from C matrix (load fp32 directly into acc)
template <size_t MR>
static inline void tqt_init_c_f32_kernel_gemm_bf16_bf16_8x1vl_rvv(size_t vl, const float *C,
                                                                  size_t stride_row,
                                                                  size_t stride_col)
{
    TQT_UNUSED(stride_col);
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");

    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);

    if constexpr (MR >= 1)
        asm volatile("vle32.v v8, (%0)" : : "r"(C + 0 * stride_row) : "v8", "v9", "memory");
    if constexpr (MR >= 2)
        asm volatile("vle32.v v10, (%0)" : : "r"(C + 1 * stride_row) : "v10", "v11", "memory");
    if constexpr (MR >= 3)
        asm volatile("vle32.v v12, (%0)" : : "r"(C + 2 * stride_row) : "v12", "v13", "memory");
    if constexpr (MR >= 4)
        asm volatile("vle32.v v14, (%0)" : : "r"(C + 3 * stride_row) : "v14", "v15", "memory");
    if constexpr (MR >= 5)
        asm volatile("vle32.v v16, (%0)" : : "r"(C + 4 * stride_row) : "v16", "v17", "memory");
    if constexpr (MR >= 6)
        asm volatile("vle32.v v18, (%0)" : : "r"(C + 5 * stride_row) : "v18", "v19", "memory");
    if constexpr (MR >= 7)
        asm volatile("vle32.v v20, (%0)" : : "r"(C + 6 * stride_row) : "v20", "v21", "memory");
    if constexpr (MR >= 8)
        asm volatile("vle32.v v22, (%0)" : : "r"(C + 7 * stride_row) : "v22", "v23", "memory");
}

/// Add 1xN fp32 bias to fp32 accumulator registers
template <size_t MR>
static inline void tqt_add_1xnbias_f32_kernel_gemm_bf16_bf16_8x1vl_rvv(size_t vl, const float *BIAS)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");

    asm volatile(
        "vsetvli            zero, %0, e32, m2, ta, ma\n\t"
        "vle32.v            v0, (%1)\n\t"
        :
        : "r"(vl), "r"(BIAS)
        : "v0", "v1", "memory");

    if constexpr (MR >= 1)
        asm volatile("vfadd.vv v8, v8, v0" : : : "v8");
    if constexpr (MR >= 2)
        asm volatile("vfadd.vv v10, v10, v0" : : : "v10");
    if constexpr (MR >= 3)
        asm volatile("vfadd.vv v12, v12, v0" : : : "v12");
    if constexpr (MR >= 4)
        asm volatile("vfadd.vv v14, v14, v0" : : : "v14");
    if constexpr (MR >= 5)
        asm volatile("vfadd.vv v16, v16, v0" : : : "v16");
    if constexpr (MR >= 6)
        asm volatile("vfadd.vv v18, v18, v0" : : : "v18");
    if constexpr (MR >= 7)
        asm volatile("vfadd.vv v20, v20, v0" : : : "v20");
    if constexpr (MR >= 8)
        asm volatile("vfadd.vv v22, v22, v0" : : : "v22");
}

/// Store fp32 accumulator registers directly to D matrix (no narrow needed)
template <size_t MR>
static inline void tqt_store_f32_kernel_gemm_bf16_bf16_8x1vl_rvv(size_t vl, float *D,
                                                                 size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");

    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);

    if constexpr (MR >= 1)
        asm volatile("vse32.v v8, (%0)" : : "r"(D + 0 * stride_row) : "memory");
    if constexpr (MR >= 2)
        asm volatile("vse32.v v10, (%0)" : : "r"(D + 1 * stride_row) : "memory");
    if constexpr (MR >= 3)
        asm volatile("vse32.v v12, (%0)" : : "r"(D + 2 * stride_row) : "memory");
    if constexpr (MR >= 4)
        asm volatile("vse32.v v14, (%0)" : : "r"(D + 3 * stride_row) : "memory");
    if constexpr (MR >= 5)
        asm volatile("vse32.v v16, (%0)" : : "r"(D + 4 * stride_row) : "memory");
    if constexpr (MR >= 6)
        asm volatile("vse32.v v18, (%0)" : : "r"(D + 5 * stride_row) : "memory");
    if constexpr (MR >= 7)
        asm volatile("vse32.v v20, (%0)" : : "r"(D + 6 * stride_row) : "memory");
    if constexpr (MR >= 8)
        asm volatile("vse32.v v22, (%0)" : : "r"(D + 7 * stride_row) : "memory");
}

// ============================================================================
// Generic fp32-scalar clamp (used by D=f16 and D=f32 paths)
// ============================================================================

/// Apply clamp using fp32 scalar min/max (caller widens f16 to float when D=f16)
template <size_t MR>
static inline void tqt_clamp_fp32scalar_kernel_gemm_bf16_bf16_8x1vl_rvv(size_t vl, float min,
                                                                        float max)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");

    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl) :);

    asm volatile(
        "flw ft8, 0(%0)\n\t"
        "flw ft9, 0(%1)\n\t"
        :
        : "r"(&min), "r"(&max)
        : "ft8", "ft9", "memory");

    if constexpr (MR >= 1) {
        asm volatile("vfmax.vf v8, v8, ft8" : : : "v8");
        asm volatile("vfmin.vf v8, v8, ft9" : : : "v8");
    }
    if constexpr (MR >= 2) {
        asm volatile("vfmax.vf v10, v10, ft8" : : : "v10");
        asm volatile("vfmin.vf v10, v10, ft9" : : : "v10");
    }
    if constexpr (MR >= 3) {
        asm volatile("vfmax.vf v12, v12, ft8" : : : "v12");
        asm volatile("vfmin.vf v12, v12, ft9" : : : "v12");
    }
    if constexpr (MR >= 4) {
        asm volatile("vfmax.vf v14, v14, ft8" : : : "v14");
        asm volatile("vfmin.vf v14, v14, ft9" : : : "v14");
    }
    if constexpr (MR >= 5) {
        asm volatile("vfmax.vf v16, v16, ft8" : : : "v16");
        asm volatile("vfmin.vf v16, v16, ft9" : : : "v16");
    }
    if constexpr (MR >= 6) {
        asm volatile("vfmax.vf v18, v18, ft8" : : : "v18");
        asm volatile("vfmin.vf v18, v18, ft9" : : : "v18");
    }
    if constexpr (MR >= 7) {
        asm volatile("vfmax.vf v20, v20, ft8" : : : "v20");
        asm volatile("vfmin.vf v20, v20, ft9" : : : "v20");
    }
    if constexpr (MR >= 8) {
        asm volatile("vfmax.vf v22, v22, ft8" : : : "v22");
        asm volatile("vfmin.vf v22, v22, ft9" : : : "v22");
    }
}
