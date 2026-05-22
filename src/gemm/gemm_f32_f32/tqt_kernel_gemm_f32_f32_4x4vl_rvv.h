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

/// Register allocation for 4x4vl GEMM kernel (LMUL=4):
///   ft0-ft3: A[0-3] (4 scalar registers)
///   v0-v3:   B matrix column (1 m4 vector group = 4 physical registers)
///   v16-v31: Accumulator matrix D[0-3] (4 m4 vector groups = 16 physical registers)
///   v4-v7:   Temporary for bias load (1 m4 group, reused)

/// Macro to initialize a single LMUL=4 vector register group to zero
#define TQT_VZERO_FP32_M4(reg_num)                \
    asm volatile(                                 \
        "vsetvli zero, zero, e32, m4, ta, ma\n\t" \
        "vmv.v.i v" #reg_num ", 0\n\t"            \
        :                                         \
        :                                         \
        : "v" #reg_num);

/// Initialize accumulator registers to zero (LMUL=4, MR max 4)
template <size_t MR>
static inline void tqt_init_zero_kernel_gemm_f32_f32_4x4vl_rvv(size_t vl)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    // Row 0: v16-v19 (m4 group)
    if constexpr (MR >= 1)
        TQT_VZERO_FP32_M4(16);
    // Row 1: v20-v23 (m4 group)
    if constexpr (MR >= 2)
        TQT_VZERO_FP32_M4(20);
    // Row 2: v24-v27 (m4 group)
    if constexpr (MR >= 3)
        TQT_VZERO_FP32_M4(24);
    // Row 3: v28-v31 (m4 group)
    if constexpr (MR >= 4)
        TQT_VZERO_FP32_M4(28);
}

/// Initialize accumulator registers from C matrix (LMUL=4)
template <size_t MR>
static inline void tqt_init_c_kernel_gemm_f32_f32_4x4vl_rvv(size_t vl, const float *C,
                                                            size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    // Each row loads vl*4 elements (one m4 vector)
    if constexpr (MR >= 1) {
        const float *c_row_0 = C + 0 * stride_row;
        asm volatile("vle32.v v16, (%0)" : : "r"(c_row_0) : "v16", "memory");
    }
    if constexpr (MR >= 2) {
        const float *c_row_1 = C + 1 * stride_row;
        asm volatile("vle32.v v20, (%0)" : : "r"(c_row_1) : "v20", "memory");
    }
    if constexpr (MR >= 3) {
        const float *c_row_2 = C + 2 * stride_row;
        asm volatile("vle32.v v24, (%0)" : : "r"(c_row_2) : "v24", "memory");
    }
    if constexpr (MR >= 4) {
        const float *c_row_3 = C + 3 * stride_row;
        asm volatile("vle32.v v28, (%0)" : : "r"(c_row_3) : "v28", "memory");
    }
}

/// Load A matrix elements (scalars) into floating-point registers
template <size_t MR>
static inline void tqt_load_a_kernel_gemm_f32_f32_4x4vl_rvv(const float *A, size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    if constexpr (MR >= 1)
        asm volatile("flw ft0, 0(%0)" : : "r"(A + 0 * stride_row) : "ft0", "memory");
    if constexpr (MR >= 2)
        asm volatile("flw ft1, 0(%0)" : : "r"(A + 1 * stride_row) : "ft1", "memory");
    if constexpr (MR >= 3)
        asm volatile("flw ft2, 0(%0)" : : "r"(A + 2 * stride_row) : "ft2", "memory");
    if constexpr (MR >= 4)
        asm volatile("flw ft3, 0(%0)" : : "r"(A + 3 * stride_row) : "ft3", "memory");
}

/// Load B matrix column (vector) into v0-v3 (LMUL=4, contiguous load)
static inline void tqt_load_b_kernel_gemm_f32_f32_4x4vl_rvv(size_t vl, const float *B)
{
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vle32.v v0, (%0)" : : "r"(B) : "v0", "memory");
}

/// Load transposed B matrix column (vector) into v0-v3 (LMUL=4, strided load)
static inline void tqt_load_bt_kernel_gemm_f32_f32_4x4vl_rvv(size_t vl, const float *B,
                                                             size_t stride_row)
{
    size_t stride_bytes = stride_row * sizeof(float);
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vlse32.v v0, (%0), %1" : : "r"(B), "r"(stride_bytes) : "v0", "memory");
}

/// Perform VFMACC operations (LMUL=4, MR x 1)
template <size_t MR>
static inline void tqt_vfmacc_kernel_gemm_f32_f32_4x4vl_rvv()
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    // Each row's accumulator (m4) multiplies with A scalar and B vector (v0)
    if constexpr (MR >= 1)
        asm volatile("vfmacc.vf v16, ft0, v0" : : : "v16");
    if constexpr (MR >= 2)
        asm volatile("vfmacc.vf v20, ft1, v0" : : : "v20");
    if constexpr (MR >= 3)
        asm volatile("vfmacc.vf v24, ft2, v0" : : : "v24");
    if constexpr (MR >= 4)
        asm volatile("vfmacc.vf v28, ft3, v0" : : : "v28");
}

/// Fused load + vfmacc kernel (A*B) with software pipelining (LMUL=4)
/// Computes current iteration's vfmacc while prefetching next iteration's A and B
template <size_t MR>
static inline void tqt_fused_load_vfmacc_axb_kernel_gemm_f32_f32_4x4vl_rvv(size_t vl,
                                                                           const float *A_next,
                                                                           size_t a_stride_row,
                                                                           const float *B_next)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    // vfmacc with v0, then prefetch next B
    if constexpr (MR >= 1)
        asm volatile("vfmacc.vf v16, ft0, v0" : : : "v16");
    if constexpr (MR >= 2)
        asm volatile("vfmacc.vf v20, ft1, v0" : : : "v20");
    if constexpr (MR >= 3)
        asm volatile("vfmacc.vf v24, ft2, v0" : : : "v24");
    if constexpr (MR >= 4)
        asm volatile("vfmacc.vf v28, ft3, v0" : : : "v28");

    // Prefetch next B (v0 is freed)
    asm volatile("vle32.v v0, (%0)" : : "r"(B_next) : "v0", "memory");

    // Prefetch next A scalars
    if constexpr (MR >= 1)
        asm volatile("flw ft0, 0(%0)" : : "r"(A_next + 0 * a_stride_row) : "ft0", "memory");
    if constexpr (MR >= 2)
        asm volatile("flw ft1, 0(%0)" : : "r"(A_next + 1 * a_stride_row) : "ft1", "memory");
    if constexpr (MR >= 3)
        asm volatile("flw ft2, 0(%0)" : : "r"(A_next + 2 * a_stride_row) : "ft2", "memory");
    if constexpr (MR >= 4)
        asm volatile("flw ft3, 0(%0)" : : "r"(A_next + 3 * a_stride_row) : "ft3", "memory");
}

/// Fused load + vfmacc kernel (A*B^T) with software pipelining (LMUL=4, strided B load)
template <size_t MR>
static inline void tqt_fused_load_vfmacc_axbt_kernel_gemm_f32_f32_4x4vl_rvv(
    size_t vl, const float *A_next, size_t a_stride_row, const float *B_next, size_t b_stride_row)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    size_t stride_bytes = b_stride_row * sizeof(float);
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    // vfmacc with v0
    if constexpr (MR >= 1)
        asm volatile("vfmacc.vf v16, ft0, v0" : : : "v16");
    if constexpr (MR >= 2)
        asm volatile("vfmacc.vf v20, ft1, v0" : : : "v20");
    if constexpr (MR >= 3)
        asm volatile("vfmacc.vf v24, ft2, v0" : : : "v24");
    if constexpr (MR >= 4)
        asm volatile("vfmacc.vf v28, ft3, v0" : : : "v28");

    // Prefetch next B (strided)
    asm volatile("vlse32.v v0, (%0), %1" : : "r"(B_next), "r"(stride_bytes) : "v0", "memory");

    // Prefetch next A scalars
    if constexpr (MR >= 1)
        asm volatile("flw ft0, 0(%0)" : : "r"(A_next + 0 * a_stride_row) : "ft0", "memory");
    if constexpr (MR >= 2)
        asm volatile("flw ft1, 0(%0)" : : "r"(A_next + 1 * a_stride_row) : "ft1", "memory");
    if constexpr (MR >= 3)
        asm volatile("flw ft2, 0(%0)" : : "r"(A_next + 2 * a_stride_row) : "ft2", "memory");
    if constexpr (MR >= 4)
        asm volatile("flw ft3, 0(%0)" : : "r"(A_next + 3 * a_stride_row) : "ft3", "memory");
}

/// Epilogue vfmacc kernel: final iteration without prefetch (LMUL=4)
template <size_t MR>
static inline void tqt_epilogue_vfmacc_kernel_gemm_f32_f32_4x4vl_rvv()
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    if constexpr (MR >= 1)
        asm volatile("vfmacc.vf v16, ft0, v0" : : : "v16");
    if constexpr (MR >= 2)
        asm volatile("vfmacc.vf v20, ft1, v0" : : : "v20");
    if constexpr (MR >= 3)
        asm volatile("vfmacc.vf v24, ft2, v0" : : : "v24");
    if constexpr (MR >= 4)
        asm volatile("vfmacc.vf v28, ft3, v0" : : : "v28");
}

/// Add 1xn bias to accumulator registers (LMUL=4)
template <size_t MR>
static inline void tqt_add_1xnbias_kernel_gemm_f32_f32_4x4vl_rvv(size_t vl, const float *BIAS)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    // Load bias into temporary m4 vector (v4-v7)
    asm volatile("vle32.v v4, (%0)" : : "r"(BIAS) : "v4", "memory");

    // Add bias to each row's accumulator
    if constexpr (MR >= 1)
        asm volatile("vfadd.vv v16, v16, v4" : : : "v16");
    if constexpr (MR >= 2)
        asm volatile("vfadd.vv v20, v20, v4" : : : "v20");
    if constexpr (MR >= 3)
        asm volatile("vfadd.vv v24, v24, v4" : : : "v24");
    if constexpr (MR >= 4)
        asm volatile("vfadd.vv v28, v28, v4" : : : "v28");
}

/// Apply clamp (min/max) to accumulator registers (LMUL=4)
template <size_t MR>
static inline void tqt_clamp_kernel_gemm_f32_f32_4x4vl_rvv(size_t vl, float min, float max)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    // Load min/max into ft8, ft9 (temp scalar registers, not used by A)
    asm volatile(
        "flw ft8, 0(%0)\n\t"
        "flw ft9, 0(%1)\n\t"
        :
        : "r"(&min), "r"(&max)
        : "ft8", "ft9", "memory");

    if constexpr (MR >= 1) {
        asm volatile("vfmax.vf v16, v16, ft8" : : : "v16");
        asm volatile("vfmin.vf v16, v16, ft9" : : : "v16");
    }
    if constexpr (MR >= 2) {
        asm volatile("vfmax.vf v20, v20, ft8" : : : "v20");
        asm volatile("vfmin.vf v20, v20, ft9" : : : "v20");
    }
    if constexpr (MR >= 3) {
        asm volatile("vfmax.vf v24, v24, ft8" : : : "v24");
        asm volatile("vfmin.vf v24, v24, ft9" : : : "v24");
    }
    if constexpr (MR >= 4) {
        asm volatile("vfmax.vf v28, v28, ft8" : : : "v28");
        asm volatile("vfmin.vf v28, v28, ft9" : : : "v28");
    }
}

/// Store accumulator registers to D matrix (LMUL=4)
template <size_t MR>
static inline void tqt_store_kernel_gemm_f32_f32_4x4vl_rvv(size_t vl, float *D, size_t stride_row)
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
