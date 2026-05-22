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

/// Register allocation for 2x8vl GEMM kernel (LMUL=8):
///   ft0-ft1: A[0-1] (2 scalar registers)
///   v0-v7:   B matrix column (1 m8 vector group = 8 physical registers)
///   v8-v15:  Temporary for bias load (1 m8 group, reused)
///   v16-v23: Accumulator row D[0] (1 m8 group)
///   v24-v31: Accumulator row D[1] (1 m8 group)
///   ft8, ft9: clamp min/max scalars (loaded only at end)

/// Macro to initialize a single LMUL=8 vector register group to zero
#define TQT_VZERO_FP32_M8(reg_num)                \
    asm volatile(                                 \
        "vsetvli zero, zero, e32, m8, ta, ma\n\t" \
        "vmv.v.i v" #reg_num ", 0\n\t"            \
        :                                         \
        :                                         \
        : "v" #reg_num);

/// Initialize accumulator registers to zero (LMUL=8, MR max 2)
template <size_t MR>
static inline void tqt_init_zero_kernel_gemm_f32_f32_2x8vl_rvv(size_t vl)
{
    static_assert(MR >= 1 && MR <= 2, "MR must be in range [1, 2]");

    asm volatile("vsetvli zero, %0, e32, m8, ta, ma\n\t" : : "r"(vl) :);

    // Row 0: v16-v23 (m8 group)
    if constexpr (MR >= 1)
        TQT_VZERO_FP32_M8(16);
    // Row 1: v24-v31 (m8 group)
    if constexpr (MR >= 2)
        TQT_VZERO_FP32_M8(24);
}

/// Initialize accumulator registers from C matrix (LMUL=8)
template <size_t MR>
static inline void tqt_init_c_kernel_gemm_f32_f32_2x8vl_rvv(size_t vl, const float *C,
                                                            size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 2, "MR must be in range [1, 2]");

    asm volatile("vsetvli zero, %0, e32, m8, ta, ma\n\t" : : "r"(vl) :);

    if constexpr (MR >= 1) {
        const float *c_row_0 = C + 0 * stride_row;
        asm volatile("vle32.v v16, (%0)" : : "r"(c_row_0) : "v16", "memory");
    }
    if constexpr (MR >= 2) {
        const float *c_row_1 = C + 1 * stride_row;
        asm volatile("vle32.v v24, (%0)" : : "r"(c_row_1) : "v24", "memory");
    }
}

/// Load A matrix elements (scalars) into floating-point registers
template <size_t MR>
static inline void tqt_load_a_kernel_gemm_f32_f32_2x8vl_rvv(const float *A, size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 2, "MR must be in range [1, 2]");

    if constexpr (MR >= 1)
        asm volatile("flw ft0, 0(%0)" : : "r"(A + 0 * stride_row) : "ft0", "memory");
    if constexpr (MR >= 2)
        asm volatile("flw ft1, 0(%0)" : : "r"(A + 1 * stride_row) : "ft1", "memory");
}

/// Load B matrix column (vector) into v0-v7 (LMUL=8, contiguous load)
static inline void tqt_load_b_kernel_gemm_f32_f32_2x8vl_rvv(size_t vl, const float *B)
{
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vle32.v v0, (%0)" : : "r"(B) : "v0", "memory");
}

/// Load transposed B matrix column (vector) into v0-v7 (LMUL=8, strided load)
static inline void tqt_load_bt_kernel_gemm_f32_f32_2x8vl_rvv(size_t vl, const float *B,
                                                             size_t stride_row)
{
    size_t stride_bytes = stride_row * sizeof(float);
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vlse32.v v0, (%0), %1" : : "r"(B), "r"(stride_bytes) : "v0", "memory");
}

/// Perform VFMACC operations (LMUL=8, MR x 1)
template <size_t MR>
static inline void tqt_vfmacc_kernel_gemm_f32_f32_2x8vl_rvv()
{
    static_assert(MR >= 1 && MR <= 2, "MR must be in range [1, 2]");

    if constexpr (MR >= 1)
        asm volatile("vfmacc.vf v16, ft0, v0" : : : "v16");
    if constexpr (MR >= 2)
        asm volatile("vfmacc.vf v24, ft1, v0" : : : "v24");
}

/// Fused load + vfmacc kernel (A*B) with software pipelining (LMUL=8)
/// Computes current iteration's vfmacc while prefetching next iteration's A and B
template <size_t MR>
static inline void tqt_fused_load_vfmacc_axb_kernel_gemm_f32_f32_2x8vl_rvv(size_t vl,
                                                                           const float *A_next,
                                                                           size_t a_stride_row,
                                                                           const float *B_next)
{
    static_assert(MR >= 1 && MR <= 2, "MR must be in range [1, 2]");

    asm volatile("vsetvli zero, %0, e32, m8, ta, ma\n\t" : : "r"(vl) :);

    // vfmacc with v0
    if constexpr (MR >= 1)
        asm volatile("vfmacc.vf v16, ft0, v0" : : : "v16");
    if constexpr (MR >= 2)
        asm volatile("vfmacc.vf v24, ft1, v0" : : : "v24");

    // Prefetch next B (v0 is freed)
    asm volatile("vle32.v v0, (%0)" : : "r"(B_next) : "v0", "memory");

    // Prefetch next A scalars
    if constexpr (MR >= 1)
        asm volatile("flw ft0, 0(%0)" : : "r"(A_next + 0 * a_stride_row) : "ft0", "memory");
    if constexpr (MR >= 2)
        asm volatile("flw ft1, 0(%0)" : : "r"(A_next + 1 * a_stride_row) : "ft1", "memory");
}

/// Fused load + vfmacc kernel (A*B^T) with software pipelining (LMUL=8, strided B load)
template <size_t MR>
static inline void tqt_fused_load_vfmacc_axbt_kernel_gemm_f32_f32_2x8vl_rvv(
    size_t vl, const float *A_next, size_t a_stride_row, const float *B_next, size_t b_stride_row)
{
    static_assert(MR >= 1 && MR <= 2, "MR must be in range [1, 2]");

    size_t stride_bytes = b_stride_row * sizeof(float);
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma\n\t" : : "r"(vl) :);

    if constexpr (MR >= 1)
        asm volatile("vfmacc.vf v16, ft0, v0" : : : "v16");
    if constexpr (MR >= 2)
        asm volatile("vfmacc.vf v24, ft1, v0" : : : "v24");

    // Prefetch next B (strided)
    asm volatile("vlse32.v v0, (%0), %1" : : "r"(B_next), "r"(stride_bytes) : "v0", "memory");

    // Prefetch next A scalars
    if constexpr (MR >= 1)
        asm volatile("flw ft0, 0(%0)" : : "r"(A_next + 0 * a_stride_row) : "ft0", "memory");
    if constexpr (MR >= 2)
        asm volatile("flw ft1, 0(%0)" : : "r"(A_next + 1 * a_stride_row) : "ft1", "memory");
}

/// Epilogue vfmacc kernel: final iteration without prefetch (LMUL=8)
template <size_t MR>
static inline void tqt_epilogue_vfmacc_kernel_gemm_f32_f32_2x8vl_rvv()
{
    static_assert(MR >= 1 && MR <= 2, "MR must be in range [1, 2]");

    if constexpr (MR >= 1)
        asm volatile("vfmacc.vf v16, ft0, v0" : : : "v16");
    if constexpr (MR >= 2)
        asm volatile("vfmacc.vf v24, ft1, v0" : : : "v24");
}

/// Add 1xn bias to accumulator registers (LMUL=8)
template <size_t MR>
static inline void tqt_add_1xnbias_kernel_gemm_f32_f32_2x8vl_rvv(size_t vl, const float *BIAS)
{
    static_assert(MR >= 1 && MR <= 2, "MR must be in range [1, 2]");

    asm volatile("vsetvli zero, %0, e32, m8, ta, ma\n\t" : : "r"(vl) :);

    // Load bias into temporary m8 vector (v8-v15)
    asm volatile("vle32.v v8, (%0)" : : "r"(BIAS) : "v8", "memory");

    if constexpr (MR >= 1)
        asm volatile("vfadd.vv v16, v16, v8" : : : "v16");
    if constexpr (MR >= 2)
        asm volatile("vfadd.vv v24, v24, v8" : : : "v24");
}

/// Apply clamp (min/max) to accumulator registers (LMUL=8)
template <size_t MR>
static inline void tqt_clamp_kernel_gemm_f32_f32_2x8vl_rvv(size_t vl, float min, float max)
{
    static_assert(MR >= 1 && MR <= 2, "MR must be in range [1, 2]");

    asm volatile("vsetvli zero, %0, e32, m8, ta, ma\n\t" : : "r"(vl) :);

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
        asm volatile("vfmax.vf v24, v24, ft8" : : : "v24");
        asm volatile("vfmin.vf v24, v24, ft9" : : : "v24");
    }
}

/// Store accumulator registers to D matrix (LMUL=8)
template <size_t MR>
static inline void tqt_store_kernel_gemm_f32_f32_2x8vl_rvv(size_t vl, float *D, size_t stride_row)
{
    static_assert(MR >= 1 && MR <= 2, "MR must be in range [1, 2]");

    asm volatile("vsetvli zero, %0, e32, m8, ta, ma\n\t" : : "r"(vl) :);

    if constexpr (MR >= 1)
        asm volatile("vse32.v v16, (%0)" : : "r"(D + 0 * stride_row) : "memory");
    if constexpr (MR >= 2)
        asm volatile("vse32.v v24, (%0)" : : "r"(D + 1 * stride_row) : "memory");
}
