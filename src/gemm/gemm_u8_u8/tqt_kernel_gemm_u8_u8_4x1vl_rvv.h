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

/// Register allocation for u8 x u8 GEMM kernel (4x1vl tile):
///   a0-a3:   A[0-3] scalars (loaded with lbu for unsigned)
///   v0:      B matrix columns (uint8 lmul=1)
///   v8-v15:  Temp uint16 products (lmul=2)
///   v16-v31: Accumulator D[0-3][0] (uint32/int32 lmul=4)
///   v0-v3:   Temp for bias/dequant operations
///   v4-v7:   Temp for store

template <size_t MR>
static inline void tqt_init_zero_kernel_gemm_u8_u8_4x1vl_rvv(size_t vl)
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

template <size_t MR>
static inline void tqt_init_c_kernel_gemm_u8_u8_4x1vl_rvv(size_t vl, const int32_t *C,
                                                          size_t stride_row, size_t stride_col)
{
    TQT_UNUSED(stride_col);
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    if constexpr (MR >= 1)
        asm volatile("vle32.v v16, (%0)"
                     :
                     : "r"(C + 0 * stride_row)
                     : "v16", "v17", "v18", "v19", "memory");
    if constexpr (MR >= 2)
        asm volatile("vle32.v v20, (%0)"
                     :
                     : "r"(C + 1 * stride_row)
                     : "v20", "v21", "v22", "v23", "memory");
    if constexpr (MR >= 3)
        asm volatile("vle32.v v24, (%0)"
                     :
                     : "r"(C + 2 * stride_row)
                     : "v24", "v25", "v26", "v27", "memory");
    if constexpr (MR >= 4)
        asm volatile("vle32.v v28, (%0)"
                     :
                     : "r"(C + 3 * stride_row)
                     : "v28", "v29", "v30", "v31", "memory");
}

template <size_t MR>
static inline void tqt_get_a_ptr_gemm_u8_u8_4x1vl_rvv(const uint8_t *A, size_t stride_row,
                                                      const uint8_t *a_ptrs[MR])
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

static inline void tqt_load_bt_kernel_gemm_u8_u8_4x1vl_rvv(size_t vl, const uint8_t *B,
                                                           size_t stride_row)
{
    size_t stride_bytes = stride_row * sizeof(uint8_t);
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vlse8.v v0, (%0), %1" : : "r"(B), "r"(stride_bytes) : "v0", "memory");
}

static inline void tqt_load_b_kernel_gemm_u8_u8_4x1vl_rvv(size_t vl, const uint8_t *B)
{
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vle8.v v0, (%0)" : : "r"(B) : "v0", "memory");
}

/// Unsigned widening multiply-accumulate: u8 x u8 -> u16 -> u32 accumulate
template <size_t MR>
static inline void tqt_vwmaccu_kernel_gemm_u8_u8_4x1vl_rvv(size_t vl,
                                                           const uint8_t *const a_ptrs[MR])
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t" : : "r"(vl) :);
    if constexpr (MR >= 1)
        asm volatile("lbu a0, 0(%0)" : : "r"(a_ptrs[0]) : "a0", "memory");
    asm volatile("vwmulu.vx v8, v0, a0" : : : "v8", "v9");
    if constexpr (MR >= 2)
        asm volatile("lbu a1, 0(%0)" : : "r"(a_ptrs[1]) : "a1", "memory");
    asm volatile("vwmulu.vx v10, v0, a1" : : : "v10", "v11");
    if constexpr (MR >= 3)
        asm volatile("lbu a2, 0(%0)" : : "r"(a_ptrs[2]) : "a2", "memory");
    asm volatile("vwmulu.vx v12, v0, a2" : : : "v12", "v13");
    if constexpr (MR >= 4)
        asm volatile("lbu a3, 0(%0)" : : "r"(a_ptrs[3]) : "a3", "memory");
    asm volatile("vwmulu.vx v14, v0, a3" : : : "v14", "v15");

    asm volatile("vsetvli zero, %0, e16, m2, ta, ma\n\t" : : "r"(vl) :);
    if constexpr (MR >= 1)
        asm volatile("vwaddu.wv v16, v16, v8" : : : "v16", "v17", "v18", "v19");
    if constexpr (MR >= 2)
        asm volatile("vwaddu.wv v20, v20, v10" : : : "v20", "v21", "v22", "v23");
    if constexpr (MR >= 3)
        asm volatile("vwaddu.wv v24, v24, v12" : : : "v24", "v25", "v26", "v27");
    if constexpr (MR >= 4)
        asm volatile("vwaddu.wv v28, v28, v14" : : : "v28", "v29", "v30", "v31");
}

/// Add per-M bias (bias_m[m] broadcast to each row) and per-N bias (bias_n[n] vector)
template <size_t MR>
static inline void tqt_add_bias_mn_kernel_gemm_u8_u8_4x1vl_rvv(size_t vl, const int32_t *bias_m,
                                                               const int32_t *bias_n)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    // Load bias_n vector into v0-v3
    asm volatile("vle32.v v0, (%0)" : : "r"(bias_n) : "v0", "v1", "v2", "v3", "memory");

    // Add bias_n (per-N, same for all rows) + bias_m (per-M, scalar per row)
    if constexpr (MR >= 1) {
        asm volatile("vadd.vv v16, v16, v0" : : : "v16", "v17", "v18", "v19");
        asm volatile("vadd.vx v16, v16, %0" : : "r"(bias_m[0]) : "v16", "v17", "v18", "v19");
    }
    if constexpr (MR >= 2) {
        asm volatile("vadd.vv v20, v20, v0" : : : "v20", "v21", "v22", "v23");
        asm volatile("vadd.vx v20, v20, %0" : : "r"(bias_m[1]) : "v20", "v21", "v22", "v23");
    }
    if constexpr (MR >= 3) {
        asm volatile("vadd.vv v24, v24, v0" : : : "v24", "v25", "v26", "v27");
        asm volatile("vadd.vx v24, v24, %0" : : "r"(bias_m[2]) : "v24", "v25", "v26", "v27");
    }
    if constexpr (MR >= 4) {
        asm volatile("vadd.vv v28, v28, v0" : : : "v28", "v29", "v30", "v31");
        asm volatile("vadd.vx v28, v28, %0" : : "r"(bias_m[3]) : "v28", "v29", "v30", "v31");
    }
}

/// Store int32 accumulator to D
template <size_t MR>
static inline void tqt_store_i32_kernel_gemm_u8_u8_4x1vl_rvv(size_t vl, int32_t *D,
                                                             size_t stride_row)
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

/// Dequantize i32 accumulator to f32: D = clamp(scvtf(acc) * scale + dst_bias, min, max)
template <size_t MR>
static inline void tqt_dequant_f32_kernel_gemm_u8_u8_4x1vl_rvv(size_t vl, float *D,
                                                               size_t stride_row, float scale,
                                                               const float *dst_bias,
                                                               float clamp_min, float clamp_max)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    // Load dst_bias vector (per-N fp32) into v0-v3 if provided
    if (dst_bias != nullptr) {
        asm volatile("vle32.v v0, (%0)" : : "r"(dst_bias) : "v0", "v1", "v2", "v3", "memory");
    }

    // Row 0
    if constexpr (MR >= 1) {
        asm volatile("vfcvt.f.x.v v16, v16" : : : "v16", "v17", "v18", "v19");
        asm volatile("vfmul.vf v16, v16, %0" : : "f"(scale) : "v16", "v17", "v18", "v19");
        if (dst_bias != nullptr)
            asm volatile("vfadd.vv v16, v16, v0" : : : "v16", "v17", "v18", "v19");
        asm volatile("vfmax.vf v16, v16, %0" : : "f"(clamp_min) : "v16", "v17", "v18", "v19");
        asm volatile("vfmin.vf v16, v16, %0" : : "f"(clamp_max) : "v16", "v17", "v18", "v19");
        asm volatile("vse32.v v16, (%0)" : : "r"(D + 0 * stride_row) : "memory");
    }
    // Row 1
    if constexpr (MR >= 2) {
        asm volatile("vfcvt.f.x.v v20, v20" : : : "v20", "v21", "v22", "v23");
        asm volatile("vfmul.vf v20, v20, %0" : : "f"(scale) : "v20", "v21", "v22", "v23");
        if (dst_bias != nullptr)
            asm volatile("vfadd.vv v20, v20, v0" : : : "v20", "v21", "v22", "v23");
        asm volatile("vfmax.vf v20, v20, %0" : : "f"(clamp_min) : "v20", "v21", "v22", "v23");
        asm volatile("vfmin.vf v20, v20, %0" : : "f"(clamp_max) : "v20", "v21", "v22", "v23");
        asm volatile("vse32.v v20, (%0)" : : "r"(D + 1 * stride_row) : "memory");
    }
    // Row 2
    if constexpr (MR >= 3) {
        asm volatile("vfcvt.f.x.v v24, v24" : : : "v24", "v25", "v26", "v27");
        asm volatile("vfmul.vf v24, v24, %0" : : "f"(scale) : "v24", "v25", "v26", "v27");
        if (dst_bias != nullptr)
            asm volatile("vfadd.vv v24, v24, v0" : : : "v24", "v25", "v26", "v27");
        asm volatile("vfmax.vf v24, v24, %0" : : "f"(clamp_min) : "v24", "v25", "v26", "v27");
        asm volatile("vfmin.vf v24, v24, %0" : : "f"(clamp_max) : "v24", "v25", "v26", "v27");
        asm volatile("vse32.v v24, (%0)" : : "r"(D + 2 * stride_row) : "memory");
    }
    // Row 3
    if constexpr (MR >= 4) {
        asm volatile("vfcvt.f.x.v v28, v28" : : : "v28", "v29", "v30", "v31");
        asm volatile("vfmul.vf v28, v28, %0" : : "f"(scale) : "v28", "v29", "v30", "v31");
        if (dst_bias != nullptr)
            asm volatile("vfadd.vv v28, v28, v0" : : : "v28", "v29", "v30", "v31");
        asm volatile("vfmax.vf v28, v28, %0" : : "f"(clamp_min) : "v28", "v29", "v30", "v31");
        asm volatile("vfmin.vf v28, v28, %0" : : "f"(clamp_max) : "v28", "v29", "v30", "v31");
        asm volatile("vse32.v v28, (%0)" : : "r"(D + 3 * stride_row) : "memory");
    }
}
