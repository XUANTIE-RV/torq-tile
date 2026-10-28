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

static inline void tqt_qsi8d256_qau4c32_init_f32(size_t vl)
{
    asm volatile(
        "vsetvli zero, %0, e32, m4, ta, ma\n\t"
        "vmv.v.i v16, 0"
        :
        : "r"(vl)
        : "v16", "v17", "v18", "v19");
}

static inline void tqt_qsi8d256_qau4c32_init_i32(size_t vl)
{
    asm volatile(
        "vsetvli zero, %0, e32, m4, ta, ma\n\t"
        "vmv.v.i v20, 0"
        :
        : "r"(vl)
        : "v20", "v21", "v22", "v23");
}

// Two i8*u4 products fit in i16: [-3840, 3810].
static inline void tqt_qsi8d256_qau4c32_dot_u4_pair(size_t vl, const uint8_t *b, int8_t a_lo,
                                                    int8_t a_hi)
{
    asm volatile(
        "vsetvli zero, %0, e8, m1, ta, ma\n\t"
        "vle8.v v2, (%1)\n\t"
        "vand.vi v0, v2, 0xf\n\t"
        "vsrl.vi v1, v2, 4\n\t"
        "vwmul.vx v8, v0, %2\n\t"
        "vwmacc.vx v8, %3, v1\n\t"
        "vsetvli zero, %0, e16, m2, ta, ma\n\t"
        "vwadd.wv v20, v20, v8\n\t"
        :
        : "r"(vl), "r"(b), "r"((intptr_t)a_lo), "r"((intptr_t)a_hi)
        : "v0", "v1", "v2", "v8", "v9", "v10", "v11", "v20", "v21", "v22", "v23", "memory");
}

static inline void tqt_qsi8d256_qau4c32_accumulate_group(size_t vl, float scale_a,
                                                         const float *scale_b,
                                                         size_t scale_b_byte_stride)
{
    asm volatile(
        "vsetvli zero, %0, e32, m4, ta, ma\n\t"
        "vfcvt.f.x.v v20, v20\n\t"
        "vlse32.v v4, (%1), %2\n\t"
        "vfmul.vf v20, v20, %3\n\t"
        "vfmacc.vv v16, v20, v4"
        :
        : "r"(vl), "r"(scale_b), "r"(scale_b_byte_stride), "f"(scale_a)
        : "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "memory");
}

static inline void tqt_qsi8d256_qau4c32_apply_real_zero(size_t vl, float scaled_activation_sum,
                                                        const float *real_zero_b,
                                                        size_t real_zero_b_byte_stride)
{
    asm volatile(
        "vsetvli zero, %0, e32, m4, ta, ma\n\t"
        "vlse32.v v4, (%1), %2\n\t"
        "vfnmsac.vf v16, %3, v4"
        :
        : "r"(vl), "r"(real_zero_b), "r"(real_zero_b_byte_stride), "f"(scaled_activation_sum)
        : "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "memory");
}

static inline void tqt_qsi8d256_qau4c32_epilogue_f32(size_t vl, float *d, const float *c,
                                                     const float *bias, float clamp_min,
                                                     float clamp_max, bool apply_clamp)
{
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl) :);
    if (c) {
        asm volatile(
            "vle32.v v4, (%0)\n\t"
            "vfadd.vv v16, v16, v4"
            :
            : "r"(c)
            : "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "memory");
    }
    if (bias) {
        asm volatile(
            "vle32.v v4, (%0)\n\t"
            "vfadd.vv v16, v16, v4"
            :
            : "r"(bias)
            : "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "memory");
    }
    // Skip min/max only for the explicitly unbounded range.
    asm volatile(
        "beqz %[apply_clamp], 90f\n\t"
        "vfmax.vf v16, v16, %0\n\t"
        "vfmin.vf v16, v16, %1\n\t"
        "90:\n\t"
        "vse32.v v16, (%2)"
        :
        : "f"(clamp_min), "f"(clamp_max), "r"(d), [apply_clamp] "r"(apply_clamp)
        : "v16", "v17", "v18", "v19", "memory");
}

static inline void tqt_compute_tile_kernel_gemm_qsi8d256_qau4c32_4x1vl_rvv(
    size_t mr, size_t nr, size_t k, size_t k_idx, float *d, const int8_t *a_packed,
    const uint8_t *b_packed, const float *scale_a, size_t ldsa, const int16_t *bsums_a,
    size_t ldbsuma, const float *scale_b, const float *real_zero_b, size_t ldsb, const float *c,
    const float *bias, size_t ldc, size_t ldd, float clamp_min, float clamp_max, bool apply_clamp)
{
    const size_t scale_b_byte_stride = ldsb * sizeof(float);

    for (size_t row = 0; row < mr; ++row) {
        tqt_qsi8d256_qau4c32_init_f32(nr);

        for (size_t group_k = 0; group_k < k; group_k += 32) {
            const size_t absolute_k = k_idx + group_k;
            const size_t weight_block = absolute_k / 32;
            const size_t activation_block = absolute_k / 256;
            const float sa = scale_a[row * ldsa + activation_block];

            tqt_qsi8d256_qau4c32_init_i32(nr);
            for (size_t kk = group_k; kk < group_k + 32; kk += 2) {
                const uint8_t *b = b_packed + (kk / 2) * nr;
                tqt_qsi8d256_qau4c32_dot_u4_pair(nr, b, a_packed[kk * mr + row],
                                                 a_packed[(kk + 1) * mr + row]);
            }

            tqt_qsi8d256_qau4c32_accumulate_group(nr, sa, scale_b + weight_block,
                                                  scale_b_byte_stride);
            const size_t sum_block = absolute_k / 16;
            const int32_t activation_sum = (int32_t)bsums_a[row * ldbsuma + sum_block] +
                                           (int32_t)bsums_a[row * ldbsuma + sum_block + 1];
            tqt_qsi8d256_qau4c32_apply_real_zero(nr, sa * (float)activation_sum,
                                                 real_zero_b + weight_block, scale_b_byte_stride);
        }

        const float *c_row = c ? c + row * ldc : nullptr;
        tqt_qsi8d256_qau4c32_epilogue_f32(nr, d + row * ldd, c_row, bias, clamp_min, clamp_max,
                                          apply_clamp);
    }
}
