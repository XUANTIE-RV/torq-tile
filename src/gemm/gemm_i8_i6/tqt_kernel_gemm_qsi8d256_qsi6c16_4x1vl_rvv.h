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

template <bool HasSecond>
static inline void tqt_qsi6c16_init_f32(size_t vl)
{
    if constexpr (HasSecond) {
        asm volatile(
            "vsetvli zero, %0, e32, m4, ta, ma\n\t"
            "vmv.v.i v16, 0\n\t"
            "vmv.v.i v20, 0"
            :
            : "r"(vl)
            : "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
    } else {
        asm volatile(
            "vsetvli zero, %0, e32, m4, ta, ma\n\t"
            "vmv.v.i v16, 0"
            :
            : "r"(vl)
            : "v16", "v17", "v18", "v19");
    }
}

template <bool HasSecond>
static inline void tqt_qsi6c16_init_i32(size_t vl)
{
    if constexpr (HasSecond) {
        asm volatile(
            "vsetvli zero, %0, e32, m4, ta, ma\n\t"
            "vmv.v.i v24, 0\n\t"
            "vmv.v.i v28, 0"
            :
            : "r"(vl)
            : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    } else {
        asm volatile(
            "vsetvli zero, %0, e32, m4, ta, ma\n\t"
            "vmv.v.i v24, 0"
            :
            : "r"(vl)
            : "v24", "v25", "v26", "v27");
    }
}

static inline void tqt_qsi6c16_load_high(size_t vl, const uint8_t *high)
{
    asm volatile(
        "vsetvli zero, %0, e8, m1, ta, ma\n\t"
        "vle8.v v3, (%1)"
        :
        : "r"(vl), "r"(high)
        : "v3", "memory");
}

template <bool HasSecond>
static inline void tqt_qsi6c16_dot_low_pair(size_t vl, const uint8_t *low, size_t high_shift,
                                            int8_t a0_lo, int8_t a0_hi, int8_t a1_lo, int8_t a1_hi)
{
    if constexpr (HasSecond) {
        asm volatile(
            "vsetvli zero, %0, e8, m1, ta, ma\n\t"
            "vle8.v v2, (%1)\n\t"
            "vand.vi v1, v2, 0xf\n\t"
            "vsrl.vi v2, v2, 4\n\t"
            "vsrl.vx v4, v3, %2\n\t"
            "vand.vi v5, v4, 3\n\t"
            "vsll.vi v5, v5, 4\n\t"
            "vor.vv v1, v1, v5\n\t"
            "vsrl.vi v4, v4, 2\n\t"
            "vand.vi v4, v4, 3\n\t"
            "vsll.vi v4, v4, 4\n\t"
            "vor.vv v2, v2, v4\n\t"
            "vwmul.vx v4, v1, %3\n\t"
            "vwmul.vx v6, v1, %5\n\t"
            "vsetvli zero, %0, e16, m2, ta, ma\n\t"
            "vwadd.wv v24, v24, v4\n\t"
            "vwadd.wv v28, v28, v6\n\t"
            "vsetvli zero, %0, e8, m1, ta, ma\n\t"
            "vwmul.vx v4, v2, %4\n\t"
            "vwmul.vx v6, v2, %6\n\t"
            "vsetvli zero, %0, e16, m2, ta, ma\n\t"
            "vwadd.wv v24, v24, v4\n\t"
            "vwadd.wv v28, v28, v6"
            :
            : "r"(vl), "r"(low), "r"(high_shift), "r"((intptr_t)a0_lo), "r"((intptr_t)a0_hi),
              "r"((intptr_t)a1_lo), "r"((intptr_t)a1_hi)
            : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v24", "v25", "v26", "v27", "v28", "v29",
              "v30", "v31", "memory");
    } else {
        asm volatile(
            "vsetvli zero, %0, e8, m1, ta, ma\n\t"
            "vle8.v v2, (%1)\n\t"
            "vand.vi v1, v2, 0xf\n\t"
            "vsrl.vi v2, v2, 4\n\t"
            "vsrl.vx v4, v3, %2\n\t"
            "vand.vi v5, v4, 3\n\t"
            "vsll.vi v5, v5, 4\n\t"
            "vor.vv v1, v1, v5\n\t"
            "vsrl.vi v4, v4, 2\n\t"
            "vand.vi v4, v4, 3\n\t"
            "vsll.vi v4, v4, 4\n\t"
            "vor.vv v2, v2, v4\n\t"
            "vwmul.vx v4, v1, %3\n\t"
            "vsetvli zero, %0, e16, m2, ta, ma\n\t"
            "vwadd.wv v24, v24, v4\n\t"
            "vsetvli zero, %0, e8, m1, ta, ma\n\t"
            "vwmul.vx v4, v2, %4\n\t"
            "vsetvli zero, %0, e16, m2, ta, ma\n\t"
            "vwadd.wv v24, v24, v4"
            :
            : "r"(vl), "r"(low), "r"(high_shift), "r"((intptr_t)a0_lo), "r"((intptr_t)a0_hi)
            : "v1", "v2", "v3", "v4", "v5", "v24", "v25", "v26", "v27", "memory");
    }
}

template <bool HasSecond>
static inline void tqt_qsi6c16_finish_group(size_t vl, int32_t correction0, int32_t correction1,
                                            float scale_a0, float scale_a1, const float *scale_b,
                                            size_t scale_b_byte_stride)
{
    if constexpr (HasSecond) {
        asm volatile(
            "vsetvli zero, %0, e32, m4, ta, ma\n\t"
            "vsub.vx v24, v24, %1\n\t"
            "vsub.vx v28, v28, %2\n\t"
            "vfcvt.f.x.v v24, v24\n\t"
            "vfcvt.f.x.v v28, v28\n\t"
            "vlse32.v v4, (%3), %4\n\t"
            "vfmul.vf v24, v24, %5\n\t"
            "vfmul.vf v28, v28, %6\n\t"
            "vfmacc.vv v16, v24, v4\n\t"
            "vfmacc.vv v20, v28, v4"
            :
            : "r"(vl), "r"((intptr_t)correction0), "r"((intptr_t)correction1), "r"(scale_b),
              "r"(scale_b_byte_stride), "f"(scale_a0), "f"(scale_a1)
            : "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
              "v25", "v26", "v27", "v28", "v29", "v30", "v31", "memory");
    } else {
        asm volatile(
            "vsetvli zero, %0, e32, m4, ta, ma\n\t"
            "vsub.vx v24, v24, %1\n\t"
            "vfcvt.f.x.v v24, v24\n\t"
            "vlse32.v v4, (%2), %3\n\t"
            "vfmul.vf v24, v24, %4\n\t"
            "vfmacc.vv v16, v24, v4"
            :
            : "r"(vl), "r"((intptr_t)correction0), "r"(scale_b), "r"(scale_b_byte_stride),
              "f"(scale_a0)
            : "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v24", "v25", "v26", "v27",
              "memory");
    }
}

template <bool HasSecond>
static inline void tqt_qsi6c16_epilogue_f32(size_t vl, float *d0, float *d1, const float *c0,
                                            const float *c1, const float *bias, float clamp_min,
                                            float clamp_max)
{
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl) :);
    if (c0) {
        asm volatile(
            "vle32.v v4, (%0)\n\t"
            "vfadd.vv v16, v16, v4"
            :
            : "r"(c0)
            : "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "memory");
        if constexpr (HasSecond) {
            asm volatile(
                "vle32.v v4, (%0)\n\t"
                "vfadd.vv v20, v20, v4"
                :
                : "r"(c1)
                : "v4", "v5", "v6", "v7", "v20", "v21", "v22", "v23", "memory");
        }
    }
    if (bias) {
        if constexpr (HasSecond) {
            asm volatile(
                "vle32.v v4, (%0)\n\t"
                "vfadd.vv v16, v16, v4\n\t"
                "vfadd.vv v20, v20, v4"
                :
                : "r"(bias)
                : "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                  "memory");
        } else {
            asm volatile(
                "vle32.v v4, (%0)\n\t"
                "vfadd.vv v16, v16, v4"
                :
                : "r"(bias)
                : "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "memory");
        }
    }
    if constexpr (HasSecond) {
        asm volatile(
            "vfmax.vf v16, v16, %0\n\t"
            "vfmin.vf v16, v16, %1\n\t"
            "vfmax.vf v20, v20, %0\n\t"
            "vfmin.vf v20, v20, %1\n\t"
            "vse32.v v16, (%2)\n\t"
            "vse32.v v20, (%3)"
            :
            : "f"(clamp_min), "f"(clamp_max), "r"(d0), "r"(d1)
            : "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "memory");
    } else {
        asm volatile(
            "vfmax.vf v16, v16, %0\n\t"
            "vfmin.vf v16, v16, %1\n\t"
            "vse32.v v16, (%2)"
            :
            : "f"(clamp_min), "f"(clamp_max), "r"(d0)
            : "v16", "v17", "v18", "v19", "memory");
    }
}

template <bool HasSecond>
static inline void tqt_qsi6c16_compute_rows(size_t row, size_t mr, size_t nr, size_t k,
                                            size_t k_idx, float *d, const int8_t *a_packed,
                                            const uint8_t *b_packed, const float *scale_a,
                                            size_t ldsa, const int16_t *bsums_a, size_t ldbsuma,
                                            const float *scale_b, size_t ldsb, const float *c,
                                            const float *bias, size_t ldc, size_t ldd,
                                            float clamp_min, float clamp_max)
{
    const size_t scale_b_byte_stride = ldsb * sizeof(float);
    tqt_qsi6c16_init_f32<HasSecond>(nr);

    for (size_t group_k = 0; group_k < k; group_k += 16) {
        const size_t absolute_k = k_idx + group_k;
        const size_t weight_block = absolute_k / 16;
        const size_t activation_block = absolute_k / 256;
        const float scale_a0 = scale_a[row * ldsa + activation_block];
        const float scale_a1 = HasSecond ? scale_a[(row + 1) * ldsa + activation_block] : 0.0f;
        const uint8_t *b_group = b_packed + (group_k / 16) * 12 * nr;

        tqt_qsi6c16_init_i32<HasSecond>(nr);
        for (size_t high_idx = 0; high_idx < 4; ++high_idx) {
            tqt_qsi6c16_load_high(nr, b_group + (8 + high_idx) * nr);
            for (size_t pair = 0; pair < 2; ++pair) {
                const size_t kk = group_k + high_idx * 4 + pair * 2;
                tqt_qsi6c16_dot_low_pair<HasSecond>(
                    nr, b_group + (high_idx * 2 + pair) * nr, pair * 4, a_packed[kk * mr + row],
                    a_packed[(kk + 1) * mr + row], HasSecond ? a_packed[kk * mr + row + 1] : 0,
                    HasSecond ? a_packed[(kk + 1) * mr + row + 1] : 0);
            }
        }

        const size_t sum_block = absolute_k / 16;
        const int32_t correction0 = 32 * (int32_t)bsums_a[row * ldbsuma + sum_block];
        const int32_t correction1 =
            HasSecond ? 32 * (int32_t)bsums_a[(row + 1) * ldbsuma + sum_block] : 0;
        tqt_qsi6c16_finish_group<HasSecond>(nr, correction0, correction1, scale_a0, scale_a1,
                                            scale_b + weight_block, scale_b_byte_stride);
    }

    const float *c0 = c ? c + row * ldc : nullptr;
    const float *c1 = HasSecond && c ? c + (row + 1) * ldc : nullptr;
    tqt_qsi6c16_epilogue_f32<HasSecond>(nr, d + row * ldd,
                                        HasSecond ? d + (row + 1) * ldd : nullptr, c0, c1, bias,
                                        clamp_min, clamp_max);
}

static inline void tqt_compute_tile_kernel_gemm_qsi8d256_qsi6c16_4x1vl_rvv(
    size_t mr, size_t nr, size_t k, size_t k_idx, float *d, const int8_t *a_packed,
    const uint8_t *b_packed, const float *scale_a, size_t ldsa, const int16_t *bsums_a,
    size_t ldbsuma, const float *scale_b, size_t ldsb, const float *c, const float *bias,
    size_t ldc, size_t ldd, float clamp_min, float clamp_max)
{
    size_t row = 0;
    for (; row + 1 < mr; row += 2) {
        tqt_qsi6c16_compute_rows<true>(row, mr, nr, k, k_idx, d, a_packed, b_packed, scale_a, ldsa,
                                       bsums_a, ldbsuma, scale_b, ldsb, c, bias, ldc, ldd,
                                       clamp_min, clamp_max);
    }
    if (row < mr) {
        tqt_qsi6c16_compute_rows<false>(row, mr, nr, k, k_idx, d, a_packed, b_packed, scale_a, ldsa,
                                        bsums_a, ldbsuma, scale_b, ldsb, c, bias, ldc, ldd,
                                        clamp_min, clamp_max);
    }
}
