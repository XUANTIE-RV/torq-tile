//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif

#include <stddef.h>
#include <stdint.h>

#include "src/gemm/gemm_i8_i8/tqt_kernel_gemm_i8_i8_4x1vl_blockquant_rvv.h"

#define TQT_QSI8D32_QSI4C32_DOT_RAW(SHIFT)     \
    "lb %[activation], 0(%[activations])\n\t"  \
    "vsrl.vi v1, v0, " #SHIFT                  \
    "\n\t"                                     \
    "vand.vi v1, v1, 15\n\t"                   \
    "vsll.vi v1, v1, 4\n\t"                    \
    "vsra.vi v1, v1, 4\n\t"                    \
    "vsetvli zero, %[vl], e16, m2, ta, ma\n\t" \
    "vsext.vf2 v8, v1\n\t"                     \
    "vwmacc.vx v20, %[activation], v8\n\t"

template <size_t Pair>
static inline void tqt_qsi8d32_qsi4c32_dot_pair_rvv(size_t vl, const uint8_t *group, size_t nr,
                                                    const int8_t *activations,
                                                    size_t activation_stride)
{
    static_assert(Pair < 16, "Pair must be in range [0, 15]");
    const uint8_t *pair = group + Pair * nr;
    intptr_t activation;
    asm volatile(
        "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
        "vle8.v v0, (%[pair])\n\t"
        TQT_QSI8D32_QSI4C32_DOT_RAW(0)
        "add %[activations], %[activations], %[activation_stride]\n\t"
        "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
        TQT_QSI8D32_QSI4C32_DOT_RAW(4)
        : [activation] "=&r"(activation), [activations] "+&r"(activations)
        : [vl] "r"(vl), [pair] "r"(pair), [activation_stride] "r"(activation_stride)
        : "v0", "v1", "v8", "v9", "v20", "v21", "v22", "v23", "memory");
}

#undef TQT_QSI8D32_QSI4C32_DOT_RAW

static inline void tqt_compute_tile_kernel_gemm_qsi8d32_qsi4c32_4x1vl_rvv(
    size_t mr, size_t nr, size_t k, size_t k_idx, float *d, const int8_t *a_packed,
    const uint8_t *b_packed, const float *scale_a, size_t ldsa, const float *scale_b, size_t ldsb,
    const float *c, const float *bias, size_t ldc, size_t ldd, float clamp_min, float clamp_max)
{
    constexpr size_t group_size = 32;
    constexpr size_t group_bytes = group_size / 2;
    const size_t scale_b_byte_stride = ldsb * sizeof(float);

    for (size_t row = 0; row < mr; ++row) {
        tqt_i8_i8_blockquant_init_f32(nr);
        for (size_t group_k = 0; group_k < k; group_k += group_size) {
            const size_t block = (k_idx + group_k) / group_size;
            const uint8_t *b_group = b_packed + (group_k / group_size) * group_bytes * nr;
            tqt_i8_i8_blockquant_init_i32(nr);
#define TQT_QSI8D32_QSI4C32_DOT(PAIR)                       \
    tqt_qsi8d32_qsi4c32_dot_pair_rvv<PAIR>(nr, b_group, nr, \
                                           a_packed + (group_k + PAIR * 2) * mr + row, mr)
            TQT_QSI8D32_QSI4C32_DOT(0);
            TQT_QSI8D32_QSI4C32_DOT(1);
            TQT_QSI8D32_QSI4C32_DOT(2);
            TQT_QSI8D32_QSI4C32_DOT(3);
            TQT_QSI8D32_QSI4C32_DOT(4);
            TQT_QSI8D32_QSI4C32_DOT(5);
            TQT_QSI8D32_QSI4C32_DOT(6);
            TQT_QSI8D32_QSI4C32_DOT(7);
            TQT_QSI8D32_QSI4C32_DOT(8);
            TQT_QSI8D32_QSI4C32_DOT(9);
            TQT_QSI8D32_QSI4C32_DOT(10);
            TQT_QSI8D32_QSI4C32_DOT(11);
            TQT_QSI8D32_QSI4C32_DOT(12);
            TQT_QSI8D32_QSI4C32_DOT(13);
            TQT_QSI8D32_QSI4C32_DOT(14);
            TQT_QSI8D32_QSI4C32_DOT(15);
#undef TQT_QSI8D32_QSI4C32_DOT
            const float sa = scale_a[row * ldsa + block];
            tqt_i8_i8_blockquant_accumulate_group(nr, sa, scale_b + block, scale_b_byte_stride);
        }
        tqt_i8_i8_blockquant_epilogue_f32(nr, d + row * ldd, c ? c + row * ldc : nullptr, bias,
                                          clamp_min, clamp_max);
    }
}
