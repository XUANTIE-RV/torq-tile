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

#define TQT_QSI8D32_QSI5C32_DOT_RAW(LOW_SHIFT, SIGN_SHIFT) \
    "lb %[activation], 0(%[activations])\n\t"              \
    "vsrl.vi v1, v0, " #LOW_SHIFT                          \
    "\n\t"                                                 \
    "vand.vi v1, v1, 15\n\t"                               \
    "vsrl.vi v3, v2, " #SIGN_SHIFT                         \
    "\n\t"                                                 \
    "vand.vi v3, v3, 1\n\t"                                \
    "vsll.vi v3, v3, 4\n\t"                                \
    "vsub.vv v1, v1, v3\n\t"                               \
    "vwmacc.vx v8, %[activation], v1\n\t"

// Eight i8*i5 products fit in i16: [-16256, 16384].
// Widen each octet before combining the original 32-element scale block.
template <size_t Octet>
static inline void tqt_qsi8d32_qsi5c32_dot_octet_rvv(size_t vl, const uint8_t *group, size_t nr,
                                                     const int8_t *activations,
                                                     size_t activation_stride)
{
    static_assert(Octet < 4, "Octet must be in range [0, 3]");
    const uint8_t *low = group + Octet * 4 * nr;
    const uint8_t *sign = group + (16 + Octet) * nr;
    intptr_t activation;
    asm volatile(
        "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
        "vmv.v.i v8, 0\n\t"
        "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
        "vle8.v v2, (%[sign])\n\t"
        "vle8.v v0, (%[low])\n\t"
        TQT_QSI8D32_QSI5C32_DOT_RAW(0, 0)
        "add %[activations], %[activations], %[activation_stride]\n\t"
        TQT_QSI8D32_QSI5C32_DOT_RAW(4, 1)
        "add %[activations], %[activations], %[activation_stride]\n\t"
        "add %[low], %[low], %[nr]\n\t"
        "vle8.v v0, (%[low])\n\t"
        TQT_QSI8D32_QSI5C32_DOT_RAW(0, 2)
        "add %[activations], %[activations], %[activation_stride]\n\t"
        TQT_QSI8D32_QSI5C32_DOT_RAW(4, 3)
        "add %[activations], %[activations], %[activation_stride]\n\t"
        "add %[low], %[low], %[nr]\n\t"
        "vle8.v v0, (%[low])\n\t"
        TQT_QSI8D32_QSI5C32_DOT_RAW(0, 4)
        "add %[activations], %[activations], %[activation_stride]\n\t"
        TQT_QSI8D32_QSI5C32_DOT_RAW(4, 5)
        "add %[activations], %[activations], %[activation_stride]\n\t"
        "add %[low], %[low], %[nr]\n\t"
        "vle8.v v0, (%[low])\n\t"
        TQT_QSI8D32_QSI5C32_DOT_RAW(0, 6)
        "add %[activations], %[activations], %[activation_stride]\n\t"
        TQT_QSI8D32_QSI5C32_DOT_RAW(4, 7)
        "vsetvli zero, %[vl], e16, m2, ta, ma\n\t"
        "vwadd.wv v20, v20, v8\n\t"
        : [activation] "=&r"(activation), [activations] "+&r"(activations), [low] "+&r"(low)
        : [vl] "r"(vl), [sign] "r"(sign), [nr] "r"(nr),
          [activation_stride] "r"(activation_stride)
        : "v0", "v1", "v2", "v3", "v8", "v9", "v20", "v21", "v22", "v23", "memory");
}

#undef TQT_QSI8D32_QSI5C32_DOT_RAW

static inline void tqt_compute_tile_kernel_gemm_qsi8d32_qsi5c32_4x1vl_rvv(
    size_t mr, size_t nr, size_t k, size_t k_idx, float *d, const int8_t *a_packed,
    const uint8_t *b_packed, const float *scale_a, size_t ldsa, const float *scale_b, size_t ldsb,
    const float *c, const float *bias, size_t ldc, size_t ldd, float clamp_min, float clamp_max,
    bool apply_clamp)
{
    constexpr size_t group_size = 32;
    constexpr size_t group_bytes = 20;
    const size_t scale_b_byte_stride = ldsb * sizeof(float);

    for (size_t row = 0; row < mr; ++row) {
        tqt_i8_i8_blockquant_init_f32(nr);
        for (size_t group_k = 0; group_k < k; group_k += group_size) {
            const size_t block = (k_idx + group_k) / group_size;
            const uint8_t *b_group = b_packed + (group_k / group_size) * group_bytes * nr;
            tqt_i8_i8_blockquant_init_i32(nr);
#define TQT_QSI8D32_QSI5C32_DOT(OCTET)                        \
    tqt_qsi8d32_qsi5c32_dot_octet_rvv<OCTET>(nr, b_group, nr, \
                                             a_packed + (group_k + OCTET * 8) * mr + row, mr)
            TQT_QSI8D32_QSI5C32_DOT(0);
            TQT_QSI8D32_QSI5C32_DOT(1);
            TQT_QSI8D32_QSI5C32_DOT(2);
            TQT_QSI8D32_QSI5C32_DOT(3);
#undef TQT_QSI8D32_QSI5C32_DOT
            const float sa = scale_a[row * ldsa + block];
            tqt_i8_i8_blockquant_accumulate_group(nr, sa, scale_b + block, scale_b_byte_stride);
        }
        tqt_i8_i8_blockquant_epilogue_f32(nr, d + row * ldd, c ? c + row * ldc : nullptr, bias,
                                          clamp_min, clamp_max, apply_clamp);
    }
}
