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

static inline void tqt_qsi8d256_qau2c16_init_f32_rvv(size_t vl)
{
    asm volatile(
        "vsetvli zero, %0, e32, m4, ta, ma\n\t"
        "vmv.v.i v16, 0"
        :
        : "r"(vl)
        : "v16", "v17", "v18", "v19");
}

static inline void tqt_qsi8d256_qau2c16_init_super_i32_rvv(size_t vl)
{
    asm volatile(
        "vsetvli zero, %0, e32, m4, ta, ma\n\t"
        "vmv.v.i v20, 0\n\t"
        "vmv.v.i v24, 0"
        :
        : "r"(vl)
        : "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
}

static inline void tqt_qsi8d256_qau2c16_init_dot_i32_rvv(size_t vl)
{
    asm volatile(
        "vsetvli zero, %0, e32, m4, ta, ma\n\t"
        "vmv.v.i v28, 0"
        :
        : "r"(vl)
        : "v28", "v29", "v30", "v31");
}

template <size_t Slot>
static inline void tqt_qsi8d256_qau2c16_decode_rvv(size_t vl, const uint8_t *group, size_t nr)
{
    static_assert(Slot < 16, "Slot must be in range [0, 15]");
    const uint8_t *low = group + (Slot / 4) * nr;
    asm volatile(
        "vsetvli zero, %0, e8, m1, ta, ma\n\t"
        "vle8.v v0, (%1)\n\t"
        "vsrl.vi v1, v0, %2\n\t"
        "vand.vi v1, v1, 3"
        :
        : "r"(vl), "r"(low), "i"((Slot % 4) * 2)
        : "v0", "v1", "memory");
}

static inline void tqt_qsi8d256_qau2c16_dot_rvv(size_t vl, int8_t activation)
{
    asm volatile(
        "vsetvli zero, %0, e16, m2, ta, ma\n\t"
        "vsext.vf2 v8, v1\n\t"
        "vwmacc.vx v28, %1, v8"
        :
        : "r"(vl), "r"((intptr_t)activation)
        : "v1", "v8", "v9", "v28", "v29", "v30", "v31");
}

static inline void tqt_qsi8d256_qau2c16_accumulate_scale_rvv(size_t vl, const uint8_t *qscale,
                                                             size_t qscale_byte_stride)
{
    asm volatile(
        "vsetvli zero, %0, e8, m1, ta, ma\n\t"
        "vlse8.v v0, (%1), %2\n\t"
        "vsetvli zero, %0, e16, m2, ta, ma\n\t"
        "vzext.vf2 v8, v0\n\t"
        "vsetvli zero, %0, e32, m4, ta, ma\n\t"
        "vzext.vf2 v4, v8\n\t"
        "vmacc.vv v20, v28, v4"
        :
        : "r"(vl), "r"(qscale), "r"(qscale_byte_stride)
        : "v0", "v4", "v5", "v6", "v7", "v8", "v9", "v20", "v21", "v22", "v23", "v28", "v29", "v30",
          "v31", "memory");
}

static inline void tqt_qsi8d256_qau2c16_accumulate_min_rvv(size_t vl, int32_t activation_sum,
                                                           const uint8_t *qmin,
                                                           size_t qmin_byte_stride)
{
    asm volatile(
        "vsetvli zero, %0, e8, m1, ta, ma\n\t"
        "vlse8.v v0, (%1), %2\n\t"
        "vsetvli zero, %0, e16, m2, ta, ma\n\t"
        "vzext.vf2 v8, v0\n\t"
        "vsetvli zero, %0, e32, m4, ta, ma\n\t"
        "vzext.vf2 v4, v8\n\t"
        "vmacc.vx v24, %3, v4"
        :
        : "r"(vl), "r"(qmin), "r"(qmin_byte_stride), "r"((intptr_t)activation_sum)
        : "v0", "v4", "v5", "v6", "v7", "v8", "v9", "v24", "v25", "v26", "v27", "memory");
}

static inline void tqt_qsi8d256_qau2c16_finish_super_rvv(size_t vl, float scale_a,
                                                         const float *super_scale_b,
                                                         const float *super_min_b,
                                                         size_t super_scale_byte_stride)
{
    asm volatile(
        "vsetvli zero, %0, e32, m4, ta, ma\n\t"
        "vfcvt.f.x.v v20, v20\n\t"
        "vlse32.v v4, (%1), %2\n\t"
        "vfmul.vf v20, v20, %3\n\t"
        "vfmacc.vv v16, v20, v4\n\t"
        "vfcvt.f.x.v v24, v24\n\t"
        "vlse32.v v4, (%4), %2\n\t"
        "vfmul.vf v24, v24, %3\n\t"
        "vfnmsac.vv v16, v24, v4"
        :
        : "r"(vl), "r"(super_scale_b), "r"(super_scale_byte_stride), "f"(scale_a), "r"(super_min_b)
        : "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
          "v25", "v26", "v27", "memory");
}

static inline void tqt_qsi8d256_qau2c16_epilogue_rvv(size_t vl, float *d, const float *c,
                                                     const float *bias, float clamp_min,
                                                     float clamp_max)
{
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vl) :);
    if (c != nullptr) {
        asm volatile(
            "vle32.v v4, (%0)\n\t"
            "vfadd.vv v16, v16, v4"
            :
            : "r"(c)
            : "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "memory");
    }
    if (bias != nullptr) {
        asm volatile(
            "vle32.v v4, (%0)\n\t"
            "vfadd.vv v16, v16, v4"
            :
            : "r"(bias)
            : "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "memory");
    }
    asm volatile(
        "vfmax.vf v16, v16, %0\n\t"
        "vfmin.vf v16, v16, %1\n\t"
        "vse32.v v16, (%2)"
        :
        : "f"(clamp_min), "f"(clamp_max), "r"(d)
        : "v16", "v17", "v18", "v19", "memory");
}

static inline void tqt_compute_tile_kernel_gemm_qsi8d256_qau2c16_4x1vl_rvv(
    size_t mr, size_t nr, size_t k, size_t k_idx, float *d, const int8_t *a_packed,
    const uint8_t *b_packed, const float *scale_a, size_t ldsa, const int16_t *bsums_a,
    size_t ldbsuma, const uint8_t *qscale_b, const uint8_t *qmin_b, size_t ldqb,
    const float *super_scale_b, const float *super_min_b, size_t ldssb, const float *c,
    const float *bias, size_t ldc, size_t ldd, float clamp_min, float clamp_max)
{
    constexpr size_t group_bytes = 4;
    const size_t qparam_byte_stride = ldqb * sizeof(uint8_t);
    const size_t super_scale_byte_stride = ldssb * sizeof(float);

    for (size_t row = 0; row < mr; ++row) {
        tqt_qsi8d256_qau2c16_init_f32_rvv(nr);
        for (size_t super_k = 0; super_k < k; super_k += 256) {
            const size_t super_block = (k_idx + super_k) / 256;
            tqt_qsi8d256_qau2c16_init_super_i32_rvv(nr);
            for (size_t group = 0; group < 16; ++group) {
                const size_t group_k = super_k + group * 16;
                const size_t weight_block = (k_idx + group_k) / 16;
                const uint8_t *b_group = b_packed + (group_k / 16) * group_bytes * nr;
                tqt_qsi8d256_qau2c16_init_dot_i32_rvv(nr);
#define TQT_QSI8D256_QAU2C16_DOT_SLOT(SLOT)                 \
    tqt_qsi8d256_qau2c16_decode_rvv<SLOT>(nr, b_group, nr); \
    tqt_qsi8d256_qau2c16_dot_rvv(nr, a_packed[(group_k + SLOT) * mr + row])
                TQT_QSI8D256_QAU2C16_DOT_SLOT(0);
                TQT_QSI8D256_QAU2C16_DOT_SLOT(1);
                TQT_QSI8D256_QAU2C16_DOT_SLOT(2);
                TQT_QSI8D256_QAU2C16_DOT_SLOT(3);
                TQT_QSI8D256_QAU2C16_DOT_SLOT(4);
                TQT_QSI8D256_QAU2C16_DOT_SLOT(5);
                TQT_QSI8D256_QAU2C16_DOT_SLOT(6);
                TQT_QSI8D256_QAU2C16_DOT_SLOT(7);
                TQT_QSI8D256_QAU2C16_DOT_SLOT(8);
                TQT_QSI8D256_QAU2C16_DOT_SLOT(9);
                TQT_QSI8D256_QAU2C16_DOT_SLOT(10);
                TQT_QSI8D256_QAU2C16_DOT_SLOT(11);
                TQT_QSI8D256_QAU2C16_DOT_SLOT(12);
                TQT_QSI8D256_QAU2C16_DOT_SLOT(13);
                TQT_QSI8D256_QAU2C16_DOT_SLOT(14);
                TQT_QSI8D256_QAU2C16_DOT_SLOT(15);
#undef TQT_QSI8D256_QAU2C16_DOT_SLOT
                tqt_qsi8d256_qau2c16_accumulate_scale_rvv(nr, qscale_b + weight_block,
                                                          qparam_byte_stride);
                const int32_t activation_sum = bsums_a[row * ldbsuma + weight_block];
                tqt_qsi8d256_qau2c16_accumulate_min_rvv(nr, activation_sum, qmin_b + weight_block,
                                                        qparam_byte_stride);
            }
            const float sa = scale_a[row * ldsa + super_block];
            tqt_qsi8d256_qau2c16_finish_super_rvv(nr, sa, super_scale_b + super_block,
                                                  super_min_b + super_block,
                                                  super_scale_byte_stride);
        }
        tqt_qsi8d256_qau2c16_epilogue_rvv(nr, d + row * ldd, c ? c + row * ldc : nullptr, bias,
                                          clamp_min, clamp_max);
    }
}
