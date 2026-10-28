//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_divvs_f32_f32_f32_8vl_rvv.h"

#include "src/common/tqt_common.h"

static inline size_t get_vlmax(size_t vlenb)
{
    return vlenb / sizeof(float);
}

static inline size_t get_n_step(size_t vlenb)
{
    return 8 * get_vlmax(vlenb);
}

size_t tqt_get_n_step_divvs_f32_f32_f32_8vl_rvv(void)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = get_n_step(VLENB);

    return n_step;
}

void tqt_run_divvs_f32_f32_f32_8vl_rvv(size_t n, const void *a_ptr, float b, void *c_ptr)
{
    const float *a = (const float *)a_ptr;
    float *c = (float *)c_ptr;

    // The divisor is a scalar, so compute its reciprocal once and turn the
    // per-element divide into a (much cheaper) multiply: c[i] = a[i] * (1 / b).
    const float inv_b = 1.0f / b;

    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vc = __riscv_vfmul_vf_f32m8(va, inv_b, vl);
        __riscv_vse32_v_f32m8(c + i, vc, vl);
        i += vl;
    }
}
