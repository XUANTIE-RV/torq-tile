//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_divvv_f32_f32_f32_8vl_rvv.h"

#include "src/common/tqt_common.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB / sizeof(float);
static const size_t n_step = 8 * vlmax;

size_t tqt_get_n_step_divvv_f32_f32_f32_8vl_rvv(void)
{
    return n_step;
}

void tqt_run_divvv_f32_f32_f32_8vl_rvv(size_t n, const void *a_ptr, const void *b_ptr, void *c_ptr)
{
    const float *a = (const float *)a_ptr;
    const float *b = (const float *)b_ptr;
    float *c = (float *)c_ptr;

    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
        vfloat32m8_t vc = __riscv_vfdiv_vv_f32m8(va, vb, vl);
        __riscv_vse32_v_f32m8(c + i, vc, vl);
        i += vl;
    }
}
