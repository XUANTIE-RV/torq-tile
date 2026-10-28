//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif  // Architectural features check.

#include "tqt_subvv_f16_f16_f16_8vl_rvv.h"

#include "src/common/tqt_common.h"

static inline size_t get_vlmax(size_t vlenb)
{
    return vlenb / sizeof(float16_t);
}

static inline size_t get_n_step(size_t vlenb)
{
    return 8 * get_vlmax(vlenb);
}

size_t tqt_get_n_step_subvv_f16_f16_f16_8vl_rvv(void)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = get_n_step(VLENB);

    return n_step;
}

void tqt_run_subvv_f16_f16_f16_8vl_rvv(size_t n, const void *a_ptr, const void *b_ptr, void *c_ptr)
{
    const float16_t *a = (const float16_t *)a_ptr;
    const float16_t *b = (const float16_t *)b_ptr;
    float16_t *c = (float16_t *)c_ptr;

    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e16m8(n - i);
        vfloat16m8_t va = __riscv_vle16_v_f16m8(a + i, vl);
        vfloat16m8_t vb = __riscv_vle16_v_f16m8(b + i, vl);
        vfloat16m8_t vc = __riscv_vfsub_vv_f16m8(va, vb, vl);
        __riscv_vse16_v_f16m8(c + i, vc, vl);
        i += vl;
    }
}
