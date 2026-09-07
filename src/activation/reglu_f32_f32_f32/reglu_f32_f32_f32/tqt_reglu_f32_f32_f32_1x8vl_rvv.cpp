//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_reglu_f32_f32_f32_1x8vl_rvv.h"

#include "src/common/tqt_common.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB / sizeof(float);
static const size_t m_step = 1;
static const size_t n_step = 8 * vlmax;

size_t tqt_get_m_step_reglu_f32_f32_f32_1x8vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_reglu_f32_f32_f32_1x8vl_rvv(void)
{
    return n_step;
}

static inline void reglu_kernel_8vl(size_t n, const float *a, const float *b, float *c)
{
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vg = __riscv_vfmax_vf_f32m8(va, 0.0f, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
        vfloat32m8_t vc = __riscv_vfmul_vv_f32m8(vg, vb, vl);
        __riscv_vse32_v_f32m8(c + i, vc, vl);
        i += vl;
    }
}

void tqt_run_reglu_f32_f32_f32_1x8vl_rvv(size_t m, size_t n, const void *a_ptr, size_t lda,
                                         const void *b_ptr, size_t ldb, void *c_ptr, size_t ldc)
{
    const float *a = (const float *)a_ptr;
    const float *b = (const float *)b_ptr;
    float *c = (float *)c_ptr;

    if (lda == n && ldb == n && ldc == n) {
        reglu_kernel_8vl(m * n, a, b, c);
        return;
    }

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        reglu_kernel_8vl(n, a + m_idx * lda, b + m_idx * ldb, c + m_idx * ldc);
    }
}
