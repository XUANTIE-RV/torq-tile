//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif  // Architectural features check.

#include "tqt_gelu_f16_f16_1x2vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/common/tqt_math_rvv.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB / sizeof(float);
static const size_t m_step = 1;
static const size_t n_step = 4 * vlmax;

static constexpr float gelu_sqrt_2_over_pi = 0.7978845608028654f;
static constexpr float gelu_cubic_scale = 0.035677408136300125f;

size_t tqt_get_m_step_gelu_f16_f16_1x2vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gelu_f16_f16_1x2vl_rvv(void)
{
    return n_step;
}

static inline void gelu_kernel_2vl(size_t n, const float16_t *a, float16_t *c)
{
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e16m2(n - i);
        vfloat32m4_t x = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(a + i, vl), vl);
        vfloat32m4_t x2 = __riscv_vfmul_vv_f32m4(x, x, vl);
        vfloat32m4_t u = __riscv_vfmv_v_f_f32m4(gelu_sqrt_2_over_pi, vl);
        u = __riscv_vfmacc_vf_f32m4(u, gelu_cubic_scale, x2, vl);
        u = __riscv_vfmul_vv_f32m4(u, x, vl);

        vfloat32m4_t tanh_u = tqt_vtanh_f32m4(u, vl);
        vfloat32m4_t half_x = __riscv_vfmul_vf_f32m4(x, 0.5f, vl);
        vfloat32m4_t y = __riscv_vfmacc_vv_f32m4(half_x, half_x, tanh_u, vl);
        __riscv_vse16_v_f16m2(c + i, __riscv_vfncvt_f_f_w_f16m2(y, vl), vl);
        i += vl;
    }
}

void tqt_run_gelu_f16_f16_1x2vl_rvv(size_t m, size_t n, const void *a_ptr, size_t lda, void *c_ptr,
                                    size_t ldc)
{
    const float16_t *a = (const float16_t *)a_ptr;
    float16_t *c = (float16_t *)c_ptr;

    if (lda == n && ldc == n) {
        gelu_kernel_2vl(m * n, a, c);
        return;
    }

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        gelu_kernel_2vl(n, a + m_idx * lda, c + m_idx * ldc);
    }
}
