//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif  // Architectural features check.

#include "tqt_geglu_f16_f16_f16_1x2vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/common/tqt_math_rvv.h"

static inline size_t get_vlmax(size_t vlenb)
{
    return vlenb / sizeof(float);
}

static inline size_t get_n_step(size_t vlenb)
{
    return 4 * get_vlmax(vlenb);
}

static const size_t m_step = 1;

// tanh-based GELU approximation:
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * x * (1 + 0.044715 * x * x))).
// gelu_cubic_scale is sqrt(2 / pi) * 0.044715.
static constexpr float gelu_sqrt_2_over_pi = 0.7978845608028654f;
static constexpr float gelu_cubic_scale = 0.035677408136300125f;

size_t tqt_get_m_step_geglu_f16_f16_f16_1x2vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_geglu_f16_f16_f16_1x2vl_rvv(void)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = get_n_step(VLENB);

    return n_step;
}

static inline void geglu_kernel_2vl(size_t n, const float16_t *a, const float16_t *b, float16_t *c)
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
        vfloat32m4_t vb = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(b + i, vl), vl);
        y = __riscv_vfmul_vv_f32m4(y, vb, vl);
        __riscv_vse16_v_f16m2(c + i, __riscv_vfncvt_f_f_w_f16m2(y, vl), vl);
        i += vl;
    }
}

void tqt_run_geglu_f16_f16_f16_1x2vl_rvv(size_t m, size_t n, const void *a_ptr, size_t lda,
                                         const void *b_ptr, size_t ldb, void *c_ptr, size_t ldc)
{
    const float16_t *a = (const float16_t *)a_ptr;
    const float16_t *b = (const float16_t *)b_ptr;
    float16_t *c = (float16_t *)c_ptr;

    if (lda == n && ldb == n && ldc == n) {
        geglu_kernel_2vl(m * n, a, b, c);
        return;
    }

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        geglu_kernel_2vl(n, a + m_idx * lda, b + m_idx * ldb, c + m_idx * ldc);
    }
}
