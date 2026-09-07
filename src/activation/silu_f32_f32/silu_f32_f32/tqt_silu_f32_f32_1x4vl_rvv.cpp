//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_silu_f32_f32_1x4vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/common/tqt_math_rvv.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB / sizeof(float);
static const size_t m_step = 1;
static const size_t n_step = 4 * vlmax;  // LMUL=4

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_silu_f32_f32_1x4vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_silu_f32_f32_1x4vl_rvv(void)
{
    return n_step;
}

// ============================================================================
// Internal: SiLU over one contiguous run of n elements
// Evaluated as a / (1 + exp(-a)); the single divide is cheaper and more accurate
// than a reciprocal estimate plus a Newton correction. The polynomial exp keeps
// several vector groups live, so the tile stays at LMUL=4 to avoid spilling.
// ============================================================================

static inline void silu_kernel_4vl(size_t n, const float *a, float *c)
{
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m4(n - i);
        vfloat32m4_t va = __riscv_vle32_v_f32m4(a + i, vl);
        vfloat32m4_t ve = tqt_vexp_f32m4(__riscv_vfneg_v_f32m4(va, vl), vl);
        vfloat32m4_t vd = __riscv_vfadd_vf_f32m4(ve, 1.0f, vl);
        __riscv_vse32_v_f32m4(c + i, __riscv_vfdiv_vv_f32m4(va, vd, vl), vl);
        i += vl;
    }
}

// ============================================================================
// Main Function
// ============================================================================

void tqt_run_silu_f32_f32_1x4vl_rvv(size_t m, size_t n, const void *a_ptr, size_t lda, void *c_ptr,
                                    size_t ldc)
{
    const float *a = (const float *)a_ptr;
    float *c = (float *)c_ptr;

    // Fully packed input and output: one contiguous run over m * n elements,
    // which drops the row loop and its per-row vsetvl tail.
    if (lda == n && ldc == n) {
        silu_kernel_4vl(m * n, a, c);
        return;
    }

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        silu_kernel_4vl(n, a + m_idx * lda, c + m_idx * ldc);
    }
}
