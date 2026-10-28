//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_swiglu_f32_f32_f32_1x4vl_rvv.h"

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

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_swiglu_f32_f32_f32_1x4vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_swiglu_f32_f32_f32_1x4vl_rvv(void)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = get_n_step(VLENB);

    return n_step;
}

// ============================================================================
// Internal: SwiGLU over one contiguous run of n elements
// The gate is folded as a / (1 + exp(-a)); the single divide is cheaper and more
// accurate than a reciprocal estimate plus a Newton correction. The polynomial
// exp keeps several vector groups live, so the tile stays at LMUL=4 to avoid
// spilling.
//
// The linear operand is loaded only after tqt_vexp_f32m4 has returned. exp holds
// up to five LMUL=4 groups live at its peak, and the caller's gate group makes
// six of the eight groups the ISA offers; hoisting this load above the exp call
// would add a seventh and push the allocator into spills. Keep it here.
// ============================================================================

static inline void swiglu_kernel_4vl(size_t n, const float *a, const float *b, float *c)
{
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m4(n - i);
        vfloat32m4_t va = __riscv_vle32_v_f32m4(a + i, vl);
        vfloat32m4_t ve = tqt_vexp_f32m4(__riscv_vfneg_v_f32m4(va, vl), vl);
        vfloat32m4_t vd = __riscv_vfadd_vf_f32m4(ve, 1.0f, vl);
        vfloat32m4_t vs = __riscv_vfdiv_vv_f32m4(va, vd, vl);
        vfloat32m4_t vb = __riscv_vle32_v_f32m4(b + i, vl);
        __riscv_vse32_v_f32m4(c + i, __riscv_vfmul_vv_f32m4(vs, vb, vl), vl);
        i += vl;
    }
}

// ============================================================================
// Main Function
// ============================================================================

void tqt_run_swiglu_f32_f32_f32_1x4vl_rvv(size_t m, size_t n, const void *a_ptr, size_t lda,
                                          const void *b_ptr, size_t ldb, void *c_ptr, size_t ldc)
{
    const float *a = (const float *)a_ptr;
    const float *b = (const float *)b_ptr;
    float *c = (float *)c_ptr;

    // Fully packed operands: one contiguous run over m * n elements, which drops
    // the row loop and its per-row vsetvl tail.
    if (lda == n && ldb == n && ldc == n) {
        swiglu_kernel_4vl(m * n, a, b, c);
        return;
    }

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        swiglu_kernel_4vl(n, a + m_idx * lda, b + m_idx * ldb, c + m_idx * ldc);
    }
}
