//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif  // Architectural features check.

#include "tqt_relu_f16_f16_1x8vl_rvv.h"

#include "src/common/tqt_common.h"

static inline size_t get_vlmax(size_t vlenb)
{
    return vlenb / sizeof(float16_t);
}

static inline size_t get_n_step(size_t vlenb)
{
    return 8 * get_vlmax(vlenb);
}

static const size_t m_step = 1;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_relu_f16_f16_1x8vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_relu_f16_f16_1x8vl_rvv(void)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = get_n_step(VLENB);

    return n_step;
}

// ============================================================================
// Internal: ReLU over one contiguous run of n elements
// Only the loaded tile is live, so LMUL=8 never spills.
// ============================================================================

static inline void relu_kernel_8vl(size_t n, const float16_t *a, float16_t *c)
{
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e16m8(n - i);
        vfloat16m8_t va = __riscv_vle16_v_f16m8(a + i, vl);
        vfloat16m8_t vc = __riscv_vfmax_vf_f16m8(va, (float16_t)0.0f, vl);
        __riscv_vse16_v_f16m8(c + i, vc, vl);
        i += vl;
    }
}

// ============================================================================
// Main Function
// ============================================================================

void tqt_run_relu_f16_f16_1x8vl_rvv(size_t m, size_t n, const void *a_ptr, size_t lda, void *c_ptr,
                                    size_t ldc)
{
    const float16_t *a = (const float16_t *)a_ptr;
    float16_t *c = (float16_t *)c_ptr;

    // Fully packed input and output: one contiguous run over m * n elements,
    // which drops the row loop and its per-row vsetvl tail.
    if (lda == n && ldc == n) {
        relu_kernel_8vl(m * n, a, c);
        return;
    }

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        relu_kernel_8vl(n, a + m_idx * lda, c + m_idx * ldc);
    }
}
