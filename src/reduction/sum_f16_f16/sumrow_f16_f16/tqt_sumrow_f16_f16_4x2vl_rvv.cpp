//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif  // Architectural features check.

#include "tqt_sumrow_f16_f16_4x2vl_rvv.h"

#include "src/common/tqt_common.h"

static inline size_t get_vlmax(size_t vlenb)
{
    return vlenb / sizeof(float);
}

static inline size_t get_n_step(size_t vlenb)
{
    return 4 * get_vlmax(vlenb);
}

static const size_t m_step = 4;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_sumrow_f16_f16_4x2vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_sumrow_f16_f16_4x2vl_rvv(void)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = get_n_step(VLENB);

    return n_step;
}

// ============================================================================
// Internal: extract scalar from vfloat32m1_t
// ============================================================================

static inline float vf32m1_extract(vfloat32m1_t v)
{
    return __riscv_vfmv_f_s_f32m1_f32(v);
}

// ============================================================================
// Internal: MR-templated row-reduction kernel
// FP16 inputs are widened to FP32 during accumulation for precision; the reduced
// sum is narrowed back to FP16 on store. Each row owns an independent accumulator
// so the up-to-4 reduction chains issue without cross-row data dependencies.
// ============================================================================

template <size_t MR>
static void sumrow_kernel_4vl(size_t n, const float16_t *a, size_t lda, float16_t *c,
                              const float16_t *init, size_t VLENB)
{
    const size_t n_step = get_n_step(VLENB);

    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    const size_t full_vl = __riscv_vsetvl_e32m4(n_step);

    float scalar_acc0 = (init != nullptr) ? (float)init[0] : 0.0f;
    vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, full_vl);

    float scalar_acc1 = 0.0f;
    vfloat32m4_t acc1;
    if constexpr (MR >= 2) {
        scalar_acc1 = (init != nullptr) ? (float)init[1] : 0.0f;
        acc1 = __riscv_vfmv_v_f_f32m4(0.0f, full_vl);
    }

    float scalar_acc2 = 0.0f;
    vfloat32m4_t acc2;
    if constexpr (MR >= 3) {
        scalar_acc2 = (init != nullptr) ? (float)init[2] : 0.0f;
        acc2 = __riscv_vfmv_v_f_f32m4(0.0f, full_vl);
    }

    float scalar_acc3 = 0.0f;
    vfloat32m4_t acc3;
    if constexpr (MR >= 4) {
        scalar_acc3 = (init != nullptr) ? (float)init[3] : 0.0f;
        acc3 = __riscv_vfmv_v_f_f32m4(0.0f, full_vl);
    }

    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m4(n - i);

        vfloat16m2_t v0 = __riscv_vle16_v_f16m2(a + i, vl);
        acc0 = __riscv_vfwadd_wv_f32m4_tu(acc0, acc0, v0, vl);

        if constexpr (MR >= 2) {
            vfloat16m2_t v1 = __riscv_vle16_v_f16m2(a + lda + i, vl);
            acc1 = __riscv_vfwadd_wv_f32m4_tu(acc1, acc1, v1, vl);
        }
        if constexpr (MR >= 3) {
            vfloat16m2_t v2 = __riscv_vle16_v_f16m2(a + 2 * lda + i, vl);
            acc2 = __riscv_vfwadd_wv_f32m4_tu(acc2, acc2, v2, vl);
        }
        if constexpr (MR >= 4) {
            vfloat16m2_t v3 = __riscv_vle16_v_f16m2(a + 3 * lda + i, vl);
            acc3 = __riscv_vfwadd_wv_f32m4_tu(acc3, acc3, v3, vl);
        }

        i += vl;
    }

    vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    vfloat32m1_t r0 = __riscv_vfredusum_vs_f32m4_f32m1(acc0, zero, full_vl);
    c[0] = (float16_t)(vf32m1_extract(r0) + scalar_acc0);

    if constexpr (MR >= 2) {
        vfloat32m1_t r1 = __riscv_vfredusum_vs_f32m4_f32m1(acc1, zero, full_vl);
        c[1] = (float16_t)(vf32m1_extract(r1) + scalar_acc1);
    }
    if constexpr (MR >= 3) {
        vfloat32m1_t r2 = __riscv_vfredusum_vs_f32m4_f32m1(acc2, zero, full_vl);
        c[2] = (float16_t)(vf32m1_extract(r2) + scalar_acc2);
    }
    if constexpr (MR >= 4) {
        vfloat32m1_t r3 = __riscv_vfredusum_vs_f32m4_f32m1(acc3, zero, full_vl);
        c[3] = (float16_t)(vf32m1_extract(r3) + scalar_acc3);
    }
}

// Function pointer type for kernel dispatch
using sumrow_kernel_fn_t = void (*)(size_t n, const float16_t *a, size_t lda, float16_t *c,
                                    const float16_t *init, size_t VLENB);

static constexpr sumrow_kernel_fn_t kernel_table[4] = {
    sumrow_kernel_4vl<1>,
    sumrow_kernel_4vl<2>,
    sumrow_kernel_4vl<3>,
    sumrow_kernel_4vl<4>,
};

// ============================================================================
// Main Function
// ============================================================================

void tqt_run_sumrow_f16_f16_4x2vl_rvv(size_t m, size_t n, const void *a_ptr, size_t lda,
                                      void *c_ptr, const void *init_ptr)
{
    const size_t VLENB = tqt_get_vlenb();

    const float16_t *a = (const float16_t *)a_ptr;
    float16_t *c = (float16_t *)c_ptr;
    const float16_t *init = (const float16_t *)init_ptr;

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        const size_t actual_m = TQT_MIN(m - m_idx, m_step);
        const size_t mr_idx = actual_m - 1;

        kernel_table[mr_idx](n, a + m_idx * lda, lda, c + m_idx, init ? init + m_idx : nullptr,
                             VLENB);
    }
}
