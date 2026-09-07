//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_sumcol_f32_f32_2x8vl_rvv.h"

#include "src/common/tqt_common.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB / sizeof(float);
static const size_t m_step = 2;              // M-direction 2-way unroll
static const size_t n_step = 2 * 4 * vlmax;  // 8vl: 2 groups of LMUL=4

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_sumcol_f32_f32_2x8vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_sumcol_f32_f32_2x8vl_rvv(void)
{
    return n_step;
}

// ============================================================================
// Internal: NR-templated column-reduction kernel
// Keeps 8vl columns (2 LMUL=4 groups) while unrolling the M loop by 2 with a pair
// of partial accumulators per group, breaking the M-direction dependency chain so
// consecutive adds issue independently. The partials are combined before store.
// ============================================================================

template <size_t NR>
static void sumcol_kernel_4vl(size_t m, size_t n_block, const float *a, size_t lda, float *c,
                              const float *init)
{
    static_assert(NR >= 1 && NR <= 2, "NR must be in range [1, 2]");

    const size_t group_width = 4 * vlmax;

    // Group 0: two partial accumulators; init seeds the "a" partial.
    const size_t remain0 = (n_block > 0) ? n_block : 0;
    const size_t vl0 = __riscv_vsetvl_e32m4(remain0);
    vfloat32m4_t acc0a;
    if (init != nullptr && remain0 > 0) {
        acc0a = __riscv_vle32_v_f32m4(init, vl0);
    } else {
        acc0a = __riscv_vfmv_v_f_f32m4(0.0f, vl0);
    }
    vfloat32m4_t acc0b = __riscv_vfmv_v_f_f32m4(0.0f, vl0);

    // Group 1
    size_t vl1 = 0;
    vfloat32m4_t acc1a, acc1b;
    if constexpr (NR == 2) {
        const size_t remain1 = (n_block > group_width) ? (n_block - group_width) : 0;
        vl1 = __riscv_vsetvl_e32m4(remain1);
        if (init != nullptr && remain1 > 0) {
            acc1a = __riscv_vle32_v_f32m4(init + group_width, vl1);
        } else {
            acc1a = __riscv_vfmv_v_f_f32m4(0.0f, vl1);
        }
        acc1b = __riscv_vfmv_v_f_f32m4(0.0f, vl1);
    }

    // Main loop: 2 rows per iteration into separate partials.
    size_t row = 0;
    for (; row + 2 <= m; row += 2) {
        const float *row_ptr0 = a + row * lda;
        const float *row_ptr1 = a + (row + 1) * lda;

        vfloat32m4_t v0a = __riscv_vle32_v_f32m4(row_ptr0, vl0);
        acc0a = __riscv_vfadd_vv_f32m4(acc0a, v0a, vl0);
        vfloat32m4_t v0b = __riscv_vle32_v_f32m4(row_ptr1, vl0);
        acc0b = __riscv_vfadd_vv_f32m4(acc0b, v0b, vl0);

        if constexpr (NR == 2) {
            if (vl1 > 0) {
                vfloat32m4_t v1a = __riscv_vle32_v_f32m4(row_ptr0 + group_width, vl1);
                acc1a = __riscv_vfadd_vv_f32m4(acc1a, v1a, vl1);
                vfloat32m4_t v1b = __riscv_vle32_v_f32m4(row_ptr1 + group_width, vl1);
                acc1b = __riscv_vfadd_vv_f32m4(acc1b, v1b, vl1);
            }
        }
    }

    // Tail: last odd row folds into the "a" partial.
    if (row < m) {
        const float *row_ptr0 = a + row * lda;
        vfloat32m4_t v0a = __riscv_vle32_v_f32m4(row_ptr0, vl0);
        acc0a = __riscv_vfadd_vv_f32m4(acc0a, v0a, vl0);
        if constexpr (NR == 2) {
            if (vl1 > 0) {
                vfloat32m4_t v1a = __riscv_vle32_v_f32m4(row_ptr0 + group_width, vl1);
                acc1a = __riscv_vfadd_vv_f32m4(acc1a, v1a, vl1);
            }
        }
    }

    // Combine partials and store.
    vfloat32m4_t acc0 = __riscv_vfadd_vv_f32m4(acc0a, acc0b, vl0);
    __riscv_vse32_v_f32m4(c, acc0, vl0);
    if constexpr (NR == 2) {
        if (vl1 > 0) {
            vfloat32m4_t acc1 = __riscv_vfadd_vv_f32m4(acc1a, acc1b, vl1);
            __riscv_vse32_v_f32m4(c + group_width, acc1, vl1);
        }
    }
}

// Function pointer type for kernel dispatch
using sumcol_kernel_fn_t = void (*)(size_t m, size_t n_block, const float *a, size_t lda, float *c,
                                    const float *init);

static constexpr sumcol_kernel_fn_t kernel_table[2] = {
    sumcol_kernel_4vl<1>,
    sumcol_kernel_4vl<2>,
};

// ============================================================================
// Main Function
// ============================================================================

void tqt_run_sumcol_f32_f32_2x8vl_rvv(size_t m, size_t n, const void *a_ptr, size_t lda,
                                      void *c_ptr, const void *init_ptr)
{
    const float *a = (const float *)a_ptr;
    float *c = (float *)c_ptr;
    const float *init = (const float *)init_ptr;

    const size_t group_width = 4 * vlmax;

    for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
        const size_t actual_n = TQT_MIN(n - n_idx, n_step);
        const size_t nr = (actual_n > group_width) ? 2 : 1;
        const size_t nr_idx = nr - 1;

        kernel_table[nr_idx](m, actual_n, a + n_idx, lda, c + n_idx, init ? init + n_idx : nullptr);
    }
}
