//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif  // Architectural features check.

#include "tqt_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_f16_i4/tqt_kernel_gemm_f16_i4_8x1vl_rvv.h"

static inline size_t get_vlmax(size_t vlenb)
{
    return vlenb / sizeof(float16_t);
}

static inline size_t get_n_step(size_t vlenb)
{
    return get_vlmax(vlenb);
}

static const size_t m_step = 8;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv(void)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = get_n_step(VLENB);

    return n_step;
}

size_t tqt_get_a_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv(size_t m_idx, size_t k_idx,
                                                                     size_t lda, size_t bl)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(bl >= 2 && bl % 2 == 0);
    TQT_ASSUME(k_idx % bl == 0);
    return (m_idx * lda + k_idx) * sizeof(float16_t);
}

size_t tqt_get_b_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv(size_t n_idx, size_t k_idx,
                                                                     size_t ldb, size_t bl)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = get_n_step(VLENB);

    TQT_ASSUME(n_idx % n_step == 0);
    TQT_ASSUME(bl >= 2 && bl % 2 == 0);
    TQT_ASSUME(k_idx % bl == 0);
    return n_idx * (ldb / 2) + k_idx / 2;
}

size_t tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv(size_t n_idx,
                                                                           size_t b_idx,
                                                                           size_t ldsb)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = get_n_step(VLENB);

    TQT_ASSUME(n_idx % n_step == 0);
    return (n_idx * ldsb + b_idx) * sizeof(float16_t);
}

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldc)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = get_n_step(VLENB);

    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(float);
}

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv(size_t n_idx)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = get_n_step(VLENB);

    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(float);
}

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldd)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = get_n_step(VLENB);

    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(float);
}

size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(float);
}

template <size_t MR>
static void gemm_kernel(size_t vl, size_t k, size_t bl, const float16_t *A, size_t lda,
                        const uint8_t *B, size_t ldb, const void *scale_b, size_t ldsb,
                        const float *offset_b, size_t ldob, const float *C, size_t ldc, float *D,
                        size_t ldd, const float *bias, float clamp_min, float clamp_max,
                        bool apply_clamp, const float16_t *lut)
{
    tqt_run_kernel_gemm_f16_i4_8x1vl_rvv<MR, false, false>(vl, k, bl, A, lda, B, ldb, scale_b, ldsb,
                                                           offset_b, ldob, C, ldc, D, ldd, bias,
                                                           clamp_min, clamp_max, apply_clamp, lut);
}
using gemm_kernel_fn_t = decltype(&gemm_kernel<1>);
static constexpr gemm_kernel_fn_t kernel_table[8] = {gemm_kernel<1>, gemm_kernel<2>, gemm_kernel<3>,
                                                     gemm_kernel<4>, gemm_kernel<5>, gemm_kernel<6>,
                                                     gemm_kernel<7>, gemm_kernel<8>};

void tqt_run_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv(
    size_t m, size_t n, size_t k, size_t bl, const void *A, size_t lda, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b, const void *scale_b, size_t ldsb, const void *C,
    size_t ldc, void *D, size_t ldd, const void *bias, float clamp_min, float clamp_max,
    const float16_t *lut)
{
    TQT_ASSUME(bl >= 2 && bl % 2 == 0);
    TQT_ASSUME(k % bl == 0 && k_idx_a % bl == 0 && k_idx_b % bl == 0);
    TQT_ASSUME(ldb % 2 == 0);
    const size_t ns = get_n_step(tqt_get_vlenb());
    const bool apply_clamp = tqt_clamp_is_enabled(clamp_min, clamp_max);
    for (size_t mi = 0; mi < m; mi += m_step) {
        const size_t mr = TQT_MIN(m - mi, m_step);
        const float16_t *ap = (const float16_t *)A + mi * lda;
        for (size_t ni = 0; ni < n; ni += ns) {
            const size_t nr = TQT_MIN(n - ni, ns);
            const uint8_t *bp = (const uint8_t *)B + ni * (ldb / 2);
            kernel_table[mr - 1](
                nr, k, bl, ap, lda, bp, ldb, (const float16_t *)scale_b + ni * ldsb, ldsb, nullptr,
                0, C ? (const float *)C + mi * ldc + ni : nullptr, ldc, (float *)D + mi * ldd + ni,
                ldd, bias ? (const float *)bias + ni : nullptr, clamp_min, clamp_max, apply_clamp,
                lut);
        }
    }
}
