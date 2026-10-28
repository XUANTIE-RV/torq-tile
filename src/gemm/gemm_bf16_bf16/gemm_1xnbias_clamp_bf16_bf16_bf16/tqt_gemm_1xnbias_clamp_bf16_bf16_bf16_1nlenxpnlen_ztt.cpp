//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_ztt)
#error This file requires Ztt support.
#endif

#include "tqt_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt.h"

#include "src/gemm/gemm_bf16_bf16/tqt_kernel_gemm_bf16_bf16_1nlenxpnlen_ztt.h"

using namespace tqt_gemm_bf16_bf16_1nlenxpnlen_ztt;

size_t tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt(void)
{
    return geometry::m_step();
}

size_t tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt(void)
{
    return geometry::n_step();
}

size_t tqt_get_a_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt(size_t m_idx,
                                                                          size_t k_idx, size_t lda)
{
    TQT_ASSUME_ALWAYS(m_idx % geometry::m_step() == 0);
    return geometry::offset(m_idx, k_idx, lda);
}

size_t tqt_get_b_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt(size_t n_idx,
                                                                          size_t k_idx, size_t ldb)
{
    TQT_ASSUME_ALWAYS(n_idx % geometry::n_step() == 0);
    return geometry::offset(k_idx, n_idx, ldb);
}

size_t tqt_get_c_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt(size_t m_idx,
                                                                          size_t n_idx, size_t ldc)
{
    TQT_ASSUME_ALWAYS(m_idx % geometry::m_step() == 0 && n_idx % geometry::n_step() == 0);
    return geometry::offset(m_idx, n_idx, ldc);
}

size_t tqt_get_d_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt(size_t m_idx,
                                                                          size_t n_idx, size_t ldd)
{
    TQT_ASSUME_ALWAYS(m_idx % geometry::m_step() == 0 && n_idx % geometry::n_step() == 0);
    return geometry::offset(m_idx, n_idx, ldd);
}

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt(size_t n_idx)
{
    TQT_ASSUME_ALWAYS(n_idx % geometry::n_step() == 0);
    return checked_mul(n_idx, sizeof(bfloat16_t));
}

size_t tqt_get_d_size_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt(size_t m, size_t n)
{
    return checked_mul(checked_mul(m, n), sizeof(bfloat16_t));
}

void tqt_run_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
    size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
    bfloat16_t clamp_min, bfloat16_t clamp_max)
{
    TQT_UNUSED(k_idx_a);
    TQT_UNUSED(k_idx_b);
    args values{m,
                n,
                k,
                0,
                static_cast<const bfloat16_t *>(A),
                static_cast<const bfloat16_t *>(B),
                static_cast<const bfloat16_t *>(C),
                static_cast<bfloat16_t *>(D),
                static_cast<const bfloat16_t *>(bias),
                lda,
                ldb,
                ldc,
                ldd};
    tqt_run_kernel_gemm_bf16_bf16_1nlenxpnlen_ztt<false, false, false>(values, clamp_min,
                                                                       clamp_max);
}
