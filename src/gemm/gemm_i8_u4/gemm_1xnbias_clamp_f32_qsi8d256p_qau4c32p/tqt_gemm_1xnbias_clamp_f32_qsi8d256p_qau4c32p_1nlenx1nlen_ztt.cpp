//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_ztt) || !defined(__riscv_flen) || __riscv_flen < 32
#error This file requires Ztt and scalar F support.
#endif

#include "tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt.h"

#include "src/gemm/gemm_i8_u4/tqt_kernel_gemm_i8_u4_1nlenx1nlen_ztt.h"

namespace impl = tqt_gemm_i8_u4_1nlenx1nlen_ztt;

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt(void)
{
    return impl::geometry::step();
}

size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt(void)
{
    return impl::geometry::step();
}

size_t tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt(
    size_t m_idx, size_t k_idx, size_t ld, size_t actual_m)
{
    (void)actual_m;
    return impl::geometry::offset(m_idx, k_idx, ld, false);
}

size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt(size_t m,
                                                                                       size_t k)
{
    return impl::geometry::size(m, k, false);
}

void tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt(
    size_t m, size_t k, size_t ld, size_t packed_ld, size_t k_idx, const void *src, void *dst)
{
    impl::pack<false>(m, k, ld, packed_ld, k_idx, src, dst);
}

size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt(
    size_t n_idx, size_t k_idx, size_t ld, size_t actual_n)
{
    (void)actual_n;
    return impl::geometry::offset(n_idx, k_idx, ld, true);
}

size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt(size_t n,
                                                                                       size_t k)
{
    return impl::geometry::size(n, k, true);
}

void tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt(
    size_t n, size_t k, size_t ld, size_t packed_ld, size_t k_idx, const void *src, void *dst)
{
    impl::pack<true>(n, k, ld, packed_ld, k_idx, src, dst);
}

size_t tqt_get_scale_a_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt(
    size_t row, size_t block_idx, size_t ld)
{
    return impl::checked_mul(impl::checked_add(impl::checked_mul(row, ld), block_idx),
                             sizeof(float));
}

size_t tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt(
    size_t row, size_t block_idx, size_t ld)
{
    return impl::checked_mul(impl::checked_add(impl::checked_mul(row, ld), block_idx),
                             sizeof(float));
}

size_t tqt_get_bsums_a_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt(
    size_t row, size_t block_idx, size_t ld)
{
    return impl::checked_mul(impl::checked_add(impl::checked_mul(row, ld), block_idx),
                             sizeof(int16_t));
}

size_t tqt_get_real_zero_b_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt(
    size_t row, size_t block_idx, size_t ld)
{
    return impl::checked_mul(impl::checked_add(impl::checked_mul(row, ld), block_idx),
                             sizeof(float));
}

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt(size_t m_idx,
                                                                                  size_t n_idx,
                                                                                  size_t ld)
{
    return impl::checked_mul(impl::checked_add(impl::checked_mul(m_idx, ld), n_idx), sizeof(float));
}

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt(size_t m_idx,
                                                                                  size_t n_idx,
                                                                                  size_t ld)
{
    return impl::checked_mul(impl::checked_add(impl::checked_mul(m_idx, ld), n_idx), sizeof(float));
}

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt(size_t n_idx)
{
    return impl::checked_mul(n_idx, sizeof(float));
}

size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt(size_t m, size_t n)
{
    return impl::checked_mul(impl::checked_mul(m, n), sizeof(float));
}

void tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float clamp_min, float clamp_max,
    const struct tqt_qsi8d256_qau4c32_params *params)
{
    if (m == 0 || n == 0)
        return;
    TQT_ASSUME_ALWAYS(k_idx_a == k_idx_b);
    TQT_ASSUME_ALWAYS(k == 0 || params != nullptr);
    impl::args v = {m,
                    n,
                    k,
                    k_idx_a,
                    lda,
                    ldb,
                    static_cast<const uint8_t *>(A_packed),
                    static_cast<const uint8_t *>(B_packed),
                    params ? params->scale_a : nullptr,
                    params ? params->ldsa : 0,
                    params ? params->bsums_a : nullptr,
                    params ? params->ldbsuma : 0,
                    params ? params->scale_b : nullptr,
                    params ? params->real_zero_b : nullptr,
                    params ? params->ldsb : 0,
                    static_cast<const float *>(C),
                    ldc,
                    static_cast<float *>(D),
                    ldd,
                    static_cast<const float *>(bias)};
    impl::run(v, clamp_min, clamp_max);
}
