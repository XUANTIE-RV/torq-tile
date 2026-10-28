//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_ztt)
#error This file requires Ztt support.
#endif

#include "tqt_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt.h"

#include "src/gemm/gemm_f32_f32/tqt_kernel_gemm_f32_f32_1nlenxpnlen_ztt.h"

using namespace tqt_gemm_f32_f32_1nlenxpnlen_ztt;

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt(void)
{
    return geometry::m_step();
}

size_t tqt_get_n_step_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt(void)
{
    return geometry::n_step();
}

size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt(size_t m, size_t k)
{
    return geometry::size(m, k);
}

size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt(size_t n, size_t k)
{
    return geometry::size(k, n, true);
}

size_t tqt_get_d_packed_size_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt(size_t m, size_t n)
{
    return geometry::size(m, n);
}

size_t tqt_get_bias_packed_size_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt(size_t n)
{
    return n == 0 ? 0 : checked_mul(round_up(n, geometry::n_step()), sizeof(float));
}

void tqt_run_a_pack_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt(size_t m, size_t k,
                                                                      size_t lda, const void *A,
                                                                      void *Ap)
{
    pack<>(m, k, lda, A, Ap);
}

void tqt_run_b_pack_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt(size_t n, size_t k,
                                                                      size_t ldb, const void *B,
                                                                      void *Bp)
{
    pack<true>(k, n, ldb, B, Bp);
}

void tqt_run_bt_pack_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt(size_t n, size_t k,
                                                                       size_t ldb, const void *Bt,
                                                                       void *Bp)
{
    pack<true, true>(k, n, ldb, Bt, Bp);
}

void tqt_run_c_pack_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt(size_t m, size_t n,
                                                                      size_t ldc, const void *C,
                                                                      void *Cp)
{
    pack<>(m, n, ldc, C, Cp);
}

void tqt_run_bias_pack_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt(size_t n, const void *bias,
                                                                         void *bp)
{
    if (n == 0) {
        return;
    }
    TQT_ASSUME_ALWAYS(bias != nullptr && bp != nullptr && bias != bp);
    const size_t bytes =
        tqt_get_bias_packed_size_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt(n);
    const size_t used = checked_mul(n, sizeof(float));
    TQT_MEMCPY(bp, bias, used);
    TQT_MEMSET(static_cast<char *>(bp) + used, 0, bytes - used);
}

void tqt_run_d_unpack_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt(size_t m, size_t n,
                                                                        size_t ldd, const void *Dp,
                                                                        void *D)
{
    unpack(m, n, ldd, Dp, D);
}

void tqt_run_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt(size_t m, size_t n, size_t k,
                                                               const void *A, const void *B,
                                                               const void *C, void *D,
                                                               const void *bias, float clamp_min,
                                                               float clamp_max)
{
    args values{m,
                n,
                k,
                0,
                static_cast<const float *>(A),
                static_cast<const float *>(B),
                static_cast<const float *>(C),
                static_cast<float *>(D),
                static_cast<const float *>(bias),
                0,
                0,
                0,
                0};
    tqt_run_kernel_gemm_f32_f32_1nlenxpnlen_ztt<false, true, false>(values, clamp_min, clamp_max);
}
