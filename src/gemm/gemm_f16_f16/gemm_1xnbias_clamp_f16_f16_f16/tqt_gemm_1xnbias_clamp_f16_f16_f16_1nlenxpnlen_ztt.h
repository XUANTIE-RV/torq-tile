//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "tqt_gemm_1xnbias_clamp_f16_f16_f16_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Adaptive Ztt GEMM: D = clamp(C + A*B + row_bias).
/// S=get_m_step(), P*S=get_n_step(), P=max(1,UDS/16).
/// M must be divisible by S; N and K must be divisible by P*S.
/// Strides count elements; offsets/sizes count bytes. k_idx_a/k_idx_b are unused.
/// C/bias may be null; C=D is allowed with identical strides; D must not overlap A/B/bias.
/// Empty M/N is a no-op. K=0 does not access A/B. Invalid preconditions fail.
/// Accumulation and bias use the public output type.
/// Only [-INFINITY, INFINITY] skips clamp; otherwise use fminf(fmaxf(value,min),max).
/// The binary requires matching UDS and supported operation/datatype tuples.
/// Ownership is acquired only if needed; M/ACC are call-clobbered. No allocation.

size_t tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt(void);
size_t tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt(void);
size_t tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt(size_t m_idx, size_t k_idx,
                                                                       size_t lda);
size_t tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt(size_t n_idx, size_t k_idx,
                                                                       size_t ldb);
size_t tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt(size_t m_idx, size_t n_idx,
                                                                       size_t ldc);
size_t tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt(size_t m_idx, size_t n_idx,
                                                                       size_t ldd);
size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt(size_t n_idx);
size_t tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt(size_t m, size_t n);
void tqt_run_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
    size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
    float16_t clamp_min, float16_t clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif
