//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Adaptive Ztt GEMM: D = clamp(C + A*B + row_bias).
/// S=get_m_step(), P*S=get_n_step(), P=max(1,UDS/16).
/// Logical M/N/K may be arbitrary; use only this implementation packing helpers.
/// A/C/D are block-row-major; B is block-column-major; square interiors are row-major.
/// C/bias may be null; C=D is allowed with identical layouts; D must not overlap A/B/bias.
/// Empty M/N is a no-op. K=0 does not access A/B. Invalid preconditions fail.
/// FP32 accumulation and bias, then RNE narrowing to the public output type.
/// Only [-INFINITY, INFINITY] skips clamp; otherwise use fminf(fmaxf(value,min),max).
/// The binary requires matching UDS and supported operation/datatype tuples.
/// Ownership is acquired only if needed; M/ACC are call-clobbered. No allocation.

size_t tqt_get_m_step_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt(void);
size_t tqt_get_n_step_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt(void);
size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt(size_t m,
                                                                                         size_t k);
size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt(size_t n,
                                                                                         size_t k);
size_t tqt_get_d_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt(size_t m,
                                                                                         size_t n);
size_t tqt_get_bias_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt(
    size_t n);
void tqt_run_a_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt(size_t m, size_t k,
                                                                                size_t lda,
                                                                                const void *A,
                                                                                void *Ap);
void tqt_run_b_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt(size_t n, size_t k,
                                                                                size_t ldb,
                                                                                const void *B,
                                                                                void *Bp);
void tqt_run_bt_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt(size_t n, size_t k,
                                                                                 size_t ldb,
                                                                                 const void *Bt,
                                                                                 void *Bp);
void tqt_run_c_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt(size_t m, size_t n,
                                                                                size_t ldc,
                                                                                const void *C,
                                                                                void *Cp);
void tqt_run_bias_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt(size_t n,
                                                                                   const void *bias,
                                                                                   void *bp);
void tqt_run_d_unpack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt(
    size_t m, size_t n, size_t ldd, const void *Dp, void *D);
void tqt_run_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt(
    size_t m, size_t n, size_t k, const void *A, const void *B, const void *C, void *D,
    const void *bias, bfloat16_t clamp_min, bfloat16_t clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif
