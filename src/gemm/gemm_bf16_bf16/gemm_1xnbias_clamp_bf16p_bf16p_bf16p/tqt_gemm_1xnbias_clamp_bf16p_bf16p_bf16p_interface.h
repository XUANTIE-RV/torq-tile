//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "src/common/tqt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Packed BF16 GEMM: D = clamp(C + A*B + row_bias). Logical M/N/K may be arbitrary.
/// A/C/D use block-row-major squares; B uses block-column-major squares.
/// Every square is row-major. Use the matching implementation pack/size helpers.
/// S = get_m_step(); P*S = get_n_step(), where P=max(1,UDS/16).
/// Buffers are tied to S/P. M is padded to S; N/K/bias are padded to P*S.
/// Size queries return bytes. A/B/C/bias pack functions zero all padding.
/// C uses get_d_packed_size(m,n); bias is a vector padded to get_n_step().
/// B [K,N] and Bt [N,K] pack into the same canonical B_packed representation.
/// Pack/unpack leading dimensions count elements in the ordinary source/destination.
/// Pack/unpack buffers must not overlap. The caller allocates every buffer.
/// C_packed/bias_packed may be null. C_packed=D_packed is supported; D must not
/// overlap A/B/bias. D padding is unspecified after bias/clamp; unpack ignores it.
/// Empty M/N is a no-op. K=0 does not access A/B. No packed subwindow API is provided.
/// The caller must enable Ztt state on hardware compatible with the build. GEMM owns
/// the backend only for the call unless already owned; M/Acc state is clobbered.
/// Helpers need no ownership.

typedef size_t (*tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_get_m_step_func_t)(void);
typedef size_t (*tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_get_n_step_func_t)(void);
typedef size_t (*tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_get_a_packed_size_func_t)(size_t m,
                                                                                    size_t k);
typedef size_t (*tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_get_b_packed_size_func_t)(size_t n,
                                                                                    size_t k);
typedef size_t (*tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_get_d_packed_size_func_t)(size_t m,
                                                                                    size_t n);
typedef size_t (*tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_get_bias_packed_size_func_t)(size_t n);
typedef void (*tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_run_a_pack_func_t)(size_t m, size_t k,
                                                                           size_t lda,
                                                                           const void *A,
                                                                           void *A_packed);
typedef void (*tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_run_b_pack_func_t)(size_t n, size_t k,
                                                                           size_t ldb,
                                                                           const void *B,
                                                                           void *B_packed);
typedef void (*tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_run_bt_pack_func_t)(size_t n, size_t k,
                                                                            size_t ldb,
                                                                            const void *Bt,
                                                                            void *B_packed);
typedef void (*tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_run_c_pack_func_t)(size_t m, size_t n,
                                                                           size_t ldc,
                                                                           const void *C,
                                                                           void *C_packed);
typedef void (*tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_run_bias_pack_func_t)(size_t n,
                                                                              const void *bias,
                                                                              void *bias_packed);
typedef void (*tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_run_d_unpack_func_t)(size_t m, size_t n,
                                                                             size_t ldd,
                                                                             const void *D_packed,
                                                                             void *D);
typedef void (*tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_run_gemm_func_t)(
    size_t m, size_t n, size_t k, const void *A_packed, const void *B_packed, const void *C_packed,
    void *D_packed, const void *bias_packed, bfloat16_t clamp_min, bfloat16_t clamp_max);

struct tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_ukernel
{
    tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_get_m_step_func_t get_m_step;
    tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_get_n_step_func_t get_n_step;
    tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_get_a_packed_size_func_t get_a_packed_size;
    tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_get_b_packed_size_func_t get_b_packed_size;
    tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_get_d_packed_size_func_t get_d_packed_size;
    tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_get_bias_packed_size_func_t get_bias_packed_size;
    tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_run_a_pack_func_t run_a_pack;
    tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_run_b_pack_func_t run_b_pack;
    tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_run_bt_pack_func_t run_bt_pack;
    tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_run_c_pack_func_t run_c_pack;
    tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_run_bias_pack_func_t run_bias_pack;
    tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_run_d_unpack_func_t run_d_unpack;
    tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_run_gemm_func_t run_gemm;
};

#ifdef __cplusplus
}  // extern "C"
#endif
