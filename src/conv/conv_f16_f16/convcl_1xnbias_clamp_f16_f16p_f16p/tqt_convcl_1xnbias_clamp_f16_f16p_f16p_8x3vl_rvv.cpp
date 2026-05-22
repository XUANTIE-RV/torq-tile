//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif  // Architectural features check.

#include "tqt_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv.h"

#include <stdlib.h>

#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16p_f16p/tqt_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv.h"

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(void)
{
    return tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv();
}

size_t tqt_get_n_step_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(void)
{
    return tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv();
}

size_t tqt_get_a_im2colpack_offset_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(size_t m_idx,
                                                                                size_t k_idx,
                                                                                size_t lda,
                                                                                size_t actual_m)
{
    return tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(m_idx, k_idx, lda,
                                                                              actual_m);
}

size_t tqt_get_b_packed_offset_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(size_t n_idx,
                                                                            size_t k_idx,
                                                                            size_t ldb,
                                                                            size_t actual_n)
{
    return tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(n_idx, k_idx, ldb,
                                                                              actual_n);
}

size_t tqt_get_c_offset_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldc)
{
    return tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(m_idx, n_idx, ldc);
}

size_t tqt_get_bias_offset_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(size_t n_idx)
{
    return tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(n_idx);
}

size_t tqt_get_d_offset_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldd)
{
    return tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(m_idx, n_idx, ldd);
}

size_t tqt_get_d_size_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(size_t m, size_t n)
{
    return tqt_get_d_size_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(m, n);
}

size_t tqt_get_a_im2colpack_size_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(size_t m, size_t k)
{
    return tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(m, k);
}

size_t tqt_get_b_packed_size_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(size_t n, size_t k)
{
    return tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(n, k);
}

// ============================================================================
// Pack Functions
// ============================================================================

void tqt_run_bt_pack_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(size_t n, size_t k, size_t ldb,
                                                                  size_t ldb_packed, size_t k_idx,
                                                                  const void *B, void *B_packed)
{
    tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(n, k, ldb, ldb_packed, k_idx, B,
                                                               B_packed);
}

// ============================================================================
// Fused im2col+pack for A-side (activation) in CL packed convolution
// ============================================================================

static void a_im2colpack_cl_fill(float16_t *dst, size_t actual_m, size_t row_start,
                                 const float16_t *input, size_t C_total, size_t c_offset,
                                 size_t IC_per_group, const tqt_conv_params *params)
{
    const size_t ID = params->input_shape[0], IH = params->input_shape[1],
                 IW = params->input_shape[2];
    const size_t OW = params->output_shape[2], OH = params->output_shape[1];
    const size_t KD = params->kernel_shape[0], KH = params->kernel_shape[1],
                 KW = params->kernel_shape[2];
    const size_t SD = params->stride[0], SH = params->stride[1], SW = params->stride[2];
    const size_t DD = params->dilation[0], DH = params->dilation[1], DW = params->dilation[2];
    const size_t PD = params->pad[0], PH = params->pad[1], PW = params->pad[2];

    // Pre-compute (od, oh, ow) for each row in this panel
    size_t od_r[8], oh_r[8], ow_r[8];
    for (size_t r = 0; r < actual_m; ++r) {
        size_t global_row = row_start + r;
        ow_r[r] = global_row % OW;
        oh_r[r] = (global_row / OW) % OH;
        od_r[r] = global_row / (OW * OH);
    }

    size_t k_col = 0;
    for (size_t kd = 0; kd < KD; ++kd) {
        for (size_t kh = 0; kh < KH; ++kh) {
            for (size_t kw = 0; kw < KW; ++kw) {
                for (size_t r = 0; r < actual_m; ++r) {
                    int id_in = (int)(od_r[r] * SD + kd * DD) - (int)PD;
                    int ih_in = (int)(oh_r[r] * SH + kh * DH) - (int)PH;
                    int iw_in = (int)(ow_r[r] * SW + kw * DW) - (int)PW;

                    if (id_in >= 0 && id_in < (int)ID && ih_in >= 0 && ih_in < (int)IH &&
                        iw_in >= 0 && iw_in < (int)IW) {
                        const float16_t *src =
                            input +
                            ((size_t)id_in * IH * IW + (size_t)ih_in * IW + (size_t)iw_in) *
                                C_total +
                            c_offset;
                        // vector load + strided store
                        size_t rem = IC_per_group;
                        const float16_t *s = src;
                        float16_t *d = dst + k_col * actual_m + r;
                        while (rem > 0) {
                            size_t vl = __riscv_vsetvl_e16m1(rem);
                            vfloat16m1_t v = __riscv_vle16_v_f16m1(s, vl);
                            __riscv_vsse16_v_f16m1(d, actual_m * sizeof(float16_t), v, vl);
                            s += vl;
                            d += vl * actual_m;
                            rem -= vl;
                        }
                    } else {
                        // strided store zeros
                        size_t rem = IC_per_group;
                        float16_t *d = dst + k_col * actual_m + r;
                        while (rem > 0) {
                            size_t vl = __riscv_vsetvl_e16m1(rem);
                            vfloat16m1_t v = __riscv_vfmv_v_f_f16m1((__fp16)0.0f, vl);
                            __riscv_vsse16_v_f16m1(d, actual_m * sizeof(float16_t), v, vl);
                            d += vl * actual_m;
                            rem -= vl;
                        }
                    }
                }
                k_col += IC_per_group;
            }
        }
    }
}

void tqt_run_a_im2colpack_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(
    size_t group_idx, const void *activation, size_t lda_packed, size_t k_idx, void *A_packed,
    const tqt_conv_params *params)
{
    const size_t IC_per_group = params->input_channel / params->groups;
    const size_t M = params->output_shape[0] * params->output_shape[1] * params->output_shape[2];

    const float16_t *input = (const float16_t *)activation;
    float16_t *packed = (float16_t *)A_packed;
    const size_t C_total = params->input_channel;
    const size_t c_offset = group_idx * IC_per_group;
    const size_t m_step = 8;

    for (size_t m_idx = 0; m_idx < M; m_idx += m_step) {
        const size_t actual_m = TQT_MIN(M - m_idx, m_step);
        float16_t *panel_ptr;
        if (m_idx == 0 || k_idx == 0 || actual_m == m_step)
            panel_ptr = packed + m_idx * lda_packed;
        else
            panel_ptr = packed + m_idx * lda_packed - k_idx * (m_step - actual_m);

        a_im2colpack_cl_fill(panel_ptr, actual_m, m_idx, input, C_total, c_offset, IC_per_group,
                             params);
    }
}

// ============================================================================
// Conv Main Function
// ============================================================================

void tqt_run_conv_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float16_t clamp_min, float16_t clamp_max)
{
    tqt_run_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(m, n, k, A_packed, lda, k_idx_a, B_packed,
                                                       ldb, k_idx_b, C, ldc, D, ldd, bias,
                                                       clamp_min, clamp_max);
}
