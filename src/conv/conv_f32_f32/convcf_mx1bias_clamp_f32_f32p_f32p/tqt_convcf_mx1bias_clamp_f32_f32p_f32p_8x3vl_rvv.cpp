//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv.h"

#include <stdlib.h>

#include "src/gemm/gemm_f32_f32/gemm_mx1bias_clamp_f32_f32p_f32p/tqt_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv.h"

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(void)
{
    return tqt_get_m_step_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv();
}

size_t tqt_get_n_step_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(void)
{
    return tqt_get_n_step_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv();
}

size_t tqt_get_a_packed_offset_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t m_idx,
                                                                            size_t k_idx,
                                                                            size_t lda,
                                                                            size_t actual_m)
{
    return tqt_get_a_packed_offset_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(m_idx, k_idx, lda,
                                                                              actual_m);
}

size_t tqt_get_c_offset_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldc)
{
    return tqt_get_c_offset_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(m_idx, n_idx, ldc);
}

size_t tqt_get_bias_offset_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t m_idx)
{
    return tqt_get_bias_offset_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(m_idx);
}

size_t tqt_get_d_offset_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldd)
{
    return tqt_get_d_offset_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(m_idx, n_idx, ldd);
}

size_t tqt_get_d_size_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t m, size_t n)
{
    return tqt_get_d_size_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(m, n);
}

size_t tqt_get_a_packed_size_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t m, size_t k)
{
    return tqt_get_a_packed_size_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(m, k);
}

size_t tqt_get_b_im2colpack_size_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t n, size_t k)
{
    return tqt_get_b_packed_size_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(n, k);
}

size_t tqt_get_b_im2colpack_offset_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t n_idx,
                                                                                size_t k_idx,
                                                                                size_t ldb,
                                                                                size_t actual_n)
{
    return tqt_get_b_packed_offset_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(n_idx, k_idx, ldb,
                                                                              actual_n);
}

// ============================================================================
// Pack Functions
// ============================================================================

void tqt_run_a_pack_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t m, size_t k, size_t lda,
                                                                 size_t lda_packed, size_t k_idx,
                                                                 const void *A, void *A_packed)
{
    tqt_run_a_pack_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(m, k, lda, lda_packed, k_idx, A,
                                                              A_packed);
}

// ============================================================================
// Fused im2col+pack for B-side (activation) in CF packed convolution
// ============================================================================

static void b_im2colpack_cf_fill(float *dst, size_t sub_n, size_t col_start,
                                 const float *input_group, size_t IC_per_group,
                                 const tqt_conv_params *params)
{
    const size_t ID = params->input_shape[0], IH = params->input_shape[1],
                 IW = params->input_shape[2];
    const size_t OW = params->output_shape[2], OH = params->output_shape[1];
    const size_t KD = params->kernel_shape[0], KH = params->kernel_shape[1],
                 KW = params->kernel_shape[2];
    const size_t SD = params->stride[0], SH = params->stride[1], SW = params->stride[2];
    const size_t DD = params->dilation[0], DH = params->dilation[1], DW = params->dilation[2];
    const size_t PD = params->pad[0], PH = params->pad[1], PW = params->pad[2];

    size_t k_row = 0;
    for (size_t ic = 0; ic < IC_per_group; ++ic) {
        for (size_t kd = 0; kd < KD; ++kd) {
            for (size_t kh = 0; kh < KH; ++kh) {
                for (size_t kw = 0; kw < KW; ++kw) {
                    float *row_dst = dst + k_row * sub_n;
                    size_t remaining = sub_n;
                    size_t col = col_start;
                    size_t wp = 0;

                    while (remaining > 0) {
                        size_t ow_v = col % OW;
                        size_t oh_v = (col / OW) % OH;
                        size_t od_v = col / (OW * OH);
                        size_t cnt = TQT_MIN(OW - ow_v, remaining);

                        int id_in = (int)(od_v * SD + kd * DD) - (int)PD;
                        int ih_in = (int)(oh_v * SH + kh * DH) - (int)PH;

                        if (id_in < 0 || id_in >= (int)ID || ih_in < 0 || ih_in >= (int)IH) {
                            memset(row_dst + wp, 0, cnt * sizeof(float));
                        } else {
                            // Contiguous vector copy when stride==1 && dilation==1
                            if (SW == 1 && DW == 1) {
                                int iw_s = (int)(ow_v + kw) - (int)PW;
                                size_t pl = (iw_s < 0) ? TQT_MIN(cnt, (size_t)(-iw_s)) : 0;
                                int iw_e = iw_s + (int)cnt - 1;
                                size_t pr = (iw_e >= (int)IW)
                                                ? TQT_MIN(cnt - pl, (size_t)(iw_e - (int)IW + 1))
                                                : 0;
                                size_t vc = cnt - pl - pr;
                                if (pl)
                                    memset(row_dst + wp, 0, pl * sizeof(float));
                                if (vc) {
                                    const float *s = input_group + ic * ID * IH * IW +
                                                     (size_t)id_in * IH * IW + (size_t)ih_in * IW +
                                                     (size_t)(iw_s + (int)pl);
                                    float *d = row_dst + wp + pl;
                                    size_t r = vc;
                                    while (r > 0) {
                                        size_t vl = __riscv_vsetvl_e32m1(r);
                                        __riscv_vse32_v_f32m1(d, __riscv_vle32_v_f32m1(s, vl), vl);
                                        s += vl;
                                        d += vl;
                                        r -= vl;
                                    }
                                }
                                if (pr)
                                    memset(row_dst + wp + cnt - pr, 0, pr * sizeof(float));
                            } else {
                                // Non-unit stride/dilation: scalar fallback
                                for (size_t i = 0; i < cnt; ++i) {
                                    int iw_in = (int)((ow_v + i) * SW + kw * DW) - (int)PW;
                                    if (iw_in >= 0 && iw_in < (int)IW)
                                        row_dst[wp + i] =
                                            input_group[ic * ID * IH * IW +
                                                        (size_t)id_in * IH * IW +
                                                        (size_t)ih_in * IW + (size_t)iw_in];
                                    else
                                        row_dst[wp + i] = 0.0f;
                                }
                            }
                        }
                        wp += cnt;
                        remaining -= cnt;
                        col += cnt;
                    }
                    k_row++;
                }
            }
        }
    }
}

void tqt_run_b_im2colpack_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(
    size_t group_idx, const void *activation, size_t ldb_packed, size_t k_idx, void *B_packed,
    const tqt_conv_params *params)
{
    const size_t IC_per_group = params->input_channel / params->groups;
    const size_t N = params->output_shape[0] * params->output_shape[1] * params->output_shape[2];

    const float *input = (const float *)activation;
    const float *input_group = input + group_idx * IC_per_group * params->input_shape[0] *
                                           params->input_shape[1] * params->input_shape[2];
    float *packed = (float *)B_packed;

    const size_t vlmax = csrr_vlenb() / sizeof(float);
    const size_t n_step = 3 * vlmax;

    for (size_t n_idx = 0; n_idx < N; n_idx += n_step) {
        const size_t actual_n = TQT_MIN(N - n_idx, n_step);
        float *panel_ptr;
        if (n_idx == 0 || k_idx == 0 || actual_n == n_step)
            panel_ptr = packed + n_idx * ldb_packed;
        else
            panel_ptr = packed + n_idx * ldb_packed - k_idx * (n_step - actual_n);

        size_t size_n = 0;
        if (actual_n >= vlmax) {
            if (actual_n == 3 * vlmax)
                size_n = 3 * vlmax;
            else if (actual_n >= 2 * vlmax)
                size_n = 2 * vlmax;
            else
                size_n = vlmax;
        }
        size_t remain_n = actual_n - size_n;

        if (size_n > 0)
            b_im2colpack_cf_fill(panel_ptr, size_n, n_idx, input_group, IC_per_group, params);
        if (remain_n > 0)
            b_im2colpack_cf_fill(panel_ptr + size_n * ldb_packed, remain_n, n_idx + size_n,
                                 input_group, IC_per_group, params);
    }
}

// ============================================================================
// Conv Main Function
// ============================================================================

void tqt_run_conv_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float clamp_min, float clamp_max)
{
    tqt_run_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(m, n, k, A_packed, lda, k_idx_a, B_packed,
                                                       ldb, k_idx_b, C, ldc, D, ldd, bias,
                                                       clamp_min, clamp_max);
}
