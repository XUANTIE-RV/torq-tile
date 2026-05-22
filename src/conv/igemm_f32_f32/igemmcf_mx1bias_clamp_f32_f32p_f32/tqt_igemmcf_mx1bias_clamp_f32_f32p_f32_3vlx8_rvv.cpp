//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv.h"

#include "src/common/tqt_common.h"
#include "src/common/tqt_gemm_pack_rvv.h"
#include "src/conv/igemm_f32_f32/tqt_kernel_igemmcf_f32_f32_3vlx8_rvv.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB / sizeof(float);
static const size_t m_step = 3 * vlmax;
static const size_t n_step = 8;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(void)
{
    return n_step;
}

size_t tqt_get_a_packed_offset_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(size_t m_idx,
                                                                            size_t k_idx,
                                                                            size_t lda,
                                                                            size_t actual_m)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * lda + k_idx * TQT_MIN(actual_m, m_step)) * sizeof(float);
}

size_t tqt_get_c_offset_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldc)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(float);
}

size_t tqt_get_bias_offset_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(size_t m_idx)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return m_idx * sizeof(float);
}

size_t tqt_get_d_offset_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldd)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(float);
}

size_t tqt_get_d_size_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(size_t m, size_t n)
{
    return m * n * sizeof(float);
}

size_t tqt_get_a_packed_size_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(size_t m, size_t k)
{
    return m * k * sizeof(float);
}

// ============================================================================
// Weight Pack Function
// ============================================================================

void tqt_run_a_pack_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(size_t m, size_t k, size_t ldw,
                                                                 size_t ldw_packed, size_t k_idx,
                                                                 const void *weight,
                                                                 void *weight_packed)
{
    // Reuse existing pack function: [OC, K] -> [OC/3vl, K, 3vl]
    // This is structurally identical to transposed B packing [N, K] -> [N/3vl, K, 3vl]
    tqt_gemm_pack_nxk_1x3vl_e32_rvv(m, k, ldw, ldw_packed, k_idx, weight, weight_packed);
}

// ============================================================================
// Internal: Template-based igemmcf kernel (packed weight)
// ============================================================================

template <size_t MR, size_t NR>
static void igemmcf_packed_kernel(size_t K_inner, size_t ic_start, size_t ic_end, float *D,
                                  const float *W, const float *input_group, const float *C,
                                  const float *bias, size_t ldd, size_t ldc, float clamp_min,
                                  float clamp_max, size_t vl, const int *od_arr, const int *oh_arr,
                                  const int *ow_arr, size_t ID, size_t IH, size_t IW, size_t KD,
                                  size_t KH, size_t KW, size_t SD, size_t SH, size_t SW, size_t DD,
                                  size_t DH, size_t DW, size_t PD, size_t PH, size_t PW,
                                  bool is_first_k, bool is_last_k)
{
    static_assert(MR >= 1 && MR <= 3, "MR must be in range [1, 3]");
    static_assert(NR >= 1 && NR <= 8, "NR must be in range [1, 8]");

    // Init accumulators
    if (is_first_k) {
        tqt_init_zero_kernel_gemm_f32_f32_8x3vl_rvv<NR, MR>(vl);
        if (bias) {
            tqt_add_1xnbias_kernel_gemm_f32_f32_8x3vl_rvv<NR, MR>(vl, bias);
        }
    } else {
        tqt_init_c_strided_kernel_igemmcf_f32_f32_3vlx8_rvv<MR, NR>(vl, C, ldc, vlmax);
    }

    // K-loop: for ic { for kd { for kh { for kw { ... } } } }
    const float *w_ptr = W;
    for (size_t ic = ic_start; ic < ic_end; ic++) {
        const float *b_base = input_group + ic * ID * IH * IW;
        for (size_t kd = 0; kd < KD; kd++) {
            int id[8];
            bool flag_d[8];
            for (size_t i = 0; i < NR; i++) {
                id[i] = (int)(od_arr[i] * SD + kd * DD) - (int)PD;
                flag_d[i] = (id[i] < 0 || id[i] >= (int)ID);
            }
            for (size_t kh = 0; kh < KH; kh++) {
                int ih[8];
                bool flag_h[8];
                for (size_t i = 0; i < NR; i++) {
                    ih[i] = (int)(oh_arr[i] * SH + kh * DH) - (int)PH;
                    flag_h[i] = flag_d[i] || (ih[i] < 0 || ih[i] >= (int)IH);
                }
                for (size_t kw = 0; kw < KW; kw++) {
                    float b_data[8] = {0};
                    for (size_t i = 0; i < NR; i++) {
                        int iw = (int)(ow_arr[i] * SW + kw * DW) - (int)PW;
                        if (!flag_h[i] && iw >= 0 && iw < (int)IW) {
                            b_data[i] = b_base[id[i] * IH * IW + ih[i] * IW + iw];
                        }
                    }
                    // Load NR input scalars into ft0..ft(NR-1)
                    tqt_load_a_kernel_gemm_f32_f32_8x3vl_rvv<NR>(b_data, 1);
                    // Load MR packed weight vectors (contiguous) into v0..v(MR-1)
                    tqt_load_b_kernel_gemm_f32_f32_8x3vl_rvv<MR>(vl, w_ptr, vlmax);
                    // FMA
                    tqt_vfmacc_kernel_gemm_f32_f32_8x3vl_rvv<NR, MR>(vl);
                    w_ptr += MR * vl;
                }
            }
        }
    }

    // Post-processing
    if (is_last_k) {
        tqt_clamp_kernel_gemm_f32_f32_8x3vl_rvv<NR, MR>(vl, clamp_min, clamp_max);
    }
    tqt_store_strided_kernel_igemmcf_f32_f32_3vlx8_rvv<MR, NR>(vl, D, ldd, vlmax);
}

// Function pointer type for kernel dispatch
using igemmcf_packed_kernel_fn_t = void (*)(size_t K_inner, size_t ic_start, size_t ic_end,
                                            float *D, const float *W, const float *input_group,
                                            const float *C, const float *bias, size_t ldd,
                                            size_t ldc, float clamp_min, float clamp_max, size_t vl,
                                            const int *od_arr, const int *oh_arr, const int *ow_arr,
                                            size_t ID, size_t IH, size_t IW, size_t KD, size_t KH,
                                            size_t KW, size_t SD, size_t SH, size_t SW, size_t DD,
                                            size_t DH, size_t DW, size_t PD, size_t PH, size_t PW,
                                            bool is_first_k, bool is_last_k);

// Kernel dispatch table: [MR][NR] = [3][8]
static constexpr igemmcf_packed_kernel_fn_t kernel_table[3][8] = {
    // MR = 1
    {igemmcf_packed_kernel<1, 1>, igemmcf_packed_kernel<1, 2>, igemmcf_packed_kernel<1, 3>,
     igemmcf_packed_kernel<1, 4>, igemmcf_packed_kernel<1, 5>, igemmcf_packed_kernel<1, 6>,
     igemmcf_packed_kernel<1, 7>, igemmcf_packed_kernel<1, 8>},
    // MR = 2
    {igemmcf_packed_kernel<2, 1>, igemmcf_packed_kernel<2, 2>, igemmcf_packed_kernel<2, 3>,
     igemmcf_packed_kernel<2, 4>, igemmcf_packed_kernel<2, 5>, igemmcf_packed_kernel<2, 6>,
     igemmcf_packed_kernel<2, 7>, igemmcf_packed_kernel<2, 8>},
    // MR = 3
    {igemmcf_packed_kernel<3, 1>, igemmcf_packed_kernel<3, 2>, igemmcf_packed_kernel<3, 3>,
     igemmcf_packed_kernel<3, 4>, igemmcf_packed_kernel<3, 5>, igemmcf_packed_kernel<3, 6>,
     igemmcf_packed_kernel<3, 7>, igemmcf_packed_kernel<3, 8>},
};

static void igemmcf_packed_dispatch(size_t M, size_t N, size_t K_inner, size_t ic_start,
                                    size_t ic_end, float *D, const float *W,
                                    const float *input_group, const float *C, const float *bias,
                                    size_t ldd, size_t ldc, float clamp_min, float clamp_max,
                                    size_t vl, const int *od_arr, const int *oh_arr,
                                    const int *ow_arr, size_t ID, size_t IH, size_t IW, size_t KD,
                                    size_t KH, size_t KW, size_t SD, size_t SH, size_t SW,
                                    size_t DD, size_t DH, size_t DW, size_t PD, size_t PH,
                                    size_t PW, bool is_first_k, bool is_last_k)
{
    TQT_ASSUME(M == 3 * vlmax || M == 2 * vlmax || (M > 0 && M <= vlmax));
    TQT_ASSUME(N > 0 && N <= 8);
    TQT_ASSUME(vl <= vlmax);

    size_t mr_idx = (M + vlmax - 1) / vlmax - 1;
    size_t nr_idx = N - 1;

    kernel_table[mr_idx][nr_idx](K_inner, ic_start, ic_end, D, W, input_group, C, bias, ldd, ldc,
                                 clamp_min, clamp_max, vl, od_arr, oh_arr, ow_arr, ID, IH, IW, KD,
                                 KH, KW, SD, SH, SW, DD, DH, DW, PD, PH, PW, is_first_k, is_last_k);
}

// ============================================================================
// Igemm Main Function (packed weight)
// ============================================================================

void tqt_run_igemm_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(
    size_t m, size_t n, size_t k, const void *input, size_t group_idx, size_t n_idx, size_t k_idx_b,
    const void *A_packed, size_t lda, size_t k_idx_a, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float clamp_min, float clamp_max, const tqt_conv_params *params)
{
    // Extract convolution parameters
    const size_t IC_per_group = params->input_channel / params->groups;
    const size_t ID = params->input_shape[0];
    const size_t IH = params->input_shape[1];
    const size_t IW = params->input_shape[2];
    const size_t OD = params->output_shape[0];
    const size_t OH = params->output_shape[1];
    const size_t OW = params->output_shape[2];
    const size_t KD = params->kernel_shape[0];
    const size_t KH = params->kernel_shape[1];
    const size_t KW = params->kernel_shape[2];
    const size_t SD = params->stride[0];
    const size_t SH = params->stride[1];
    const size_t SW = params->stride[2];
    const size_t DD = params->dilation[0];
    const size_t DH = params->dilation[1];
    const size_t DW = params->dilation[2];
    const size_t PD = params->pad[0];
    const size_t PH = params->pad[1];
    const size_t PW = params->pad[2];

    const size_t K_inner = KD * KH * KW;
    const size_t K_total = IC_per_group * K_inner;

    TQT_ASSERT(k_idx_a % K_inner == 0);
    TQT_ASSERT(k_idx_b % K_inner == 0);

    const bool is_first_k = (k_idx_b == 0);
    const bool is_last_k = (k_idx_b + k == K_total);
    const size_t ic_start = k_idx_b / K_inner;
    const size_t ic_end = (k_idx_b + k + K_inner - 1) / K_inner;

    const float *input_group = (const float *)input + group_idx * IC_per_group * ID * IH * IW;

    // N tiling (spatial direction)
    for (size_t n_idx_local = 0; n_idx_local < n; n_idx_local += n_step) {
        const size_t actual_n = TQT_MIN(n - n_idx_local, n_step);
        const size_t global_n = n_idx + n_idx_local;

        // Precompute output spatial coordinates
        int od_arr[8] = {0};
        int oh_arr[8] = {0};
        int ow_arr[8] = {0};
        for (size_t i = 0; i < actual_n; i++) {
            od_arr[i] = (int)((global_n + i) / (OH * OW));
            oh_arr[i] = (int)(((global_n + i) / OW) % OH);
            ow_arr[i] = (int)((global_n + i) % OW);
        }

        // M tiling (OC direction)
        for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
            const size_t actual_m = TQT_MIN(m - m_idx, m_step);

            // Packed weight pointer: locate the packed block for this OC tile
            const float *w_ptr;
            if (actual_m == m_step || k_idx_a == 0) {
                w_ptr = (const float *)A_packed + m_idx * lda;
            } else {
                w_ptr = (const float *)A_packed + m_idx * lda - k_idx_a * (m_step - actual_m);
            }

            const float *bias_ptr = bias ? (const float *)bias + m_idx : NULL;
            const float *c_ptr = C ? (const float *)C + m_idx * ldc + global_n : NULL;
            float *d_ptr = (float *)D + m_idx * ldd + global_n;

            // M tail handling
            size_t size_m = 0;
            if (actual_m >= vlmax) {
                if (actual_m == 3 * vlmax) {
                    size_m = 3 * vlmax;
                } else if (actual_m >= 2 * vlmax) {
                    size_m = 2 * vlmax;
                } else {
                    size_m = vlmax;
                }
                igemmcf_packed_dispatch(
                    size_m, actual_n, K_inner, ic_start, ic_end, d_ptr, w_ptr, input_group, c_ptr,
                    bias_ptr, ldd, ldc, clamp_min, clamp_max, vlmax, od_arr, oh_arr, ow_arr, ID, IH,
                    IW, KD, KH, KW, SD, SH, SW, DD, DH, DW, PD, PH, PW, is_first_k, is_last_k);
            }
            size_t remain_m = actual_m - size_m;
            if (remain_m > 0) {
                // Advance pointers for remainder sub-block
                const float *w_remain = w_ptr + size_m * lda;
                const float *bias_remain = bias_ptr ? bias_ptr + size_m : NULL;
                const float *c_remain = c_ptr ? c_ptr + size_m * ldc : NULL;
                float *d_remain = d_ptr + size_m * ldd;
                igemmcf_packed_dispatch(remain_m, actual_n, K_inner, ic_start, ic_end, d_remain,
                                        w_remain, input_group, c_remain, bias_remain, ldd, ldc,
                                        clamp_min, clamp_max, remain_m, od_arr, oh_arr, ow_arr, ID,
                                        IH, IW, KD, KH, KW, SD, SH, SW, DD, DH, DW, PD, PH, PW,
                                        is_first_k, is_last_k);
            }
        }
    }
}
