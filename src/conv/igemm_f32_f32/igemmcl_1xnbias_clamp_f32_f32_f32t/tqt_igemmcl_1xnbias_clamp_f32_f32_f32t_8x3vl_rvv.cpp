//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_f32_f32/tqt_kernel_gemm_f32_f32_8x3vl_rvv.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB / sizeof(float);
static const size_t m_step = 8;
static const size_t n_step = 3 * vlmax;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(void)
{
    return n_step;
}

size_t tqt_get_b_offset_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t n_idx, size_t k_idx,
                                                                     size_t ldb)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return (n_idx * ldb + k_idx) * sizeof(float);
}

size_t tqt_get_c_offset_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldc)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(float);
}

size_t tqt_get_bias_offset_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t n_idx)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(float);
}

size_t tqt_get_d_offset_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldd)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(float);
}

size_t tqt_get_d_size_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(float);
}

// ============================================================================
// Internal: Template-based igemmcl kernel (transposed weight)
// ============================================================================

// Direct reuse of GEMM 8x3vl kernel functions — NO template swap needed.
// igemmcl vectorization direction (N=OC) matches GEMM N direction.
// CL output [spatial, OC] is contiguous in OC → standard vle32/vse32.
template <size_t MR, size_t NR>
static void igemmcl_kernel(size_t IC, size_t IC_stride, size_t kd_start, size_t kd_end, size_t KH,
                           size_t KW, float *D, const float *W, const float *input_group,
                           const float *C, const float *bias, size_t ldd, size_t ldw, size_t ldc,
                           float clamp_min, float clamp_max, size_t vl, const int *od_arr,
                           const int *oh_arr, const int *ow_arr, size_t ID, size_t IH, size_t IW,
                           size_t SD, size_t SH, size_t SW, size_t DD, size_t DH, size_t DW,
                           size_t PD, size_t PH, size_t PW, bool is_first_k, bool is_last_k)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    size_t c_stride_row = ldc;
    size_t c_stride_col = vlmax;
    size_t d_stride_row = ldd;
    size_t d_stride_col = vlmax;

    // Init accumulators
    if (is_first_k) {
        tqt_init_zero_kernel_gemm_f32_f32_8x3vl_rvv<MR, NR>(vl);
        if (bias) {
            tqt_add_1xnbias_kernel_gemm_f32_f32_8x3vl_rvv<MR, NR>(vl, bias);
        }
    } else {
        tqt_init_c_kernel_gemm_f32_f32_8x3vl_rvv<MR, NR>(vl, C, c_stride_row, c_stride_col);
    }

    // K-loop: for kd { for kh { for kw { for ic { ... } } } }
    // CL input layout: [ID, IH, IW, IC_total] — IC_stride is total channel count
    const float *w_ptr = W;
    for (size_t kd = kd_start; kd < kd_end; kd++) {
        int id_arr[8];
        bool flag_d[8];
        for (size_t i = 0; i < MR; i++) {
            id_arr[i] = (int)(od_arr[i] * SD + kd * DD) - (int)PD;
            flag_d[i] = (id_arr[i] < 0 || id_arr[i] >= (int)ID);
        }
        for (size_t kh = 0; kh < KH; kh++) {
            int ih_arr[8];
            bool flag_h[8];
            for (size_t i = 0; i < MR; i++) {
                ih_arr[i] = (int)(oh_arr[i] * SH + kh * DH) - (int)PH;
                flag_h[i] = flag_d[i] || (ih_arr[i] < 0 || ih_arr[i] >= (int)IH);
            }
            for (size_t kw = 0; kw < KW; kw++) {
                // Precompute base addresses for MR spatial positions.
                // CL: input[id, ih, iw, ic] = base + id*IH*IW*IC + ih*IW*IC + iw*IC + ic
                // Base address (without ic) is stable for the entire IC loop.
                const float *b_bases[8] = {nullptr};
                for (size_t i = 0; i < MR; i++) {
                    int iw_val = (int)(ow_arr[i] * SW + kw * DW) - (int)PW;
                    if (!flag_h[i] && iw_val >= 0 && iw_val < (int)IW) {
                        b_bases[i] = input_group + id_arr[i] * IH * IW * IC_stride +
                                     ih_arr[i] * IW * IC_stride + iw_val * IC_stride;
                    }
                }

                // IC inner loop — boundary check already done, no branches here
                for (size_t ic = 0; ic < IC; ic++) {
                    float a_data[8] = {0};
                    for (size_t i = 0; i < MR; i++) {
                        if (b_bases[i]) {
                            a_data[i] = b_bases[i][ic];
                        }
                    }
                    // Load MR input scalars into ft0..ft(MR-1)
                    tqt_load_a_kernel_gemm_f32_f32_8x3vl_rvv<MR>(a_data, 1);
                    // Load NR weight vectors (transposed B, strided) into v0..v(NR-1)
                    tqt_load_bt_kernel_gemm_f32_f32_8x3vl_rvv<NR>(vl, w_ptr, ldw);
                    // FMA
                    tqt_vfmacc_kernel_gemm_f32_f32_8x3vl_rvv<MR, NR>(vl);
                    w_ptr += 1;
                }
            }
        }
    }

    // Post-processing
    if (is_last_k) {
        tqt_clamp_kernel_gemm_f32_f32_8x3vl_rvv<MR, NR>(vl, clamp_min, clamp_max);
    }
    tqt_store_kernel_gemm_f32_f32_8x3vl_rvv<MR, NR>(vl, D, d_stride_row, d_stride_col);
}

// Function pointer type for kernel dispatch
using igemmcl_kernel_fn_t = void (*)(size_t IC, size_t IC_stride, size_t kd_start, size_t kd_end,
                                     size_t KH, size_t KW, float *D, const float *W,
                                     const float *input_group, const float *C, const float *bias,
                                     size_t ldd, size_t ldw, size_t ldc, float clamp_min,
                                     float clamp_max, size_t vl, const int *od_arr,
                                     const int *oh_arr, const int *ow_arr, size_t ID, size_t IH,
                                     size_t IW, size_t SD, size_t SH, size_t SW, size_t DD,
                                     size_t DH, size_t DW, size_t PD, size_t PH, size_t PW,
                                     bool is_first_k, bool is_last_k);

// Kernel dispatch table: [MR][NR] = [8][3]
static constexpr igemmcl_kernel_fn_t kernel_table[8][3] = {
    {igemmcl_kernel<1, 1>, igemmcl_kernel<1, 2>, igemmcl_kernel<1, 3>},
    {igemmcl_kernel<2, 1>, igemmcl_kernel<2, 2>, igemmcl_kernel<2, 3>},
    {igemmcl_kernel<3, 1>, igemmcl_kernel<3, 2>, igemmcl_kernel<3, 3>},
    {igemmcl_kernel<4, 1>, igemmcl_kernel<4, 2>, igemmcl_kernel<4, 3>},
    {igemmcl_kernel<5, 1>, igemmcl_kernel<5, 2>, igemmcl_kernel<5, 3>},
    {igemmcl_kernel<6, 1>, igemmcl_kernel<6, 2>, igemmcl_kernel<6, 3>},
    {igemmcl_kernel<7, 1>, igemmcl_kernel<7, 2>, igemmcl_kernel<7, 3>},
    {igemmcl_kernel<8, 1>, igemmcl_kernel<8, 2>, igemmcl_kernel<8, 3>},
};

static void igemmcl_dispatch(size_t M, size_t N, size_t IC, size_t IC_stride, size_t kd_start,
                             size_t kd_end, size_t KH, size_t KW, float *D, const float *W,
                             const float *input_group, const float *C, const float *bias,
                             size_t ldd, size_t ldw, size_t ldc, float clamp_min, float clamp_max,
                             size_t vl, const int *od_arr, const int *oh_arr, const int *ow_arr,
                             size_t ID, size_t IH, size_t IW, size_t SD, size_t SH, size_t SW,
                             size_t DD, size_t DH, size_t DW, size_t PD, size_t PH, size_t PW,
                             bool is_first_k, bool is_last_k)
{
    TQT_ASSUME(M > 0 && M <= 8);
    TQT_ASSUME(N == 3 * vlmax || N == 2 * vlmax || (N > 0 && N <= vlmax));
    TQT_ASSUME(vl <= vlmax);

    size_t mr_idx = M - 1;
    size_t nr_idx = (N + vlmax - 1) / vlmax - 1;

    kernel_table[mr_idx][nr_idx](IC, IC_stride, kd_start, kd_end, KH, KW, D, W, input_group, C,
                                 bias, ldd, ldw, ldc, clamp_min, clamp_max, vl, od_arr, oh_arr,
                                 ow_arr, ID, IH, IW, SD, SH, SW, DD, DH, DW, PD, PH, PW, is_first_k,
                                 is_last_k);
}

// ============================================================================
// Igemm Main Function (channel-last, transposed weight)
// ============================================================================

void tqt_run_igemm_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(
    size_t m, size_t n, size_t k, const void *input, size_t group_idx, size_t m_idx, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
    const void *bias, float clamp_min, float clamp_max, const tqt_conv_params *params)
{
    TQT_UNUSED(k_idx_a);
    TQT_UNUSED(k_idx_b);

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

    const size_t K_total = KD * KH * KW * IC_per_group;

    TQT_ASSERT(k_idx_a % IC_per_group == 0);
    TQT_ASSERT(k_idx_b % IC_per_group == 0);

    const bool is_first_k = (k_idx_a == 0);
    const bool is_last_k = (k_idx_a + k == K_total);

    // Decompose k_idx_a to kd_start for CL K-order: kd { kh { kw { ic } } }
    const size_t K_spatial = KH * KW * IC_per_group;
    const size_t kd_start = k_idx_a / K_spatial;
    const size_t kd_end = (k_idx_a + k + K_spatial - 1) / K_spatial;

    const float *input_group =
        (const float *)input + group_idx * IC_per_group;  // CL: IC is last dim

    // M tiling (spatial direction, scalar)
    for (size_t m_idx_local = 0; m_idx_local < m; m_idx_local += m_step) {
        const size_t actual_m = TQT_MIN(m - m_idx_local, m_step);
        const size_t global_m = m_idx + m_idx_local;

        // Precompute output spatial coordinates for this M tile
        int od_arr[8] = {0};
        int oh_arr[8] = {0};
        int ow_arr[8] = {0};
        for (size_t i = 0; i < actual_m; i++) {
            od_arr[i] = (int)((global_m + i) / (OH * OW));
            oh_arr[i] = (int)(((global_m + i) / OW) % OH);
            ow_arr[i] = (int)((global_m + i) % OW);
        }

        // N tiling (OC direction, vectorized)
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);

            // Weight pointer: B[n_idx, k_idx_b] in [OC, K] layout
            const float *w_ptr = (const float *)B + n_idx * ldb + k_idx_b;
            const float *bias_ptr = bias ? (const float *)bias + n_idx : nullptr;
            const float *c_ptr = C ? (const float *)C + (global_m)*ldc + n_idx : nullptr;
            float *d_ptr = (float *)D + (global_m)*ldd + n_idx;

            // N tail handling (same as GEMM 8x3vl)
            size_t size_n = 0;
            if (actual_n >= vlmax) {
                if (actual_n == 3 * vlmax) {
                    size_n = 3 * vlmax;
                } else if (actual_n >= 2 * vlmax) {
                    size_n = 2 * vlmax;
                } else {
                    size_n = vlmax;
                }
                igemmcl_dispatch(actual_m, size_n, IC_per_group, params->input_channel, kd_start,
                                 kd_end, KH, KW, d_ptr, w_ptr, input_group, c_ptr, bias_ptr, ldd,
                                 ldb, ldc, clamp_min, clamp_max, vlmax, od_arr, oh_arr, ow_arr, ID,
                                 IH, IW, SD, SH, SW, DD, DH, DW, PD, PH, PW, is_first_k, is_last_k);
            }
            size_t remain_n = actual_n - size_n;
            if (remain_n > 0) {
                const float *w_remain = w_ptr + size_n * ldb;
                const float *bias_remain = bias_ptr ? bias_ptr + size_n : nullptr;
                const float *c_remain = c_ptr ? c_ptr + size_n : nullptr;
                float *d_remain = d_ptr + size_n;
                igemmcl_dispatch(actual_m, remain_n, IC_per_group, params->input_channel, kd_start,
                                 kd_end, KH, KW, d_remain, w_remain, input_group, c_remain,
                                 bias_remain, ldd, ldb, ldc, clamp_min, clamp_max, remain_n, od_arr,
                                 oh_arr, ow_arr, ID, IH, IW, SD, SH, SW, DD, DH, DW, PD, PH, PW,
                                 is_first_k, is_last_k);
            }
        }
    }
}
