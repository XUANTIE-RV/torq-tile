//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/common/tqt_gemm_pack_rvv.h"
#include "src/gemm/gemm_i8_i8/tqt_kernel_gemm_i8_i8_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/tqt_utils_i8_i8.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB;
static const size_t m_step = 4;
static const size_t n_step = vlmax;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(void)
{
    return n_step;
}

size_t tqt_get_b_packed_offset_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(
    size_t n_idx, size_t k_idx, size_t ldb, size_t actual_n)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return (n_idx * ldb + k_idx * TQT_MIN(actual_n, n_step)) * sizeof(int8_t);
}

size_t tqt_get_c_offset_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(size_t m_idx,
                                                                                     size_t n_idx,
                                                                                     size_t ldc)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(int32_t);
}

size_t tqt_get_bias_offset_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(
    size_t n_idx)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(int32_t);
}

size_t tqt_get_d_offset_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(size_t m_idx,
                                                                                     size_t n_idx,
                                                                                     size_t ldd)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(int8_t);
}

size_t tqt_get_d_size_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(size_t m,
                                                                                   size_t n)
{
    return m * n * sizeof(int8_t);
}

size_t tqt_get_b_packed_size_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(size_t n,
                                                                                          size_t k)
{
    return n * k * sizeof(int8_t);
}

// ============================================================================
// Weight Pack Function
// ============================================================================

void tqt_run_bt_pack_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(
    size_t n, size_t k, size_t ldw, size_t ldw_packed, size_t k_idx, const void *weight,
    void *weight_packed)
{
    // Pack weight [OC, K] -> [OC/vl, K, vl]
    tqt_gemm_pack_nxk_1x1vl_e8_rvv(n, k, ldw, ldw_packed, k_idx, weight, weight_packed);
}

// ============================================================================
// Internal: Template-based igemmcl kernel (packed weight, int8)
// ============================================================================

template <size_t MR>
static void igemmcl_packed_kernel(size_t IC, size_t IC_stride, size_t KD, size_t KH, size_t KW,
                                  int8_t *D, const int8_t *W, const int8_t *input_group,
                                  const int32_t *bias, size_t ldd, int8_t clamp_min,
                                  int8_t clamp_max, size_t vl, const int *od_arr, const int *oh_arr,
                                  const int *ow_arr, size_t ID, size_t IH, size_t IW, size_t SD,
                                  size_t SH, size_t SW, size_t DD, size_t DH, size_t DW, size_t PD,
                                  size_t PH, size_t PW, const int32_t *mult, const int32_t *shift,
                                  int32_t zp)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    // Always zero init (no K-tiling)
    tqt_init_zero_kernel_gemm_i8_i8_4x1vl_rvv<MR>(vl);

    // K-loop: for kd { for kh { for kw { for ic { ... } } } }
    const int8_t *w_ptr = W;
    for (size_t kd = 0; kd < KD; kd++) {
        int id_arr[4];
        bool flag_d[4];
        for (size_t i = 0; i < MR; i++) {
            id_arr[i] = (int)(od_arr[i] * SD + kd * DD) - (int)PD;
            flag_d[i] = (id_arr[i] < 0 || id_arr[i] >= (int)ID);
        }
        for (size_t kh = 0; kh < KH; kh++) {
            int ih_arr[4];
            bool flag_h[4];
            for (size_t i = 0; i < MR; i++) {
                ih_arr[i] = (int)(oh_arr[i] * SH + kh * DH) - (int)PH;
                flag_h[i] = flag_d[i] || (ih_arr[i] < 0 || ih_arr[i] >= (int)IH);
            }
            for (size_t kw = 0; kw < KW; kw++) {
                const int8_t *b_bases[4] = {nullptr};
                for (size_t i = 0; i < MR; i++) {
                    int iw_val = (int)(ow_arr[i] * SW + kw * DW) - (int)PW;
                    if (!flag_h[i] && iw_val >= 0 && iw_val < (int)IW) {
                        b_bases[i] = input_group + id_arr[i] * IH * IW * IC_stride +
                                     ih_arr[i] * IW * IC_stride + iw_val * IC_stride;
                    }
                }

                for (size_t ic = 0; ic < IC; ic++) {
                    int8_t a_data[4] = {0};
                    for (size_t i = 0; i < MR; i++) {
                        if (b_bases[i]) {
                            a_data[i] = b_bases[i][ic];
                        }
                    }
                    const int8_t *a_ptrs[4] = {&a_data[0], &a_data[1], &a_data[2], &a_data[3]};

                    // Packed weight: contiguous load
                    tqt_load_b_kernel_gemm_i8_i8_4x1vl_rvv(vl, w_ptr);
                    // Widening multiply-accumulate
                    tqt_vmacc_kernel_gemm_i8_i8_4x1vl_rvv<MR>(vl, a_ptrs);
                    w_ptr += vl;
                }
            }
        }
    }

    // Post: bias + requantize + clamp + store
    if (bias) {
        tqt_add_1xnbias_kernel_gemm_i8_i8_4x1vl_rvv<MR>(vl, bias);
    }
    tqt_requantize_1xn_kernel_gemm_i8_i8_4x1vl_rvv<MR>(vl, mult, shift, zp);
    tqt_clamp_i8_kernel_gemm_i8_i8_4x1vl_rvv<MR>(vl, clamp_min, clamp_max);
    tqt_store_i8_kernel_gemm_i8_i8_4x1vl_rvv<MR>(vl, D, ldd);  // contiguous for CL
}

// Function pointer type for kernel dispatch
using igemmcl_packed_kernel_fn_t = void (*)(
    size_t IC, size_t IC_stride, size_t KD, size_t KH, size_t KW, int8_t *D, const int8_t *W,
    const int8_t *input_group, const int32_t *bias, size_t ldd, int8_t clamp_min, int8_t clamp_max,
    size_t vl, const int *od_arr, const int *oh_arr, const int *ow_arr, size_t ID, size_t IH,
    size_t IW, size_t SD, size_t SH, size_t SW, size_t DD, size_t DH, size_t DW, size_t PD,
    size_t PH, size_t PW, const int32_t *mult, const int32_t *shift, int32_t zp);

// Kernel dispatch table: [MR] = [4]
static constexpr igemmcl_packed_kernel_fn_t kernel_table[4] = {
    igemmcl_packed_kernel<1>,
    igemmcl_packed_kernel<2>,
    igemmcl_packed_kernel<3>,
    igemmcl_packed_kernel<4>,
};

static void igemmcl_packed_dispatch(
    size_t M, size_t IC, size_t IC_stride, size_t KD, size_t KH, size_t KW, int8_t *D,
    const int8_t *W, const int8_t *input_group, const int32_t *bias, size_t ldd, int8_t clamp_min,
    int8_t clamp_max, size_t vl, const int *od_arr, const int *oh_arr, const int *ow_arr, size_t ID,
    size_t IH, size_t IW, size_t SD, size_t SH, size_t SW, size_t DD, size_t DH, size_t DW,
    size_t PD, size_t PH, size_t PW, const int32_t *mult, const int32_t *shift, int32_t zp)
{
    TQT_ASSUME(M > 0 && M <= 4);
    TQT_ASSUME(vl <= vlmax);

    size_t mr_idx = M - 1;

    kernel_table[mr_idx](IC, IC_stride, KD, KH, KW, D, W, input_group, bias, ldd, clamp_min,
                         clamp_max, vl, od_arr, oh_arr, ow_arr, ID, IH, IW, SD, SH, SW, DD, DH, DW,
                         PD, PH, PW, mult, shift, zp);
}

// ============================================================================
// Igemm Main Function (channel-last, packed weight, int8)
// ============================================================================

void tqt_run_igemm_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *input, size_t group_idx, size_t m_idx, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, int8_t clamp_min, int8_t clamp_max,
    const tqt_conv_params *conv_params, const struct tqt_qai8_qai8_qsi8cx_params *quant_params)
{
    TQT_UNUSED(k_idx_a);
    TQT_UNUSED(k_idx_b);
    TQT_UNUSED(C);
    TQT_UNUSED(ldc);
    TQT_UNUSED(k);

    if (!conv_params || !quant_params) {
        return;
    }

    const size_t IC_per_group = conv_params->input_channel / conv_params->groups;
    const size_t ID = conv_params->input_shape[0];
    const size_t IH = conv_params->input_shape[1];
    const size_t IW = conv_params->input_shape[2];
    const size_t OD = conv_params->output_shape[0];
    const size_t OH = conv_params->output_shape[1];
    const size_t OW = conv_params->output_shape[2];
    const size_t KD = conv_params->kernel_shape[0];
    const size_t KH = conv_params->kernel_shape[1];
    const size_t KW = conv_params->kernel_shape[2];
    const size_t SD = conv_params->stride[0];
    const size_t SH = conv_params->stride[1];
    const size_t SW = conv_params->stride[2];
    const size_t DD = conv_params->dilation[0];
    const size_t DH = conv_params->dilation[1];
    const size_t DW = conv_params->dilation[2];
    const size_t PD = conv_params->pad[0];
    const size_t PH = conv_params->pad[1];
    const size_t PW = conv_params->pad[2];

    const int8_t *input_group =
        (const int8_t *)input + group_idx * IC_per_group;  // CL: IC is last dim

    // Pre-compute per-OC requantization parameters
    int32_t mult_arr[n];
    int32_t shift_arr[n];
    double real_scale = (double)quant_params->scale_a * (double)quant_params->scale_b[0] /
                        (double)quant_params->scale_d;
    tqt_quantize_multiplier(real_scale, &mult_arr[0], &shift_arr[0]);
    shift_arr[0] = -shift_arr[0] - 1;
    for (size_t i = 1; i < n; i++) {
        if (quant_params->quant_channel_b == 1) {
            mult_arr[i] = mult_arr[0];
            shift_arr[i] = shift_arr[0];
        } else {
            real_scale = (double)quant_params->scale_a * (double)quant_params->scale_b[i] /
                         (double)quant_params->scale_d;
            tqt_quantize_multiplier(real_scale, &mult_arr[i], &shift_arr[i]);
            shift_arr[i] = -shift_arr[i] - 1;
        }
    }
    int32_t zp = quant_params->zero_point_d;

    // M tiling (spatial direction)
    for (size_t m_idx_local = 0; m_idx_local < m; m_idx_local += m_step) {
        const size_t actual_m = TQT_MIN(m - m_idx_local, m_step);
        const size_t global_m = m_idx + m_idx_local;

        int od_arr[4] = {0};
        int oh_arr[4] = {0};
        int ow_arr[4] = {0};
        for (size_t i = 0; i < actual_m; i++) {
            od_arr[i] = (int)((global_m + i) / (OH * OW));
            oh_arr[i] = (int)(((global_m + i) / OW) % OH);
            ow_arr[i] = (int)((global_m + i) % OW);
        }

        // N tiling (OC direction)
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);

            // Packed weight pointer
            const int8_t *w_ptr;
            if (actual_n == n_step || k_idx_b == 0) {
                w_ptr = (const int8_t *)B_packed + n_idx * ldb;
            } else {
                w_ptr = (const int8_t *)B_packed + n_idx * ldb - k_idx_b * (n_step - actual_n);
            }

            const int32_t *bias_ptr = bias ? (const int32_t *)bias + n_idx : nullptr;
            int8_t *d_ptr = (int8_t *)D + global_m * ldd + n_idx;

            const int32_t *mult_ptr = mult_arr + n_idx;
            const int32_t *shift_ptr = shift_arr + n_idx;

            igemmcl_packed_dispatch(actual_m, IC_per_group, conv_params->input_channel, KD, KH, KW,
                                    d_ptr, w_ptr, input_group, bias_ptr, ldd, clamp_min, clamp_max,
                                    actual_n, od_arr, oh_arr, ow_arr, ID, IH, IW, SD, SH, SW, DD,
                                    DH, DW, PD, PH, PW, mult_ptr, shift_ptr, zp);
        }
    }
}
