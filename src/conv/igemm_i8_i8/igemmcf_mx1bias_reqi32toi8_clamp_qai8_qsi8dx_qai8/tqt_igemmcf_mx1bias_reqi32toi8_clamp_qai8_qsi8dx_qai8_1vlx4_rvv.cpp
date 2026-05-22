//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv.h"

#include "src/common/tqt_common.h"
#include "src/conv/igemm_i8_i8/tqt_kernel_igemmcf_i8_i8_1vlx4_rvv.h"
#include "src/gemm/gemm_i8_i8/tqt_kernel_gemm_i8_i8_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/tqt_utils_i8_i8.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB;
static const size_t m_step = vlmax;
static const size_t n_step = 4;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv(void)
{
    return n_step;
}

size_t tqt_get_a_offset_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv(size_t m_idx,
                                                                                    size_t k_idx,
                                                                                    size_t lda)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * lda + k_idx) * sizeof(int8_t);
}

size_t tqt_get_c_offset_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv(size_t m_idx,
                                                                                    size_t n_idx,
                                                                                    size_t ldc)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(int32_t);
}

size_t tqt_get_bias_offset_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv(size_t m_idx)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return m_idx * sizeof(int32_t);
}

size_t tqt_get_d_offset_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv(size_t m_idx,
                                                                                    size_t n_idx,
                                                                                    size_t ldd)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(int8_t);
}

size_t tqt_get_d_size_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv(size_t m,
                                                                                  size_t n)
{
    return m * n * sizeof(int8_t);
}

// ============================================================================
// Internal: Template-based igemmcf kernel (non-packed weight, int8)
// ============================================================================

// Template parameters:
//   NR = number of scalar spatial positions (1-4)
//
// CF igemmcf i8 1vlx4: M (OC) is vectorized with 1 LMUL=4 group (vl elements),
// N (spatial) is scalar (up to 4 positions).
// Reuses GEMM i32_i8_i8 kernel functions with NR as the MR parameter (spatial = rows).
// NO K-tiling: accumulators always zero-initialized, full K processed per call.
template <size_t NR>
static void igemmcf_kernel(size_t ic_start, size_t ic_end, int8_t *D, const int8_t *W,
                           const int8_t *input_group, const int32_t *bias, size_t ldd, size_t ldw,
                           int8_t clamp_min, int8_t clamp_max, size_t vl, const int *od_arr,
                           const int *oh_arr, const int *ow_arr, size_t ID, size_t IH, size_t IW,
                           size_t KD, size_t KH, size_t KW, size_t SD, size_t SH, size_t SW,
                           size_t DD, size_t DH, size_t DW, size_t PD, size_t PH, size_t PW,
                           const int32_t *mult, const int32_t *shift, int32_t zp)
{
    static_assert(NR >= 1 && NR <= 4, "NR must be in range [1, 4]");

    // Always zero init (no K-tiling for quantized)
    tqt_init_zero_kernel_gemm_i8_i8_4x1vl_rvv<NR>(vl);

    // K-loop: for ic { for kd { for kh { for kw { ... } } } }
    // CF input layout: [IC, ID, IH, IW]
    const int8_t *w_ptr = W;
    for (size_t ic = ic_start; ic < ic_end; ic++) {
        const int8_t *b_base = input_group + ic * ID * IH * IW;
        for (size_t kd = 0; kd < KD; kd++) {
            int id[4];
            bool flag_d[4];
            for (size_t i = 0; i < NR; i++) {
                id[i] = (int)(od_arr[i] * SD + kd * DD) - (int)PD;
                flag_d[i] = (id[i] < 0 || id[i] >= (int)ID);
            }
            for (size_t kh = 0; kh < KH; kh++) {
                int ih[4];
                bool flag_h[4];
                for (size_t i = 0; i < NR; i++) {
                    ih[i] = (int)(oh_arr[i] * SH + kh * DH) - (int)PH;
                    flag_h[i] = flag_d[i] || (ih[i] < 0 || ih[i] >= (int)IH);
                }
                for (size_t kw = 0; kw < KW; kw++) {
                    int8_t b_data[4] = {0};
                    for (size_t i = 0; i < NR; i++) {
                        int iw = (int)(ow_arr[i] * SW + kw * DW) - (int)PW;
                        if (!flag_h[i] && iw >= 0 && iw < (int)IW) {
                            b_data[i] = b_base[id[i] * IH * IW + ih[i] * IW + iw];
                        }
                    }
                    const int8_t *b_ptrs[4] = {&b_data[0], &b_data[1], &b_data[2], &b_data[3]};

                    // Load weight vector (OC direction, strided for non-packed) into v0
                    tqt_load_bt_kernel_gemm_i8_i8_4x1vl_rvv(vl, w_ptr, ldw);
                    // Widening multiply-accumulate: NR spatial scalars x 1 OC vector
                    tqt_vmacc_kernel_gemm_i8_i8_4x1vl_rvv<NR>(vl, b_ptrs);
                    w_ptr += 1;
                }
            }
        }
    }

    // Post: bias + requantize + clamp + strided store
    if (bias) {
        tqt_add_1xnbias_kernel_gemm_i8_i8_4x1vl_rvv<NR>(vl, bias);
    }
    tqt_requantize_1xn_kernel_gemm_i8_i8_4x1vl_rvv<NR>(vl, mult, shift, zp);
    tqt_clamp_i8_kernel_gemm_i8_i8_4x1vl_rvv<NR>(vl, clamp_min, clamp_max);
    // CF output [OC, N]: stride_d = ldd (= OD*OH*OW)
    tqt_store_i8_strided_kernel_igemmcf_i8_i8_1vlx4_rvv<1, NR>(vl, D, ldd);
}

// Function pointer type for kernel dispatch
using igemmcf_kernel_fn_t = void (*)(size_t ic_start, size_t ic_end, int8_t *D, const int8_t *W,
                                     const int8_t *input_group, const int32_t *bias, size_t ldd,
                                     size_t ldw, int8_t clamp_min, int8_t clamp_max, size_t vl,
                                     const int *od_arr, const int *oh_arr, const int *ow_arr,
                                     size_t ID, size_t IH, size_t IW, size_t KD, size_t KH,
                                     size_t KW, size_t SD, size_t SH, size_t SW, size_t DD,
                                     size_t DH, size_t DW, size_t PD, size_t PH, size_t PW,
                                     const int32_t *mult, const int32_t *shift, int32_t zp);

// Kernel dispatch table: [NR] = [4] (MR is always 1 for i8 CF)
static constexpr igemmcf_kernel_fn_t kernel_table[4] = {
    igemmcf_kernel<1>,
    igemmcf_kernel<2>,
    igemmcf_kernel<3>,
    igemmcf_kernel<4>,
};

static void igemmcf_dispatch(size_t N, size_t ic_start, size_t ic_end, int8_t *D, const int8_t *W,
                             const int8_t *input_group, const int32_t *bias, size_t ldd, size_t ldw,
                             int8_t clamp_min, int8_t clamp_max, size_t vl, const int *od_arr,
                             const int *oh_arr, const int *ow_arr, size_t ID, size_t IH, size_t IW,
                             size_t KD, size_t KH, size_t KW, size_t SD, size_t SH, size_t SW,
                             size_t DD, size_t DH, size_t DW, size_t PD, size_t PH, size_t PW,
                             const int32_t *mult, const int32_t *shift, int32_t zp)
{
    TQT_ASSUME(N > 0 && N <= 4);
    TQT_ASSUME(vl <= vlmax);

    size_t nr_idx = N - 1;

    kernel_table[nr_idx](ic_start, ic_end, D, W, input_group, bias, ldd, ldw, clamp_min, clamp_max,
                         vl, od_arr, oh_arr, ow_arr, ID, IH, IW, KD, KH, KW, SD, SH, SW, DD, DH, DW,
                         PD, PH, PW, mult, shift, zp);
}

// ============================================================================
// Igemm Main Function (channel-first, non-packed weight, int8)
// ============================================================================

void tqt_run_igemm_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv(
    size_t m, size_t n, size_t k, const void *input, size_t group_idx, size_t n_idx, size_t k_idx_b,
    const void *A, size_t lda, size_t k_idx_a, const void *C, size_t ldc, void *D, size_t ldd,
    const void *bias, int8_t clamp_min, int8_t clamp_max, const tqt_conv_params *conv_params,
    const struct tqt_qai8_qsi8dx_qai8_params *quant_params)
{
    TQT_UNUSED(k_idx_a);
    TQT_UNUSED(k_idx_b);
    TQT_UNUSED(C);
    TQT_UNUSED(ldc);
    TQT_UNUSED(k);

    if (!conv_params || !quant_params) {
        return;
    }

    // Extract convolution parameters
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
        (const int8_t *)input + group_idx * IC_per_group * ID * IH * IW;  // CF: [IC, ID, IH, IW]

    // Pre-compute per-OC requantization parameters
    // CF: scale_a is per-channel weight array (per-OC), scale_b is scalar (activation)
    int32_t mult_arr[m];
    int32_t shift_arr[m];
    double real_scale = (double)quant_params->scale_a[0] * (double)quant_params->scale_b /
                        (double)quant_params->scale_d;
    tqt_quantize_multiplier(real_scale, &mult_arr[0], &shift_arr[0]);
    shift_arr[0] = -shift_arr[0] - 1;
    for (size_t i = 1; i < m; i++) {
        if (quant_params->quant_channel_a == 1) {
            mult_arr[i] = mult_arr[0];
            shift_arr[i] = shift_arr[0];
        } else {
            real_scale = (double)quant_params->scale_a[i] * (double)quant_params->scale_b /
                         (double)quant_params->scale_d;
            tqt_quantize_multiplier(real_scale, &mult_arr[i], &shift_arr[i]);
            shift_arr[i] = -shift_arr[i] - 1;
        }
    }
    int32_t zp = quant_params->zero_point_d;

    // N tiling (spatial direction, n_step = 4)
    for (size_t n_idx_local = 0; n_idx_local < n; n_idx_local += n_step) {
        const size_t actual_n = TQT_MIN(n - n_idx_local, n_step);
        const size_t global_n = n_idx + n_idx_local;

        // Precompute output spatial coordinates for this N tile
        int od_arr[4] = {0};
        int oh_arr[4] = {0};
        int ow_arr[4] = {0};
        for (size_t i = 0; i < actual_n; i++) {
            od_arr[i] = (int)((global_n + i) / (OH * OW));
            oh_arr[i] = (int)(((global_n + i) / OW) % OH);
            ow_arr[i] = (int)((global_n + i) % OW);
        }

        // M tiling (OC direction, vectorized, m_step = vlmax)
        for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
            const size_t actual_m = TQT_MIN(m - m_idx, m_step);

            const int8_t *w_ptr = (const int8_t *)A + m_idx * lda;
            const int32_t *bias_ptr = bias ? (const int32_t *)bias + m_idx : nullptr;
            int8_t *d_ptr = (int8_t *)D + m_idx * ldd + global_n;

            const int32_t *mult_ptr = mult_arr + m_idx;
            const int32_t *shift_ptr = shift_arr + m_idx;

            igemmcf_dispatch(actual_n, 0, IC_per_group, d_ptr, w_ptr, input_group, bias_ptr, ldd,
                             lda, clamp_min, clamp_max, actual_m, od_arr, oh_arr, ow_arr, ID, IH,
                             IW, KD, KH, KW, SD, SH, SW, DD, DH, DW, PD, PH, PW, mult_ptr,
                             shift_ptr, zp);
        }
    }
}
