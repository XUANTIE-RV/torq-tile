//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "tqt_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_4x1x1vl_rvv.h"

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif

#include <riscv_vector.h>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"
#include "src/common/tqt_dwconv_pack_rvv.h"

// ----- Specialized constants -----
static const size_t kh = 3;
static const size_t kw = 3;
static const size_t sh = 1;
static const size_t sw = 1;
static const size_t oh_step = 4;

// ----- Size queries -----

size_t tqt_get_input_size_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_4x1x1vl_rvv(
    const tqt_conv_params *params)
{
    const size_t spatial = params->input_shape[0] * params->input_shape[1] * params->input_shape[2];
    return spatial * params->input_channel * sizeof(float);
}

size_t tqt_get_output_size_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_4x1x1vl_rvv(
    const tqt_conv_params *params)
{
    const size_t spatial =
        params->output_shape[0] * params->output_shape[1] * params->output_shape[2];
    return spatial * params->output_channel * sizeof(float);
}

size_t tqt_get_packed_weight_size_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_4x1x1vl_rvv(
    const tqt_conv_params *params)
{
    const size_t kernel_size =
        params->kernel_shape[0] * params->kernel_shape[1] * params->kernel_shape[2];
    return tqt_dwconv_packed_weight_bytes_f32_rvv(params->output_channel, kernel_size);
}

size_t tqt_get_oh_step_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_4x1x1vl_rvv(void)
{
    return oh_step;
}

// ----- Pack (cl-family weight only) -----

void tqt_run_pack_weight_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_4x1x1vl_rvv(
    const void *weight, void *packed_weight, const tqt_conv_params *params)
{
    tqt_dwconv_pack_weight_cl_f32_rvv((const float *)weight, (float *)packed_weight, params);
}

// ============================================================================
// Inner kernel (NHWC direct-read variant, derived from
// tqt_dwconv_3x3s1_packed_kernel_f32_rvv but with input/output stride along
// spatial = IC / OC instead of vl).
//
// Optimization strategy (mirrors the packed kernel):
//   - 4 acc registers (4 OH outputs sharing weight loads)
//   - 9 weight registers fully unrolled and held across all (oh_block, ow)
//   - 6 rows x 2 cols = 12 input registers held across the OW loop
//   - OW sliding-window reuse (s=1): each row streams its new col2 vector, then
//     shifts col1->col0 and col2->col1
//   - bias preloaded once per channel block
//   - Remainder OH < 4 falls back to per-row generic-style accumulation
// ============================================================================

// Helper: load one input cell from NHWC tensor with full padding zero-fill.
// `in_channel_base` already points to (ih=0, iw=0, ic=ic_offset).
static inline vfloat32m1_t tqt_dwconvcl_3x3s1_nhwc_load_cell(const float *in_channel_base,
                                                             long ih_in, long iw_in, long IH_l,
                                                             long IW_l, size_t IW, size_t IC,
                                                             size_t vl)
{
    if (ih_in < 0 || ih_in >= IH_l || iw_in < 0 || iw_in >= IW_l) {
        return __riscv_vfmv_v_f_f32m1(0.0f, vl);
    }
    return __riscv_vle32_v_f32m1(in_channel_base + ((size_t)ih_in * IW + (size_t)iw_in) * IC, vl);
}

// ----- Compute -----

void tqt_run_dwconv_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_4x1x1vl_rvv(
    size_t oh_start, size_t oh_end, const void *input, const void *packed_weight, const void *bias,
    void *output, const tqt_conv_params *params, float clamp_min, float clamp_max)
{
    // Specialization assertions
    TQT_ASSERT(params->kernel_shape[0] == 1 && params->input_shape[0] == 1 &&
               params->output_shape[0] == 1);
    TQT_ASSERT(params->kernel_shape[1] == kh && params->kernel_shape[2] == kw);
    TQT_ASSERT(params->stride[1] == sh && params->stride[2] == sw);
    TQT_ASSERT(params->dilation[0] == 1 && params->dilation[1] == 1 && params->dilation[2] == 1);

    const size_t IC = params->input_channel;
    const size_t OC = params->output_channel;
    const size_t IH = params->input_shape[1];
    const size_t IW = params->input_shape[2];
    const size_t OW = params->output_shape[2];
    const size_t PH = params->pad[1];
    const size_t PW = params->pad[2];
    const size_t kernel_size = kh * kw;

    const size_t C0 = __riscv_vsetvlmax_e32m1();
    const size_t IC1 = (IC + C0 - 1) / C0;
    const size_t tail_vl = IC - (IC1 - 1) * C0;
    const size_t OC1 = (OC / IC) * IC1;

    const long IH_l = (long)IH;
    const long IW_l = (long)IW;

    const float *input_base = (const float *)input;
    const float *weight_base = (const float *)packed_weight;
    const float *bias_base = (const float *)bias;
    float *output_base = (float *)output;

    for (size_t oc1 = 0; oc1 < OC1; ++oc1) {
        const size_t ic1 = oc1 % IC1;
        const size_t m_idx = oc1 / IC1;
        const size_t requested_vl = (ic1 == IC1 - 1) ? tail_vl : C0;
        const size_t vl = __riscv_vsetvl_e32m1(requested_vl);

        const size_t ic_offset = ic1 * C0;
        const size_t oc_offset = m_idx * IC + ic1 * C0;

        const float *in_channel_base = input_base + ic_offset;  // NHWC (ih=0, iw=0, ic=ic_offset)
        const float *wt_block = weight_base + oc_offset * kernel_size;
        float *out_channel_base = output_base + oc_offset;  // NHWC (oh=0, ow=0, oc=oc_offset)
        const float *bias_block = (bias_base != nullptr) ? (bias_base + oc_offset) : nullptr;

        // Preload bias and 9 weight vectors held across all (oh, ow) iterations.
        vfloat32m1_t v_bias =
            bias_block ? __riscv_vle32_v_f32m1(bias_block, vl) : __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t w00 = __riscv_vle32_v_f32m1(wt_block + 0 * vl, vl);
        vfloat32m1_t w01 = __riscv_vle32_v_f32m1(wt_block + 1 * vl, vl);
        vfloat32m1_t w02 = __riscv_vle32_v_f32m1(wt_block + 2 * vl, vl);
        vfloat32m1_t w10 = __riscv_vle32_v_f32m1(wt_block + 3 * vl, vl);
        vfloat32m1_t w11 = __riscv_vle32_v_f32m1(wt_block + 4 * vl, vl);
        vfloat32m1_t w12 = __riscv_vle32_v_f32m1(wt_block + 5 * vl, vl);
        vfloat32m1_t w20 = __riscv_vle32_v_f32m1(wt_block + 6 * vl, vl);
        vfloat32m1_t w21 = __riscv_vle32_v_f32m1(wt_block + 7 * vl, vl);
        vfloat32m1_t w22 = __riscv_vle32_v_f32m1(wt_block + 8 * vl, vl);

        size_t oh = oh_start;

        // ========== 4-OH tile loop ==========
        for (; oh + 4 <= oh_end; oh += 4) {
            // Per-row ih state (constants across the OW loop)
            const long ih0 = (long)oh + 0 - (long)PH;
            const long ih1 = (long)oh + 1 - (long)PH;
            const long ih2 = (long)oh + 2 - (long)PH;
            const long ih3 = (long)oh + 3 - (long)PH;
            const long ih4 = (long)oh + 4 - (long)PH;
            const long ih5 = (long)oh + 5 - (long)PH;

            // Keep two columns per row across OW iterations. The third column is
            // streamed one row at a time to leave scheduling headroom.
            vfloat32m1_t r0c0, r0c1;
            vfloat32m1_t r1c0, r1c1;
            vfloat32m1_t r2c0, r2c1;
            vfloat32m1_t r3c0, r3c1;
            vfloat32m1_t r4c0, r4c1;
            vfloat32m1_t r5c0, r5c1;

            // ----- ow = 0: initialize the two persistent columns -----
            {
                const long iw0 = 0 - (long)PW;
                const long iw1 = iw0 + 1;
                r0c0 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, ih0, iw0, IH_l, IW_l, IW,
                                                         IC, vl);
                r0c1 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, ih0, iw1, IH_l, IW_l, IW,
                                                         IC, vl);
                r1c0 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, ih1, iw0, IH_l, IW_l, IW,
                                                         IC, vl);
                r1c1 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, ih1, iw1, IH_l, IW_l, IW,
                                                         IC, vl);
                r2c0 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, ih2, iw0, IH_l, IW_l, IW,
                                                         IC, vl);
                r2c1 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, ih2, iw1, IH_l, IW_l, IW,
                                                         IC, vl);
                r3c0 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, ih3, iw0, IH_l, IW_l, IW,
                                                         IC, vl);
                r3c1 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, ih3, iw1, IH_l, IW_l, IW,
                                                         IC, vl);
                r4c0 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, ih4, iw0, IH_l, IW_l, IW,
                                                         IC, vl);
                r4c1 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, ih4, iw1, IH_l, IW_l, IW,
                                                         IC, vl);
                r5c0 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, ih5, iw0, IH_l, IW_l, IW,
                                                         IC, vl);
                r5c1 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, ih5, iw1, IH_l, IW_l, IW,
                                                         IC, vl);
            }

            for (size_t ow = 0; ow < OW; ++ow) {
                vfloat32m1_t acc0 = v_bias;
                vfloat32m1_t acc1 = v_bias;
                vfloat32m1_t acc2 = v_bias;
                vfloat32m1_t acc3 = v_bias;
                const long iw2 = (long)ow + 2 - (long)PW;

                // 36 fmacc: 4 acc rows x 9 weights
                // row 0 (ih0) -> acc0 (kh=0)
                vfloat32m1_t r0c2 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, ih0, iw2,
                                                                      IH_l, IW_l, IW, IC, vl);
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, r0c0, w00, vl);
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, r0c1, w01, vl);
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, r0c2, w02, vl);
                r0c0 = r0c1;
                r0c1 = r0c2;
                // row 1 (ih1) -> acc0 (kh=1), acc1 (kh=0)
                vfloat32m1_t r1c2 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, ih1, iw2,
                                                                      IH_l, IW_l, IW, IC, vl);
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, r1c0, w10, vl);
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, r1c1, w11, vl);
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, r1c2, w12, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, r1c0, w00, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, r1c1, w01, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, r1c2, w02, vl);
                r1c0 = r1c1;
                r1c1 = r1c2;
                // row 2 (ih2) -> acc0 (kh=2), acc1 (kh=1), acc2 (kh=0)
                vfloat32m1_t r2c2 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, ih2, iw2,
                                                                      IH_l, IW_l, IW, IC, vl);
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, r2c0, w20, vl);
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, r2c1, w21, vl);
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, r2c2, w22, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, r2c0, w10, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, r2c1, w11, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, r2c2, w12, vl);
                acc2 = __riscv_vfmacc_vv_f32m1(acc2, r2c0, w00, vl);
                acc2 = __riscv_vfmacc_vv_f32m1(acc2, r2c1, w01, vl);
                acc2 = __riscv_vfmacc_vv_f32m1(acc2, r2c2, w02, vl);
                r2c0 = r2c1;
                r2c1 = r2c2;
                // row 3 (ih3) -> acc1 (kh=2), acc2 (kh=1), acc3 (kh=0)
                vfloat32m1_t r3c2 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, ih3, iw2,
                                                                      IH_l, IW_l, IW, IC, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, r3c0, w20, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, r3c1, w21, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, r3c2, w22, vl);
                acc2 = __riscv_vfmacc_vv_f32m1(acc2, r3c0, w10, vl);
                acc2 = __riscv_vfmacc_vv_f32m1(acc2, r3c1, w11, vl);
                acc2 = __riscv_vfmacc_vv_f32m1(acc2, r3c2, w12, vl);
                acc3 = __riscv_vfmacc_vv_f32m1(acc3, r3c0, w00, vl);
                acc3 = __riscv_vfmacc_vv_f32m1(acc3, r3c1, w01, vl);
                acc3 = __riscv_vfmacc_vv_f32m1(acc3, r3c2, w02, vl);
                r3c0 = r3c1;
                r3c1 = r3c2;
                // row 4 (ih4) -> acc2 (kh=2), acc3 (kh=1)
                vfloat32m1_t r4c2 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, ih4, iw2,
                                                                      IH_l, IW_l, IW, IC, vl);
                acc2 = __riscv_vfmacc_vv_f32m1(acc2, r4c0, w20, vl);
                acc2 = __riscv_vfmacc_vv_f32m1(acc2, r4c1, w21, vl);
                acc2 = __riscv_vfmacc_vv_f32m1(acc2, r4c2, w22, vl);
                acc3 = __riscv_vfmacc_vv_f32m1(acc3, r4c0, w10, vl);
                acc3 = __riscv_vfmacc_vv_f32m1(acc3, r4c1, w11, vl);
                acc3 = __riscv_vfmacc_vv_f32m1(acc3, r4c2, w12, vl);
                r4c0 = r4c1;
                r4c1 = r4c2;
                // row 5 (ih5) -> acc3 (kh=2)
                vfloat32m1_t r5c2 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, ih5, iw2,
                                                                      IH_l, IW_l, IW, IC, vl);
                acc3 = __riscv_vfmacc_vv_f32m1(acc3, r5c0, w20, vl);
                acc3 = __riscv_vfmacc_vv_f32m1(acc3, r5c1, w21, vl);
                acc3 = __riscv_vfmacc_vv_f32m1(acc3, r5c2, w22, vl);
                r5c0 = r5c1;
                r5c1 = r5c2;

                acc0 = __riscv_vfmax_vf_f32m1(acc0, clamp_min, vl);
                acc0 = __riscv_vfmin_vf_f32m1(acc0, clamp_max, vl);
                acc1 = __riscv_vfmax_vf_f32m1(acc1, clamp_min, vl);
                acc1 = __riscv_vfmin_vf_f32m1(acc1, clamp_max, vl);
                acc2 = __riscv_vfmax_vf_f32m1(acc2, clamp_min, vl);
                acc2 = __riscv_vfmin_vf_f32m1(acc2, clamp_max, vl);
                acc3 = __riscv_vfmax_vf_f32m1(acc3, clamp_min, vl);
                acc3 = __riscv_vfmin_vf_f32m1(acc3, clamp_max, vl);

                __riscv_vse32_v_f32m1(out_channel_base + ((oh + 0) * OW + ow) * OC, acc0, vl);
                __riscv_vse32_v_f32m1(out_channel_base + ((oh + 1) * OW + ow) * OC, acc1, vl);
                __riscv_vse32_v_f32m1(out_channel_base + ((oh + 2) * OW + ow) * OC, acc2, vl);
                __riscv_vse32_v_f32m1(out_channel_base + ((oh + 3) * OW + ow) * OC, acc3, vl);
            }
        }

        // ========== Remainder 1-OH fallback ==========
        for (; oh < oh_end; ++oh) {
            for (size_t ow = 0; ow < OW; ++ow) {
                const long iw_lo = (long)ow - (long)PW;
                vfloat32m1_t acc = v_bias;
                vfloat32m1_t v0, v1, v2;

                v0 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, (long)oh + 0 - (long)PH,
                                                       iw_lo + 0, IH_l, IW_l, IW, IC, vl);
                v1 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, (long)oh + 0 - (long)PH,
                                                       iw_lo + 1, IH_l, IW_l, IW, IC, vl);
                v2 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, (long)oh + 0 - (long)PH,
                                                       iw_lo + 2, IH_l, IW_l, IW, IC, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, v0, w00, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, v1, w01, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, v2, w02, vl);

                v0 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, (long)oh + 1 - (long)PH,
                                                       iw_lo + 0, IH_l, IW_l, IW, IC, vl);
                v1 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, (long)oh + 1 - (long)PH,
                                                       iw_lo + 1, IH_l, IW_l, IW, IC, vl);
                v2 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, (long)oh + 1 - (long)PH,
                                                       iw_lo + 2, IH_l, IW_l, IW, IC, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, v0, w10, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, v1, w11, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, v2, w12, vl);

                v0 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, (long)oh + 2 - (long)PH,
                                                       iw_lo + 0, IH_l, IW_l, IW, IC, vl);
                v1 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, (long)oh + 2 - (long)PH,
                                                       iw_lo + 1, IH_l, IW_l, IW, IC, vl);
                v2 = tqt_dwconvcl_3x3s1_nhwc_load_cell(in_channel_base, (long)oh + 2 - (long)PH,
                                                       iw_lo + 2, IH_l, IW_l, IW, IC, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, v0, w20, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, v1, w21, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, v2, w22, vl);

                acc = __riscv_vfmax_vf_f32m1(acc, clamp_min, vl);
                acc = __riscv_vfmin_vf_f32m1(acc, clamp_max, vl);
                __riscv_vse32_v_f32m1(out_channel_base + (oh * OW + ow) * OC, acc, vl);
            }
        }
    }
}
