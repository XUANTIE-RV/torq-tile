//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif

#include <riscv_vector.h>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

// ============================================================================
// Family-level shared kernel for SPECIALIZED packed dwconv 5x5 stride=1
// (fp32, RVV). Tile: 4x1x1vl.
//
// 5x5 has 25 weights, too many to keep fully unrolled. Strategy:
//   - Outer KW loop (5 iters)
//   - Per KW iter: load 5 weight (column) + 8 input rows (one column),
//     accumulate to 4 acc registers
//
// 4 OH outputs share 4 + KH - 1 = 8 input rows per column. Per iter the
// helper produces a 4x1 partial sum; over 5 KW iters the full 4x1 result
// accumulates.
//
// Register budget (fp32, VLEN=128, vlmax=4):
//   4 acc + 5 weight + 8 input + bias + temps ~= 18-19 / 32 (loose)
// ============================================================================

// ----- Helper: load 8x1 input tile (8 rows x 1 col) with ih padding zero-fill. -----
// Caller has already validated the iw column is in-bounds.
static inline void tqt_dwconv_5x5s1_load_input_8x1(vfloat32m1_t *r0, vfloat32m1_t *r1,
                                                   vfloat32m1_t *r2, vfloat32m1_t *r3,
                                                   vfloat32m1_t *r4, vfloat32m1_t *r5,
                                                   vfloat32m1_t *r6, vfloat32m1_t *r7,
                                                   const float *col_base_in_block, long ih_first,
                                                   long IH_l, size_t IW, size_t vl)
{
    // col_base_in_block points to in_block + iw_in * vl (constant col offset).
    // Each row r is at col_base + (ih_in + r) * IW * vl, valid iff 0 <= ih_in+r < IH.
    const size_t row_stride = IW * vl;

#define TQT_LOAD_OR_ZERO(VAR, R)                                                           \
    do {                                                                                   \
        long ih = ih_first + (R);                                                          \
        VAR = (ih >= 0 && ih < IH_l)                                                       \
                  ? __riscv_vle32_v_f32m1(col_base_in_block + (size_t)ih * row_stride, vl) \
                  : __riscv_vfmv_v_f_f32m1(0.0f, vl);                                      \
    } while (0)

    TQT_LOAD_OR_ZERO(*r0, 0);
    TQT_LOAD_OR_ZERO(*r1, 1);
    TQT_LOAD_OR_ZERO(*r2, 2);
    TQT_LOAD_OR_ZERO(*r3, 3);
    TQT_LOAD_OR_ZERO(*r4, 4);
    TQT_LOAD_OR_ZERO(*r5, 5);
    TQT_LOAD_OR_ZERO(*r6, 6);
    TQT_LOAD_OR_ZERO(*r7, 7);
#undef TQT_LOAD_OR_ZERO
}

// ----- Helper: zero out 8x1 input tile when iw column is out-of-bounds. -----
static inline void tqt_dwconv_5x5s1_zero_input_8x1(vfloat32m1_t *r0, vfloat32m1_t *r1,
                                                   vfloat32m1_t *r2, vfloat32m1_t *r3,
                                                   vfloat32m1_t *r4, vfloat32m1_t *r5,
                                                   vfloat32m1_t *r6, vfloat32m1_t *r7, size_t vl)
{
    *r0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    *r1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    *r2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    *r3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    *r4 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    *r5 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    *r6 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    *r7 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
}

static inline void tqt_dwconv_5x5s1_packed_kernel_f32_rvv(size_t oh_start, size_t oh_end,
                                                          const float *packed_input,
                                                          const float *packed_weight,
                                                          const float *bias, float *packed_output,
                                                          const tqt_conv_params *params,
                                                          float clamp_min, float clamp_max)
{
    const size_t IC = params->input_channel;
    const size_t OC = params->output_channel;
    const size_t IH = params->input_shape[1];
    const size_t IW = params->input_shape[2];
    const size_t OW = params->output_shape[2];
    const size_t PH = params->pad[1];
    const size_t PW = params->pad[2];
    // 5x5 s=1 specialization
    const size_t KH = 5;
    const size_t KW = 5;

    const size_t in_spatial = IH * IW;
    const size_t out_spatial = params->output_shape[1] * OW;
    const size_t kernel_size = KH * KW;

    const size_t C0 = __riscv_vsetvlmax_e32m1();
    const size_t IC1 = (IC + C0 - 1) / C0;
    const size_t tail_vl = IC - (IC1 - 1) * C0;
    const size_t OC1 = (OC / IC) * IC1;

    for (size_t oc1 = 0; oc1 < OC1; ++oc1) {
        const size_t ic1 = oc1 % IC1;
        const size_t m_idx = oc1 / IC1;
        const size_t requested_vl = (ic1 == IC1 - 1) ? tail_vl : C0;
        const size_t vl = __riscv_vsetvl_e32m1(requested_vl);

        const size_t channel_offset = m_idx * IC + ic1 * C0;
        const float *in_block = packed_input + ic1 * C0 * in_spatial;
        const float *wt_block = packed_weight + channel_offset * kernel_size;
        float *out_block = packed_output + channel_offset * out_spatial;
        const float *bias_block = (bias != NULL) ? (bias + channel_offset) : NULL;

        vfloat32m1_t v_bias =
            bias_block ? __riscv_vle32_v_f32m1(bias_block, vl) : __riscv_vfmv_v_f_f32m1(0.0f, vl);

        size_t oh = oh_start;

        // ========== 4-OH tile loop ==========
        for (; oh + 4 <= oh_end; oh += 4) {
            const long ih_first = (long)oh - (long)PH;  // first input row for 4-OH window
            for (size_t ow = 0; ow < OW; ++ow) {
                vfloat32m1_t acc0 = v_bias, acc1 = v_bias, acc2 = v_bias, acc3 = v_bias;

                // KW loop: process one input/weight column at a time
                for (size_t kw = 0; kw < KW; ++kw) {
                    const long iw_in = (long)ow + (long)kw - (long)PW;

                    vfloat32m1_t i0, i1, i2, i3, i4, i5, i6, i7;
                    if (iw_in >= 0 && iw_in < (long)IW) {
                        const float *col_base = in_block + (size_t)iw_in * vl;
                        tqt_dwconv_5x5s1_load_input_8x1(&i0, &i1, &i2, &i3, &i4, &i5, &i6, &i7,
                                                        col_base, ih_first, (long)IH, IW, vl);
                    } else {
                        tqt_dwconv_5x5s1_zero_input_8x1(&i0, &i1, &i2, &i3, &i4, &i5, &i6, &i7, vl);
                    }

                    // Load 5 weights for current column (kh = 0..4)
                    // Weight layout: [KH, KW, vl] -> wt_block + (kh*KW + kw) * vl
                    vfloat32m1_t w0 = __riscv_vle32_v_f32m1(wt_block + (0 * KW + kw) * vl, vl);
                    vfloat32m1_t w1 = __riscv_vle32_v_f32m1(wt_block + (1 * KW + kw) * vl, vl);
                    vfloat32m1_t w2 = __riscv_vle32_v_f32m1(wt_block + (2 * KW + kw) * vl, vl);
                    vfloat32m1_t w3 = __riscv_vle32_v_f32m1(wt_block + (3 * KW + kw) * vl, vl);
                    vfloat32m1_t w4 = __riscv_vle32_v_f32m1(wt_block + (4 * KW + kw) * vl, vl);

                    // acc[i] += sum_{kh=0..4} input_row[i+kh] * w[kh]
                    acc0 = __riscv_vfmacc_vv_f32m1(acc0, i0, w0, vl);
                    acc0 = __riscv_vfmacc_vv_f32m1(acc0, i1, w1, vl);
                    acc0 = __riscv_vfmacc_vv_f32m1(acc0, i2, w2, vl);
                    acc0 = __riscv_vfmacc_vv_f32m1(acc0, i3, w3, vl);
                    acc0 = __riscv_vfmacc_vv_f32m1(acc0, i4, w4, vl);

                    acc1 = __riscv_vfmacc_vv_f32m1(acc1, i1, w0, vl);
                    acc1 = __riscv_vfmacc_vv_f32m1(acc1, i2, w1, vl);
                    acc1 = __riscv_vfmacc_vv_f32m1(acc1, i3, w2, vl);
                    acc1 = __riscv_vfmacc_vv_f32m1(acc1, i4, w3, vl);
                    acc1 = __riscv_vfmacc_vv_f32m1(acc1, i5, w4, vl);

                    acc2 = __riscv_vfmacc_vv_f32m1(acc2, i2, w0, vl);
                    acc2 = __riscv_vfmacc_vv_f32m1(acc2, i3, w1, vl);
                    acc2 = __riscv_vfmacc_vv_f32m1(acc2, i4, w2, vl);
                    acc2 = __riscv_vfmacc_vv_f32m1(acc2, i5, w3, vl);
                    acc2 = __riscv_vfmacc_vv_f32m1(acc2, i6, w4, vl);

                    acc3 = __riscv_vfmacc_vv_f32m1(acc3, i3, w0, vl);
                    acc3 = __riscv_vfmacc_vv_f32m1(acc3, i4, w1, vl);
                    acc3 = __riscv_vfmacc_vv_f32m1(acc3, i5, w2, vl);
                    acc3 = __riscv_vfmacc_vv_f32m1(acc3, i6, w3, vl);
                    acc3 = __riscv_vfmacc_vv_f32m1(acc3, i7, w4, vl);
                }

                acc0 = __riscv_vfmax_vf_f32m1(acc0, clamp_min, vl);
                acc0 = __riscv_vfmin_vf_f32m1(acc0, clamp_max, vl);
                acc1 = __riscv_vfmax_vf_f32m1(acc1, clamp_min, vl);
                acc1 = __riscv_vfmin_vf_f32m1(acc1, clamp_max, vl);
                acc2 = __riscv_vfmax_vf_f32m1(acc2, clamp_min, vl);
                acc2 = __riscv_vfmin_vf_f32m1(acc2, clamp_max, vl);
                acc3 = __riscv_vfmax_vf_f32m1(acc3, clamp_min, vl);
                acc3 = __riscv_vfmin_vf_f32m1(acc3, clamp_max, vl);

                __riscv_vse32_v_f32m1(out_block + ((oh + 0) * OW + ow) * vl, acc0, vl);
                __riscv_vse32_v_f32m1(out_block + ((oh + 1) * OW + ow) * vl, acc1, vl);
                __riscv_vse32_v_f32m1(out_block + ((oh + 2) * OW + ow) * vl, acc2, vl);
                __riscv_vse32_v_f32m1(out_block + ((oh + 3) * OW + ow) * vl, acc3, vl);
            }
        }

        // ========== Remainder 1-OH fallback ==========
        for (; oh < oh_end; ++oh) {
            for (size_t ow = 0; ow < OW; ++ow) {
                vfloat32m1_t acc = v_bias;
                for (size_t kh = 0; kh < KH; ++kh) {
                    const long ih_in = (long)oh + (long)kh - (long)PH;
                    if (ih_in < 0 || ih_in >= (long)IH)
                        continue;
                    for (size_t kw = 0; kw < KW; ++kw) {
                        const long iw_in = (long)ow + (long)kw - (long)PW;
                        if (iw_in < 0 || iw_in >= (long)IW)
                            continue;
                        const float *in_pos = in_block + ((size_t)ih_in * IW + (size_t)iw_in) * vl;
                        const float *wt_pos = wt_block + (kh * KW + kw) * vl;
                        vfloat32m1_t v_in = __riscv_vle32_v_f32m1(in_pos, vl);
                        vfloat32m1_t v_w = __riscv_vle32_v_f32m1(wt_pos, vl);
                        acc = __riscv_vfmacc_vv_f32m1(acc, v_in, v_w, vl);
                    }
                }
                acc = __riscv_vfmax_vf_f32m1(acc, clamp_min, vl);
                acc = __riscv_vfmin_vf_f32m1(acc, clamp_max, vl);
                __riscv_vse32_v_f32m1(out_block + (oh * OW + ow) * vl, acc, vl);
            }
        }
    }
}
