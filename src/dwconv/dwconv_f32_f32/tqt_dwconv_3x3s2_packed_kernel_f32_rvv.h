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
// Family-level shared kernel for SPECIALIZED packed dwconv 3x3 stride=2
// (fp32, RVV). Tile: 2x1x1vl.
//
// stride=2: 2 OH outputs require 5 input rows. Weight is fully unrolled
// (9 vector registers). 5 rows x 3 cols = 15 input registers held across the
// OW loop for sliding-window reuse.
//
// OW sliding-window reuse (s=2): each ow step consumes a window starting 2
// input columns later, so only the old col2 -> new col0 column survives.
// Per ow iteration we shift col2->col0 then load 5 new col1 + 5 new col2
// vectors (saves 5 vle per ow iteration vs. full reload of 15).
//
// Register budget (fp32, VLEN=128, vlmax=4):
//   15 input + 9 weight + 2 acc + 1 bias (dies after acc init) = 27 / 32
// ============================================================================

// ----- Helper: load one input cell with full padding zero-fill. -----
static inline vfloat32m1_t tqt_dwconv_3x3s2_load_cell(const float *in_block, long ih_in, long iw_in,
                                                      long IH_l, long IW_l, size_t IW, size_t vl)
{
    if (ih_in < 0 || ih_in >= IH_l || iw_in < 0 || iw_in >= IW_l) {
        return __riscv_vfmv_v_f_f32m1(0.0f, vl);
    }
    return __riscv_vle32_v_f32m1(in_block + ((size_t)ih_in * IW + (size_t)iw_in) * vl, vl);
}

static inline void tqt_dwconv_3x3s2_packed_kernel_f32_rvv(size_t oh_start, size_t oh_end,
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
    // 3x3 s=2 specialization
    const size_t KH = 3;
    const size_t KW = 3;
    const size_t SH = 2;
    const size_t SW = 2;

    const size_t in_spatial = IH * IW;
    const size_t out_spatial = params->output_shape[1] * OW;
    const size_t kernel_size = KH * KW;

    const size_t C0 = __riscv_vsetvlmax_e32m1();
    const size_t IC1 = (IC + C0 - 1) / C0;
    const size_t tail_vl = IC - (IC1 - 1) * C0;
    const size_t OC1 = (OC / IC) * IC1;

    const long IH_l = (long)IH;
    const long IW_l = (long)IW;

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

        // ========== 2-OH tile loop ==========
        for (; oh + 2 <= oh_end; oh += 2) {
            // Per-row ih state (constants across the OW loop)
            const long ih0 = (long)oh * (long)SH + 0 - (long)PH;
            const long ih1 = (long)oh * (long)SH + 1 - (long)PH;
            const long ih2 = (long)oh * (long)SH + 2 - (long)PH;
            const long ih3 = (long)oh * (long)SH + 3 - (long)PH;
            const long ih4 = (long)oh * (long)SH + 4 - (long)PH;

            // 15 input registers held across the OW loop for sliding-window reuse.
            vfloat32m1_t r0c0, r0c1, r0c2;
            vfloat32m1_t r1c0, r1c1, r1c2;
            vfloat32m1_t r2c0, r2c1, r2c2;
            vfloat32m1_t r3c0, r3c1, r3c2;
            vfloat32m1_t r4c0, r4c1, r4c2;

            // ----- ow = 0: full init (load all 15 cells) -----
            {
                const long iw0 = 0 * (long)SW - (long)PW;
                const long iw1 = iw0 + 1;
                const long iw2 = iw0 + 2;
                r0c0 = tqt_dwconv_3x3s2_load_cell(in_block, ih0, iw0, IH_l, IW_l, IW, vl);
                r0c1 = tqt_dwconv_3x3s2_load_cell(in_block, ih0, iw1, IH_l, IW_l, IW, vl);
                r0c2 = tqt_dwconv_3x3s2_load_cell(in_block, ih0, iw2, IH_l, IW_l, IW, vl);
                r1c0 = tqt_dwconv_3x3s2_load_cell(in_block, ih1, iw0, IH_l, IW_l, IW, vl);
                r1c1 = tqt_dwconv_3x3s2_load_cell(in_block, ih1, iw1, IH_l, IW_l, IW, vl);
                r1c2 = tqt_dwconv_3x3s2_load_cell(in_block, ih1, iw2, IH_l, IW_l, IW, vl);
                r2c0 = tqt_dwconv_3x3s2_load_cell(in_block, ih2, iw0, IH_l, IW_l, IW, vl);
                r2c1 = tqt_dwconv_3x3s2_load_cell(in_block, ih2, iw1, IH_l, IW_l, IW, vl);
                r2c2 = tqt_dwconv_3x3s2_load_cell(in_block, ih2, iw2, IH_l, IW_l, IW, vl);
                r3c0 = tqt_dwconv_3x3s2_load_cell(in_block, ih3, iw0, IH_l, IW_l, IW, vl);
                r3c1 = tqt_dwconv_3x3s2_load_cell(in_block, ih3, iw1, IH_l, IW_l, IW, vl);
                r3c2 = tqt_dwconv_3x3s2_load_cell(in_block, ih3, iw2, IH_l, IW_l, IW, vl);
                r4c0 = tqt_dwconv_3x3s2_load_cell(in_block, ih4, iw0, IH_l, IW_l, IW, vl);
                r4c1 = tqt_dwconv_3x3s2_load_cell(in_block, ih4, iw1, IH_l, IW_l, IW, vl);
                r4c2 = tqt_dwconv_3x3s2_load_cell(in_block, ih4, iw2, IH_l, IW_l, IW, vl);
            }

            for (size_t ow = 0; ow < OW; ++ow) {
                vfloat32m1_t acc0 = v_bias;
                vfloat32m1_t acc1 = v_bias;

                // 18 fmacc: 2 acc rows x 9 weights
                // row 0 (ih0) -> acc0 (kh=0)
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, r0c0, w00, vl);
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, r0c1, w01, vl);
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, r0c2, w02, vl);
                // row 1 (ih1) -> acc0 (kh=1)
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, r1c0, w10, vl);
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, r1c1, w11, vl);
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, r1c2, w12, vl);
                // row 2 (ih2) -> acc0 (kh=2), acc1 (kh=0)
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, r2c0, w20, vl);
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, r2c1, w21, vl);
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, r2c2, w22, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, r2c0, w00, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, r2c1, w01, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, r2c2, w02, vl);
                // row 3 (ih3) -> acc1 (kh=1)
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, r3c0, w10, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, r3c1, w11, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, r3c2, w12, vl);
                // row 4 (ih4) -> acc1 (kh=2)
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, r4c0, w20, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, r4c1, w21, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, r4c2, w22, vl);

                acc0 = __riscv_vfmax_vf_f32m1(acc0, clamp_min, vl);
                acc0 = __riscv_vfmin_vf_f32m1(acc0, clamp_max, vl);
                acc1 = __riscv_vfmax_vf_f32m1(acc1, clamp_min, vl);
                acc1 = __riscv_vfmin_vf_f32m1(acc1, clamp_max, vl);

                __riscv_vse32_v_f32m1(out_block + ((oh + 0) * OW + ow) * vl, acc0, vl);
                __riscv_vse32_v_f32m1(out_block + ((oh + 1) * OW + ow) * vl, acc1, vl);

                // Sliding-window (s=2): old col2 -> new col0; load new col1, col2.
                // iw_new = (ow+1)*2 - PW. New col1 = iw_new+1, col2 = iw_new+2.
                if (ow + 1 < OW) {
                    r0c0 = r0c2;
                    r1c0 = r1c2;
                    r2c0 = r2c2;
                    r3c0 = r3c2;
                    r4c0 = r4c2;
                    const long iw_new = (long)(ow + 1) * (long)SW - (long)PW;
                    const long iw_new_1 = iw_new + 1;
                    const long iw_new_2 = iw_new + 2;
                    r0c1 = tqt_dwconv_3x3s2_load_cell(in_block, ih0, iw_new_1, IH_l, IW_l, IW, vl);
                    r0c2 = tqt_dwconv_3x3s2_load_cell(in_block, ih0, iw_new_2, IH_l, IW_l, IW, vl);
                    r1c1 = tqt_dwconv_3x3s2_load_cell(in_block, ih1, iw_new_1, IH_l, IW_l, IW, vl);
                    r1c2 = tqt_dwconv_3x3s2_load_cell(in_block, ih1, iw_new_2, IH_l, IW_l, IW, vl);
                    r2c1 = tqt_dwconv_3x3s2_load_cell(in_block, ih2, iw_new_1, IH_l, IW_l, IW, vl);
                    r2c2 = tqt_dwconv_3x3s2_load_cell(in_block, ih2, iw_new_2, IH_l, IW_l, IW, vl);
                    r3c1 = tqt_dwconv_3x3s2_load_cell(in_block, ih3, iw_new_1, IH_l, IW_l, IW, vl);
                    r3c2 = tqt_dwconv_3x3s2_load_cell(in_block, ih3, iw_new_2, IH_l, IW_l, IW, vl);
                    r4c1 = tqt_dwconv_3x3s2_load_cell(in_block, ih4, iw_new_1, IH_l, IW_l, IW, vl);
                    r4c2 = tqt_dwconv_3x3s2_load_cell(in_block, ih4, iw_new_2, IH_l, IW_l, IW, vl);
                }
            }
        }

        // ========== Remainder 1-OH fallback ==========
        for (; oh < oh_end; ++oh) {
            for (size_t ow = 0; ow < OW; ++ow) {
                const long iw_lo = (long)ow * (long)SW - (long)PW;
                vfloat32m1_t acc = v_bias;
                vfloat32m1_t v0, v1, v2;

                v0 = tqt_dwconv_3x3s2_load_cell(in_block, (long)oh * (long)SH + 0 - (long)PH,
                                                iw_lo + 0, IH_l, IW_l, IW, vl);
                v1 = tqt_dwconv_3x3s2_load_cell(in_block, (long)oh * (long)SH + 0 - (long)PH,
                                                iw_lo + 1, IH_l, IW_l, IW, vl);
                v2 = tqt_dwconv_3x3s2_load_cell(in_block, (long)oh * (long)SH + 0 - (long)PH,
                                                iw_lo + 2, IH_l, IW_l, IW, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, v0, w00, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, v1, w01, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, v2, w02, vl);

                v0 = tqt_dwconv_3x3s2_load_cell(in_block, (long)oh * (long)SH + 1 - (long)PH,
                                                iw_lo + 0, IH_l, IW_l, IW, vl);
                v1 = tqt_dwconv_3x3s2_load_cell(in_block, (long)oh * (long)SH + 1 - (long)PH,
                                                iw_lo + 1, IH_l, IW_l, IW, vl);
                v2 = tqt_dwconv_3x3s2_load_cell(in_block, (long)oh * (long)SH + 1 - (long)PH,
                                                iw_lo + 2, IH_l, IW_l, IW, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, v0, w10, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, v1, w11, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, v2, w12, vl);

                v0 = tqt_dwconv_3x3s2_load_cell(in_block, (long)oh * (long)SH + 2 - (long)PH,
                                                iw_lo + 0, IH_l, IW_l, IW, vl);
                v1 = tqt_dwconv_3x3s2_load_cell(in_block, (long)oh * (long)SH + 2 - (long)PH,
                                                iw_lo + 1, IH_l, IW_l, IW, vl);
                v2 = tqt_dwconv_3x3s2_load_cell(in_block, (long)oh * (long)SH + 2 - (long)PH,
                                                iw_lo + 2, IH_l, IW_l, IW, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, v0, w20, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, v1, w21, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, v2, w22, vl);

                acc = __riscv_vfmax_vf_f32m1(acc, clamp_min, vl);
                acc = __riscv_vfmin_vf_f32m1(acc, clamp_max, vl);
                __riscv_vse32_v_f32m1(out_block + (oh * OW + ow) * vl, acc, vl);
            }
        }
    }
}
