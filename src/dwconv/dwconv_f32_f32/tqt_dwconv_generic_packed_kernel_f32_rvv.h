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
// Family-level shared kernel for GENERIC packed dwconv (fp32, RVV).
//
// Both cf-packed and cl-packed micro-kernels share this implementation because
// after packing both layouts converge to NC1DHWC0. Only the pack/unpack helpers
// differ between cf and cl families.
// ============================================================================

static inline void tqt_dwconv_generic_packed_kernel_f32_rvv(const float *packed_input,
                                                            const float *packed_weight,
                                                            const float *bias, float *packed_output,
                                                            const tqt_conv_params *params,
                                                            float clamp_min, float clamp_max)
{
    TQT_ASSERT(params->groups == params->input_channel);
    TQT_ASSERT(params->output_channel % params->input_channel == 0);

    const size_t IC = params->input_channel;
    const size_t OC = params->output_channel;
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

    const size_t in_spatial = ID * IH * IW;
    const size_t out_spatial = OD * OH * OW;
    const size_t kernel_size = KD * KH * KW;

    const size_t C0 = __riscv_vsetvlmax_e32m1();
    const size_t IC1 = (IC + C0 - 1) / C0;
    const size_t tail_vl = IC - (IC1 - 1) * C0;
    const size_t OC1 = (OC / IC) * IC1;  // dm * IC1

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

        for (size_t od = 0; od < OD; ++od) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    vfloat32m1_t acc = v_bias;

                    for (size_t kd = 0; kd < KD; ++kd) {
                        const long id_in = (long)(od * SD + kd * DD) - (long)PD;
                        if (id_in < 0 || id_in >= (long)ID)
                            continue;

                        for (size_t kh = 0; kh < KH; ++kh) {
                            const long ih_in = (long)(oh * SH + kh * DH) - (long)PH;
                            if (ih_in < 0 || ih_in >= (long)IH)
                                continue;

                            for (size_t kw = 0; kw < KW; ++kw) {
                                const long iw_in = (long)(ow * SW + kw * DW) - (long)PW;
                                if (iw_in < 0 || iw_in >= (long)IW)
                                    continue;

                                const float *in_pos =
                                    in_block +
                                    ((size_t)id_in * IH * IW + (size_t)ih_in * IW + (size_t)iw_in) *
                                        vl;
                                const float *wt_pos = wt_block + (kd * KH * KW + kh * KW + kw) * vl;

                                vfloat32m1_t v_in = __riscv_vle32_v_f32m1(in_pos, vl);
                                vfloat32m1_t v_w = __riscv_vle32_v_f32m1(wt_pos, vl);
                                acc = __riscv_vfmacc_vv_f32m1(acc, v_in, v_w, vl);
                            }
                        }
                    }

                    acc = __riscv_vfmax_vf_f32m1(acc, clamp_min, vl);
                    acc = __riscv_vfmin_vf_f32m1(acc, clamp_max, vl);

                    float *out_pos =
                        out_block + ((size_t)od * OH * OW + (size_t)oh * OW + (size_t)ow) * vl;
                    __riscv_vse32_v_f32m1(out_pos, acc, vl);
                }
            }
        }
    }
}
