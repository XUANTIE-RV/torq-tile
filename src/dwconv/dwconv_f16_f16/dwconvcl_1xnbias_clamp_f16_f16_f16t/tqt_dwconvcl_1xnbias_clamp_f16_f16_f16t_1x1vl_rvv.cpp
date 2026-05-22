//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "tqt_dwconvcl_1xnbias_clamp_f16_f16_f16t_1x1vl_rvv.h"

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif

#include <riscv_vector.h>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

size_t tqt_get_output_size_dwconvcl_1xnbias_clamp_f16_f16_f16t_1x1vl_rvv(
    const tqt_conv_params *params)
{
    const size_t spatial =
        params->output_shape[0] * params->output_shape[1] * params->output_shape[2];
    return params->output_channel * spatial * sizeof(float16_t);
}

void tqt_run_dwconv_dwconvcl_1xnbias_clamp_f16_f16_f16t_1x1vl_rvv(
    const void *input, const void *weight, const void *bias, void *output,
    const tqt_conv_params *params, float16_t clamp_min, float16_t clamp_max)
{
    TQT_ASSERT(params->groups == params->input_channel);
    TQT_ASSERT(params->output_channel % params->input_channel == 0);

    const float16_t *in_base = (const float16_t *)input;
    const float16_t *wt_base = (const float16_t *)weight;
    const float16_t *bias_base = (const float16_t *)bias;
    float16_t *out_base = (float16_t *)output;

    const size_t IC = params->input_channel;
    const size_t OC = params->output_channel;
    const size_t dm = OC / IC;
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

    // CL layout: channels are innermost; vectorize along channel dimension.
    // Loop split: outer m (0..dm-1), inner ic (vectorized) so that oc = ic + m*IC,
    // and within a single vl chunk all oc values share the same input ic group
    // (continuous in cl input layout).
    for (size_t m = 0; m < dm; ++m) {
        const size_t oc_off = m * IC;

        for (size_t od = 0; od < OD; ++od) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    size_t ic = 0;
                    while (ic < IC) {
                        const size_t vl = __riscv_vsetvl_e16m1(IC - ic);
                        const size_t oc = oc_off + ic;

                        vfloat16m1_t acc;
                        if (bias_base != NULL) {
                            acc = __riscv_vle16_v_f16m1(bias_base + oc, vl);
                        } else {
                            acc = __riscv_vfmv_v_f_f16m1((float16_t)0.0f, vl);
                        }

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

                                    const float16_t *in_pos = in_base +
                                                              ((size_t)id_in * IH * IW +
                                                               (size_t)ih_in * IW + (size_t)iw_in) *
                                                                  IC +
                                                              ic;
                                    const float16_t *wt_pos =
                                        wt_base + (kd * KH * KW + kh * KW + kw) * OC + oc;

                                    vfloat16m1_t v_in = __riscv_vle16_v_f16m1(in_pos, vl);
                                    vfloat16m1_t v_w = __riscv_vle16_v_f16m1(wt_pos, vl);
                                    acc = __riscv_vfmacc_vv_f16m1(acc, v_in, v_w, vl);
                                }
                            }
                        }

                        acc = __riscv_vfmax_vf_f16m1(acc, clamp_min, vl);
                        acc = __riscv_vfmin_vf_f16m1(acc, clamp_max, vl);

                        float16_t *out_pos =
                            out_base + ((size_t)od * OH * OW + (size_t)oh * OW + (size_t)ow) * OC +
                            oc;
                        __riscv_vse16_v_f16m1(out_pos, acc, vl);

                        ic += vl;
                    }
                }
            }
        }
    }
}
