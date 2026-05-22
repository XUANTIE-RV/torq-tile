//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "tqt_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv.h"

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif

#include <riscv_vector.h>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

size_t tqt_get_output_size_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv(
    const tqt_conv_params *params)
{
    const size_t spatial =
        params->output_shape[0] * params->output_shape[1] * params->output_shape[2];
    return params->output_channel * spatial * sizeof(float);
}

void tqt_run_dwconv_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv(const void *input,
                                                                 const void *weight,
                                                                 const void *bias, void *output,
                                                                 const tqt_conv_params *params,
                                                                 float clamp_min, float clamp_max)
{
    // dwconv hard constraints
    TQT_ASSERT(params->groups == params->input_channel);
    TQT_ASSERT(params->output_channel % params->input_channel == 0);

    const float *in_base = (const float *)input;
    const float *wt_base = (const float *)weight;
    const float *bias_base = (const float *)bias;  // may be NULL
    float *out_base = (float *)output;

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

    // Vectorize along OW dimension (output W is contiguous in NCHW).
    // For each (oc, od, oh): inner OW loop with vlmax-step vectorization.
    // Inputs are gathered per-(kd,kh,kw) using strided load when SW != 1.
    for (size_t oc = 0; oc < OC; ++oc) {
        const size_t ic = oc % IC;
        const float *in_ic = in_base + ic * in_spatial;
        const float *wt_oc = wt_base + oc * kernel_size;
        float *out_oc = out_base + oc * out_spatial;
        const float bias_val = (bias_base != NULL) ? bias_base[oc] : 0.0f;

        for (size_t od = 0; od < OD; ++od) {
            for (size_t oh = 0; oh < OH; ++oh) {
                size_t ow = 0;
                while (ow < OW) {
                    const size_t vl = __riscv_vsetvl_e32m1(OW - ow);
                    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(bias_val, vl);

                    for (size_t kd = 0; kd < KD; ++kd) {
                        const long id_in = (long)(od * SD + kd * DD) - (long)PD;
                        if (id_in < 0 || id_in >= (long)ID)
                            continue;

                        for (size_t kh = 0; kh < KH; ++kh) {
                            const long ih_in = (long)(oh * SH + kh * DH) - (long)PH;
                            if (ih_in < 0 || ih_in >= (long)IH)
                                continue;

                            for (size_t kw = 0; kw < KW; ++kw) {
                                const float w_val = wt_oc[kd * KH * KW + kh * KW + kw];

                                // For each of vl ow positions, iw = (ow+i)*SW + kw*DW - PW
                                // We need to bounds-check each iw and skip out-of-range.
                                // Strategy: load with strided access if all in-range; otherwise
                                // fall back to scalar loop for boundary OW chunks.
                                const long iw_first = (long)(ow * SW + kw * DW) - (long)PW;
                                const long iw_last =
                                    (long)((ow + vl - 1) * SW + kw * DW) - (long)PW;

                                if (iw_first >= 0 && iw_last < (long)IW) {
                                    // Fully in-range: use strided load
                                    const float *src = in_ic + (size_t)id_in * IH * IW +
                                                       (size_t)ih_in * IW + (size_t)iw_first;
                                    vfloat32m1_t v_in = __riscv_vlse32_v_f32m1(
                                        src, (ptrdiff_t)(SW * sizeof(float)), vl);
                                    acc = __riscv_vfmacc_vf_f32m1(acc, w_val, v_in, vl);
                                } else {
                                    // Partially out-of-range: scalar fallback for the chunk.
                                    // RVV vlmax for e32m1 cannot exceed VLEN/32 <= 32.
                                    float tmp[64];
                                    for (size_t i = 0; i < vl; ++i) {
                                        const long iw = (long)((ow + i) * SW + kw * DW) - (long)PW;
                                        tmp[i] = (iw >= 0 && iw < (long)IW)
                                                     ? in_ic[(size_t)id_in * IH * IW +
                                                             (size_t)ih_in * IW + (size_t)iw]
                                                     : 0.0f;
                                    }
                                    vfloat32m1_t v_in = __riscv_vle32_v_f32m1(tmp, vl);
                                    acc = __riscv_vfmacc_vf_f32m1(acc, w_val, v_in, vl);
                                }
                            }
                        }
                    }

                    // Clamp + store
                    acc = __riscv_vfmax_vf_f32m1(acc, clamp_min, vl);
                    acc = __riscv_vfmin_vf_f32m1(acc, clamp_max, vl);
                    __riscv_vse32_v_f32m1(out_oc + od * OH * OW + oh * OW + ow, acc, vl);

                    ow += vl;
                }
            }
        }
    }
}
