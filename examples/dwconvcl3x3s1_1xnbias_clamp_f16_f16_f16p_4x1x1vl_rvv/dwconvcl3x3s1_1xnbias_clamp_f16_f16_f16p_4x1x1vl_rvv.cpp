//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"
#include "tqt_dwconvcl3x3s1_1xnbias_clamp_f16_f16_f16p_4x1x1vl_rvv.h"
#include "tqt_dwconvcl3x3s1_1xnbias_clamp_f16_f16_f16p_interface.h"

namespace
{
const tqt_dwconvcl3x3s1_1xnbias_clamp_f16_f16_f16p_ukernel ukernel{
    tqt_get_input_size_dwconvcl3x3s1_1xnbias_clamp_f16_f16_f16p_4x1x1vl_rvv,
    tqt_get_output_size_dwconvcl3x3s1_1xnbias_clamp_f16_f16_f16p_4x1x1vl_rvv,
    tqt_get_packed_weight_size_dwconvcl3x3s1_1xnbias_clamp_f16_f16_f16p_4x1x1vl_rvv,
    tqt_get_oh_step_dwconvcl3x3s1_1xnbias_clamp_f16_f16_f16p_4x1x1vl_rvv,
    tqt_run_pack_weight_dwconvcl3x3s1_1xnbias_clamp_f16_f16_f16p_4x1x1vl_rvv,
    tqt_run_dwconv_dwconvcl3x3s1_1xnbias_clamp_f16_f16_f16p_4x1x1vl_rvv,
};

void fill_random(float16_t *data, size_t count, unsigned seed)
{
    srand(seed);
    for (size_t i = 0; i < count; ++i)
        data[i] = (float16_t)(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
}

void reference_dwconv_cl(const float16_t *input, const float16_t *weight, const float16_t *bias,
                         float16_t *output, const tqt_conv_params *p, float16_t cmin,
                         float16_t cmax)
{
    const size_t IC = p->input_channel, OC = p->output_channel;
    const size_t IH = p->input_shape[1], IW = p->input_shape[2];
    const size_t OH = p->output_shape[1], OW = p->output_shape[2];
    const size_t KH = p->kernel_shape[1], KW = p->kernel_shape[2];
    for (size_t oc = 0; oc < OC; ++oc) {
        const size_t ic = oc / (OC / IC);
        for (size_t oh = 0; oh < OH; ++oh)
            for (size_t ow = 0; ow < OW; ++ow) {
                float acc = bias ? static_cast<float>(bias[oc]) : 0.0f;
                for (size_t kh = 0; kh < KH; ++kh)
                    for (size_t kw = 0; kw < KW; ++kw) {
                        int ih = (int)(oh * p->stride[1] + kh) - (int)p->pad[1];
                        int iw = (int)(ow * p->stride[2] + kw) - (int)p->pad[2];
                        if (ih < 0 || ih >= (int)IH || iw < 0 || iw >= (int)IW)
                            continue;
                        // input [IH, IW, IC]
                        acc += static_cast<float>(input[((size_t)ih * IW + (size_t)iw) * IC + ic]) *
                               // weight [KH, KW, 1, OC]
                               static_cast<float>(weight[(kh * KW + kw) * OC + oc]);
                    }
                acc = fmaxf(acc, static_cast<float>(cmin));
                acc = fminf(acc, static_cast<float>(cmax));
                // output [OH, OW, OC]
                output[(oh * OW + ow) * OC + oc] = static_cast<float16_t>(acc);
            }
    }
}

// Calculate cosine similarity between two vectors
float calculate_cosine_similarity(size_t size, const float16_t *ref, const float16_t *act)
{
    double dot_product = 0.0;
    double ref_norm = 0.0;
    double act_norm = 0.0;
    for (size_t i = 0; i < size; ++i) {
        float ref_val = static_cast<float>(ref[i]);
        float act_val = static_cast<float>(act[i]);
        dot_product += ref_val * act_val;
        ref_norm += ref_val * ref_val;
        act_norm += act_val * act_val;
    }
    ref_norm = std::sqrt(ref_norm);
    act_norm = std::sqrt(act_norm);
    if (ref_norm == 0.0 || act_norm == 0.0)
        return (ref_norm == act_norm) ? 1.0f : 0.0f;
    return (float)(dot_product / (ref_norm * act_norm));
}

}  // namespace

int main(void)
{
    const size_t IC = 32, OC = 32, IH = 28, IW = 28;
    tqt_conv_params p;
    p.input_shape[0] = 1;
    p.input_shape[1] = IH;
    p.input_shape[2] = IW;
    p.kernel_shape[0] = 1;
    p.kernel_shape[1] = 3;
    p.kernel_shape[2] = 3;
    p.stride[0] = 1;
    p.stride[1] = 1;
    p.stride[2] = 1;
    p.dilation[0] = 1;
    p.dilation[1] = 1;
    p.dilation[2] = 1;
    p.pad[0] = 0;
    p.pad[1] = 1;
    p.pad[2] = 1;
    p.input_channel = IC;
    p.output_channel = OC;
    p.groups = IC;
    for (int i = 0; i < 3; ++i) {
        p.output_shape[i] =
            (p.input_shape[i] + 2 * p.pad[i] - p.dilation[i] * (p.kernel_shape[i] - 1) - 1) /
                p.stride[i] +
            1;
    }
    const size_t in_count = IC * IH * IW;
    const size_t wt_count = OC * 9;
    const size_t out_count = OC * p.output_shape[1] * p.output_shape[2];

    float16_t *input = (float16_t *)std::malloc(in_count * sizeof(float16_t));
    float16_t *weight = (float16_t *)std::malloc(wt_count * sizeof(float16_t));
    float16_t *bias = (float16_t *)std::malloc(OC * sizeof(float16_t));
    float16_t *out_ref = (float16_t *)std::malloc(out_count * sizeof(float16_t));
    float16_t *out_actual = (float16_t *)std::malloc(out_count * sizeof(float16_t));
    fill_random(input, in_count, 42);
    fill_random(weight, wt_count, 43);
    fill_random(bias, OC, 44);

    const float16_t cmin = (float16_t)0.0f;
    const float16_t cmax = (float16_t)(60000.0f);
    reference_dwconv_cl(input, weight, bias, out_ref, &p, cmin, cmax);

    void *packed_wt = std::malloc(ukernel.get_packed_weight_size(&p));
    ukernel.run_pack_weight(weight, packed_wt, &p);
    const size_t oh_step = ukernel.get_oh_step();
    const size_t OH = p.output_shape[1];
    for (size_t oh = 0; oh < OH; oh += oh_step) {
        const size_t oh_end = (oh + oh_step < OH) ? (oh + oh_step) : OH;
        ukernel.run_dwconv(oh, oh_end, input, packed_wt, bias, out_actual, &p, cmin, cmax);
    }

#ifdef TQT_DEBUG
    for (size_t dbg_i = 0; dbg_i < out_count; ++dbg_i) {
        float dbg_ref = (float)out_ref[dbg_i];
        float dbg_act = (float)out_actual[dbg_i];
        float dbg_diff = fabsf(dbg_ref - dbg_act);
        if (dbg_diff > 1.0f)
            printf("[%zu]: ref=%.6f vs act=%.6f (diff=%.6f) [FAIL]\n", dbg_i, dbg_ref, dbg_act,
                   dbg_diff);
        else
            printf("[%zu]: ref=%.6f vs act=%.6f (diff=%.6f)\n", dbg_i, dbg_ref, dbg_act, dbg_diff);
    }
#endif

    const float cosine_sim = calculate_cosine_similarity(out_count, out_ref, out_actual);
    const bool passed = cosine_sim > 0.9999f;
    std::free(input);
    std::free(weight);
    std::free(bias);
    std::free(out_ref);
    std::free(out_actual);
    std::free(packed_wt);
    std::printf("TEST[dwconvcl3x3s1_1xnbias_clamp_f16_f16_f16p]\n");
    std::printf("- ukernel: dwconvcl3x3s1_1xnbias_clamp_f16_f16_f16p_4x1x1vl_rvv\n");
    std::printf("- Cosine Similarity: %f\n", cosine_sim);
    std::printf("- Status: %s\n", passed ? "PASSED" : "FAILED");
    return passed ? 0 : 1;
}
