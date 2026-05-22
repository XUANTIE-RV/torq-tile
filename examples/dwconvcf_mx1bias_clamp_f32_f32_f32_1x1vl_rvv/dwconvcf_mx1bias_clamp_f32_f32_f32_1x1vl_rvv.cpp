//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"
#include "tqt_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv.h"
#include "tqt_dwconvcf_mx1bias_clamp_f32_f32_f32_interface.h"

namespace
{

const tqt_dwconvcf_mx1bias_clamp_f32_f32_f32_ukernel ukernel{
    tqt_get_output_size_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv,
    tqt_run_dwconv_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv,
};

void fill_random(float *data, size_t count, unsigned seed)
{
    srand(seed);
    for (size_t i = 0; i < count; ++i) {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

// Naive reference dwconv (NCDHW). Caller-supplied bias may be NULL.
void reference_dwconv_cf(const float *input, const float *weight, const float *bias, float *output,
                         const tqt_conv_params *p, float cmin, float cmax)
{
    const size_t IC = p->input_channel;
    const size_t OC = p->output_channel;
    const size_t ID = p->input_shape[0], IH = p->input_shape[1], IW = p->input_shape[2];
    const size_t OD = p->output_shape[0], OH = p->output_shape[1], OW = p->output_shape[2];
    const size_t KD = p->kernel_shape[0], KH = p->kernel_shape[1], KW = p->kernel_shape[2];
    for (size_t oc = 0; oc < OC; ++oc) {
        const size_t ic = oc % IC;
        for (size_t od = 0; od < OD; ++od) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    float acc = bias ? bias[oc] : 0.0f;
                    for (size_t kd = 0; kd < KD; ++kd) {
                        for (size_t kh = 0; kh < KH; ++kh) {
                            for (size_t kw = 0; kw < KW; ++kw) {
                                int id =
                                    (int)(od * p->stride[0] + kd * p->dilation[0]) - (int)p->pad[0];
                                int ih =
                                    (int)(oh * p->stride[1] + kh * p->dilation[1]) - (int)p->pad[1];
                                int iw =
                                    (int)(ow * p->stride[2] + kw * p->dilation[2]) - (int)p->pad[2];
                                if (id < 0 || id >= (int)ID || ih < 0 || ih >= (int)IH || iw < 0 ||
                                    iw >= (int)IW) {
                                    continue;
                                }
                                acc += input[ic * ID * IH * IW + (size_t)id * IH * IW +
                                             (size_t)ih * IW + (size_t)iw] *
                                       weight[oc * KD * KH * KW + kd * KH * KW + kh * KW + kw];
                            }
                        }
                    }
                    if (acc < cmin)
                        acc = cmin;
                    if (acc > cmax)
                        acc = cmax;
                    output[oc * OD * OH * OW + od * OH * OW + oh * OW + ow] = acc;
                }
            }
        }
    }
}

// Calculate cosine similarity between two vectors
float calculate_cosine_similarity(size_t size, const float *ref, const float *act)
{
    double dot_product = 0.0;
    double ref_norm = 0.0;
    double act_norm = 0.0;
    for (size_t i = 0; i < size; ++i) {
        float ref_val = ref[i];
        float act_val = act[i];
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
    // Test config: dwconv 3x3 stride=1 pad=1 with dm=1
    const size_t IC = 8;
    const size_t OC = 8;  // dm=1
    const size_t IH = 16, IW = 16;
    const size_t KH = 3, KW = 3;

    tqt_conv_params p;
    p.input_shape[0] = 1;
    p.input_shape[1] = IH;
    p.input_shape[2] = IW;
    p.kernel_shape[0] = 1;
    p.kernel_shape[1] = KH;
    p.kernel_shape[2] = KW;
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

    const size_t in_count = IC * p.input_shape[0] * p.input_shape[1] * p.input_shape[2];
    const size_t wt_count = OC * p.kernel_shape[0] * p.kernel_shape[1] * p.kernel_shape[2];
    const size_t bs_count = OC;
    const size_t out_count = OC * p.output_shape[0] * p.output_shape[1] * p.output_shape[2];

    float *input = (float *)std::malloc(in_count * sizeof(float));
    float *weight = (float *)std::malloc(wt_count * sizeof(float));
    float *bias = (float *)std::malloc(bs_count * sizeof(float));
    float *out_ref = (float *)std::malloc(out_count * sizeof(float));
    float *out_actual = (float *)std::malloc(out_count * sizeof(float));

    fill_random(input, in_count, 42);
    fill_random(weight, wt_count, 43);
    fill_random(bias, bs_count, 44);

    const float cmin = 0.0f;  // ReLU
    const float cmax = FLT_MAX;

    // Reference
    reference_dwconv_cf(input, weight, bias, out_ref, &p, cmin, cmax);

    // Micro-kernel
    const size_t out_bytes = ukernel.get_output_size(&p);
    if (out_bytes != out_count * sizeof(float)) {
        std::printf("FAIL: get_output_size mismatch (got %zu, expected %zu)\n", out_bytes,
                    out_count * sizeof(float));
        return 1;
    }
    ukernel.run_dwconv(input, weight, bias, out_actual, &p, cmin, cmax);

    const float cosine_sim = calculate_cosine_similarity(out_count, out_ref, out_actual);
    const bool passed = cosine_sim > 0.9999f;

    std::free(input);
    std::free(weight);
    std::free(bias);
    std::free(out_ref);
    std::free(out_actual);

    std::printf("TEST[dwconvcf_mx1bias_clamp_f32_f32_f32]\n");
    std::printf("- ukernel: dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv\n");
    std::printf("- Cosine Similarity: %f\n", cosine_sim);
    std::printf("- Status: %s\n", passed ? "PASSED" : "FAILED");
    return passed ? 0 : 1;
}
