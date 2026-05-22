//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

// Include micro-kernel variants
#include "tqt_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv.h"
#include "tqt_convcf_mx1bias_clamp_f32_f32_f32_interface.h"

namespace
{
/// Micro-kernel interface
const tqt_convcf_mx1bias_clamp_f32_f32_f32_ukernel ukernel{
    tqt_get_m_step_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
    tqt_get_n_step_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
    tqt_get_a_offset_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
    tqt_get_c_offset_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
    tqt_get_bias_offset_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
    tqt_get_d_offset_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
    tqt_get_d_size_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
    tqt_get_b_im2col_size_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
    tqt_get_b_im2col_offset_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
    tqt_run_im2col_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
    tqt_run_conv_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv};

// Fill buffer with random values
void fill_random(float *data, size_t count, unsigned seed)
{
    srand(seed);
    for (size_t i = 0; i < count; ++i)
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

// Naive reference conv (CF layout)
void reference_conv_cf(const float *activation, const float *weight, const float *bias,
                       float *output, const tqt_conv_params *p, float cmin, float cmax)
{
    size_t IC_pg = p->input_channel / p->groups;
    size_t OC_pg = p->output_channel / p->groups;
    size_t OD = p->output_shape[0], OH = p->output_shape[1], OW = p->output_shape[2];
    size_t ID = p->input_shape[0], IH = p->input_shape[1], IW = p->input_shape[2];
    size_t KD = p->kernel_shape[0], KH = p->kernel_shape[1], KW = p->kernel_shape[2];

    for (size_t oc = 0; oc < OC_pg; ++oc) {
        for (size_t od = 0; od < OD; ++od) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    float acc = 0.0f;
                    for (size_t ic = 0; ic < IC_pg; ++ic) {
                        for (size_t kd = 0; kd < KD; ++kd) {
                            for (size_t kh = 0; kh < KH; ++kh) {
                                for (size_t kw = 0; kw < KW; ++kw) {
                                    int id = (int)(od * p->stride[0] + kd * p->dilation[0]) -
                                             (int)p->pad[0];
                                    int ih = (int)(oh * p->stride[1] + kh * p->dilation[1]) -
                                             (int)p->pad[1];
                                    int iw = (int)(ow * p->stride[2] + kw * p->dilation[2]) -
                                             (int)p->pad[2];
                                    if (id >= 0 && id < (int)ID && ih >= 0 && ih < (int)IH &&
                                        iw >= 0 && iw < (int)IW) {
                                        acc +=
                                            activation[ic * ID * IH * IW + (size_t)id * IH * IW +
                                                       (size_t)ih * IW + (size_t)iw] *
                                            weight[oc * IC_pg * KD * KH * KW + ic * KD * KH * KW +
                                                   kd * KH * KW + kh * KW + kw];
                                    }
                                }
                            }
                        }
                    }
                    if (bias)
                        acc += bias[oc];
                    acc = fmaxf(acc, cmin);
                    acc = fminf(acc, cmax);
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

    ref_norm = sqrt(ref_norm);
    act_norm = sqrt(act_norm);

    if (ref_norm > 0.0 && act_norm > 0.0) {
        return (float)(dot_product / (ref_norm * act_norm));
    }

    return 0.0f;
}

}  // namespace

int main()
{
    // 2D conv: IC=3, OC=16, IH=8, IW=8, K=3x3, pad=1
    tqt_conv_params params = {};
    params.input_channel = 3;
    params.output_channel = 16;
    params.groups = 1;
    params.input_shape[0] = 1;
    params.input_shape[1] = 8;
    params.input_shape[2] = 8;
    params.kernel_shape[0] = 1;
    params.kernel_shape[1] = 3;
    params.kernel_shape[2] = 3;
    params.stride[0] = 1;
    params.stride[1] = 1;
    params.stride[2] = 1;
    params.dilation[0] = 1;
    params.dilation[1] = 1;
    params.dilation[2] = 1;
    params.pad[0] = 0;
    params.pad[1] = 1;
    params.pad[2] = 1;
    for (int i = 0; i < 3; ++i)
        params.output_shape[i] = (params.input_shape[i] + 2 * params.pad[i] -
                                  params.dilation[i] * (params.kernel_shape[i] - 1) - 1) /
                                     params.stride[i] +
                                 1;

    size_t IC_pg = params.input_channel;
    size_t OC_pg = params.output_channel;
    size_t OD = params.output_shape[0], OH = params.output_shape[1], OW = params.output_shape[2];
    size_t ID = params.input_shape[0], IH = params.input_shape[1], IW = params.input_shape[2];
    size_t M = OC_pg;
    size_t N = OD * OH * OW;
    size_t K = IC_pg * params.kernel_shape[0] * params.kernel_shape[1] * params.kernel_shape[2];

    // Allocate the memory
    float *activation = (float *)malloc(IC_pg * ID * IH * IW * sizeof(float));
    float *weight = (float *)malloc(OC_pg * K * sizeof(float));
    float *bias = (float *)malloc(OC_pg * sizeof(float));
    float *ref_output = (float *)malloc(M * N * sizeof(float));
    float *act_output = (float *)malloc(M * N * sizeof(float));

    if (!activation || !weight || !bias || !ref_output || !act_output) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize data
    fill_random(activation, IC_pg * ID * IH * IW, 42);
    fill_random(weight, OC_pg * K, 123);
    fill_random(bias, OC_pg, 456);

    // Run reference implementation
    reference_conv_cf(activation, weight, bias, ref_output, &params, -FLT_MAX, FLT_MAX);

    // Run micro-kernel implementation using ukernel interface
    size_t im2col_size = ukernel.get_b_im2col_size(K, N);
    void *im2col_buf = malloc(im2col_size);
    ukernel.run_im2col(0, activation, im2col_buf, &params);
    ukernel.run_conv(M, N, K, weight, K, 0, im2col_buf, N, 0, NULL, N, act_output, N, bias,
                     -FLT_MAX, FLT_MAX);

    // Calculate cosine similarity
    const float cosine_sim = calculate_cosine_similarity(M * N, ref_output, act_output);
    const bool passed = cosine_sim > 0.9999f;

    printf("TEST[convcf_mx1bias_clamp_f32_f32_f32]\n");
    printf("- ukernel: convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(activation);
    free(weight);
    free(bias);
    free(ref_output);
    free(act_output);
    free(im2col_buf);

    return passed ? 0 : 1;
}
