//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

// Include micro-kernel variants
#include "tqt_igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv.h"
#include "tqt_igemmcl_1xnbias_clamp_f16_f16_f16p_interface.h"

namespace
{
/// Micro-kernel interface
const tqt_igemmcl_1xnbias_clamp_f16_f16_f16p_ukernel ukernel{
    tqt_get_m_step_igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_get_n_step_igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_get_b_packed_offset_igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_get_c_offset_igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_get_bias_offset_igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_get_d_offset_igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_get_d_size_igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_get_b_packed_size_igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_run_bt_pack_igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_run_igemm_igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv};

// Fill buffer with random float16_t values
void fill_random(float16_t *data, size_t count, unsigned seed)
{
    srand(seed);
    for (size_t i = 0; i < count; ++i)
        data[i] = (float16_t)(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
}

// Naive reference conv (CL layout: input [IH, IW, IC], output [OH, OW, OC])
void reference_conv_cl(const float16_t *activation, const float16_t *weight, const float16_t *bias,
                       float16_t *output, const tqt_conv_params *p, float16_t cmin, float16_t cmax)
{
    size_t IC_pg = p->input_channel / p->groups;
    size_t OC_pg = p->output_channel / p->groups;
    size_t OD = p->output_shape[0], OH = p->output_shape[1], OW = p->output_shape[2];
    size_t ID = p->input_shape[0], IH = p->input_shape[1], IW = p->input_shape[2];
    size_t KD = p->kernel_shape[0], KH = p->kernel_shape[1], KW = p->kernel_shape[2];

    for (size_t od = 0; od < OD; ++od) {
        for (size_t oh = 0; oh < OH; ++oh) {
            for (size_t ow = 0; ow < OW; ++ow) {
                for (size_t oc = 0; oc < OC_pg; ++oc) {
                    float acc = 0.0f;
                    for (size_t kd = 0; kd < KD; ++kd) {
                        for (size_t kh = 0; kh < KH; ++kh) {
                            for (size_t kw = 0; kw < KW; ++kw) {
                                int id =
                                    (int)(od * p->stride[0] + kd * p->dilation[0]) - (int)p->pad[0];
                                int ih =
                                    (int)(oh * p->stride[1] + kh * p->dilation[1]) - (int)p->pad[1];
                                int iw =
                                    (int)(ow * p->stride[2] + kw * p->dilation[2]) - (int)p->pad[2];
                                if (id >= 0 && id < (int)ID && ih >= 0 && ih < (int)IH && iw >= 0 &&
                                    iw < (int)IW) {
                                    for (size_t ic = 0; ic < IC_pg; ++ic) {
                                        // CL input: [ID, IH, IW, IC]
                                        float in_val =
                                            (float)activation[((size_t)id * IH * IW +
                                                               (size_t)ih * IW + (size_t)iw) *
                                                                  IC_pg +
                                                              ic];
                                        // Weight: [OC, KD*KH*KW*IC] row-major
                                        float w_val =
                                            (float)weight[oc * KD * KH * KW * IC_pg +
                                                          kd * KH * KW * IC_pg + kh * KW * IC_pg +
                                                          kw * IC_pg + ic];
                                        acc += in_val * w_val;
                                    }
                                }
                            }
                        }
                    }
                    if (bias)
                        acc += (float)bias[oc];
                    acc = fmaxf(acc, (float)cmin);
                    acc = fminf(acc, (float)cmax);
                    // CL output: [OD, OH, OW, OC]
                    output[(od * OH * OW + oh * OW + ow) * OC_pg + oc] = (float16_t)acc;
                }
            }
        }
    }
}

// Calculate cosine similarity between two float16_t vectors
float calculate_cosine_similarity(size_t size, const float16_t *ref, const float16_t *act)
{
    double dot_product = 0.0;
    double ref_norm = 0.0;
    double act_norm = 0.0;

    for (size_t i = 0; i < size; ++i) {
        float ref_val = (float)ref[i];
        float act_val = (float)act[i];

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
    // CL layout: M = spatial (OD*OH*OW), N = OC
    size_t M = OD * OH * OW;
    size_t N = OC_pg;
    size_t K = params.kernel_shape[0] * params.kernel_shape[1] * params.kernel_shape[2] * IC_pg;

    // Allocate the memory
    // CL activation layout: [ID, IH, IW, IC]
    float16_t *activation = (float16_t *)malloc(ID * IH * IW * IC_pg * sizeof(float16_t));
    // Weight: [OC, K] row-major
    float16_t *weight = (float16_t *)malloc(OC_pg * K * sizeof(float16_t));
    float16_t *bias = (float16_t *)malloc(OC_pg * sizeof(float16_t));
    // CL output layout: [OD, OH, OW, OC]
    float16_t *ref_output = (float16_t *)malloc(M * N * sizeof(float16_t));
    float16_t *act_output = (float16_t *)malloc(M * N * sizeof(float16_t));

    if (!activation || !weight || !bias || !ref_output || !act_output) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize data
    fill_random(activation, ID * IH * IW * IC_pg, 42);
    fill_random(weight, OC_pg * K, 123);
    fill_random(bias, OC_pg, 456);

    float16_t clamp_min = (float16_t)(-65504.0f);
    float16_t clamp_max = (float16_t)(65504.0f);

    // Run reference implementation (CL layout)
    reference_conv_cl(activation, weight, bias, ref_output, &params, clamp_min, clamp_max);

    // Pack weight [N, K] -> packed format
    size_t w_packed_size = ukernel.get_b_packed_size(N, K);
    void *w_packed = malloc(w_packed_size);
    ukernel.run_bt_pack(N, K, K, K, 0, weight, w_packed);

    // Run igemmcl micro-kernel with packed weight (no im2col needed)
    // M = spatial (OD*OH*OW), N = OC, K = KD*KH*KW*IC
    // ldb = K (weight row stride), ldd = N (output row stride = OC)
    ukernel.run_igemm(M, N, K, activation, 0, 0, 0, w_packed, K, 0, NULL, N, act_output, N, bias,
                      clamp_min, clamp_max, &params);

    // Calculate cosine similarity
    const float cosine_sim = calculate_cosine_similarity(M * N, ref_output, act_output);
    const bool passed = cosine_sim > 0.9999f;

    printf("TEST[igemmcl_1xnbias_clamp_f16_f16_f16p]\n");
    printf("- ukernel: igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(activation);
    free(weight);
    free(bias);
    free(ref_output);
    free(act_output);
    free(w_packed);

    return passed ? 0 : 1;
}
