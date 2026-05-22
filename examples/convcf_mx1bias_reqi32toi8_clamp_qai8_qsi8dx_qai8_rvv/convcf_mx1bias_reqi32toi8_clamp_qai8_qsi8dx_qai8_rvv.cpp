//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"
#include "src/gemm/gemm_i8_i8/tqt_utils_i8_i8.h"

// Include micro-kernel variants
#include "tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv.h"
#include "tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_interface.h"

namespace
{

/// Micro-kernel interface
const tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_ukernel ukernel{
    tqt_get_m_step_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    tqt_get_n_step_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    tqt_get_a_offset_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    tqt_get_c_offset_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    tqt_get_bias_offset_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    tqt_get_d_offset_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    tqt_get_d_size_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    tqt_get_b_im2col_size_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    tqt_get_b_im2col_offset_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    tqt_run_im2col_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    tqt_run_conv_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv};

// Naive reference conv (CF layout, fp32)
// CF layout: activation [IC, ID, IH, IW], weight [OC, IC, KD, KH, KW]
void reference_conv_cf(const float *activation, const float *weight, const float *bias,
                       float *output, const tqt_conv_params *p)
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
                    output[oc * OD * OH * OW + od * OH * OW + oh * OW + ow] = acc;
                }
            }
        }
    }
}

void find_min_max(const float *data, size_t size, float *min_val, float *max_val)
{
    float fmin = FLT_MAX;
    float fmax = -FLT_MAX;
    for (size_t i = 0; i < size; ++i) {
        if (data[i] < fmin)
            fmin = data[i];
        if (data[i] > fmax)
            fmax = data[i];
    }
    *min_val = fmin;
    *max_val = fmax;
}

void get_quant_info_asym_i8(const float *data, size_t size, float *scale, int32_t *zero_point)
{
    float fmin, fmax;
    find_min_max(data, size, &fmin, &fmax);

    fmax = fmax > 0.0f ? fmax : 0.0f;
    fmin = fmin < 0.0f ? fmin : 0.0f;

    float scale_tmp = (fmax - fmin) / 255.0f;
    if (scale_tmp == 0.0f) {
        *scale = 1.0f;
        *zero_point = 0;
        return;
    }

    float zp_tmp = -128.0f - fmin / scale_tmp;
    int32_t zp = (int32_t)nearbyintf(zp_tmp);
    if (zp == -128)
        zp = -127;

    *scale = scale_tmp;
    *zero_point = zp;
}

void get_quant_info_sym_i8(const float *data, size_t size, float *scale)
{
    float fmin, fmax;
    find_min_max(data, size, &fmin, &fmax);

    float abs_max = (fabsf(fmax) >= fabsf(fmin)) ? fabsf(fmax) : fabsf(fmin);
    float scale_tmp = 2.0f * abs_max / 255.0f;
    if (scale_tmp == 0.0f)
        scale_tmp = 1.0f;

    *scale = scale_tmp;
}

void quantize_fp32_to_i8(const float *src, int8_t *dst, size_t size, float scale,
                         int32_t zero_point)
{
    for (size_t i = 0; i < size; ++i) {
        float val = nearbyintf(src[i] / scale) + (float)zero_point;
        if (val > 127.0f)
            val = 127.0f;
        if (val < -128.0f)
            val = -128.0f;
        dst[i] = (int8_t)val;
    }
}

void quantize_fp32_to_i32(const float *src, int32_t *dst, size_t size, float scale,
                          int32_t zero_point)
{
    for (size_t i = 0; i < size; ++i) {
        float val = nearbyintf(src[i] / scale) + (float)zero_point;
        if (val > 2147483647.0f)
            val = 2147483647.0f;
        if (val < -2147483648.0f)
            val = -2147483648.0f;
        dst[i] = (int32_t)val;
    }
}

void dequantize_i8_to_fp32(const int8_t *src, float *dst, size_t size, float scale,
                           int32_t zero_point)
{
    for (size_t i = 0; i < size; ++i) {
        dst[i] = ((float)src[i] - (float)zero_point) * scale;
    }
}

// Fuse zero_point of asymmetric-quantized B (activation) into bias
// For CF mx1bias: A = weight [M,K] (per-channel sym), B = activation [K,N] (global asym)
// bias_fused[m] = bias_q[m] - zp_b * sum(A_i8[m, 0..K-1])
void fuse_zp_to_bias(int32_t *bias_fused, const int32_t *bias, const int8_t *weight_i8,
                     int32_t zero_point, size_t num_channels, size_t channel_size)
{
    for (size_t ch = 0; ch < num_channels; ++ch) {
        int32_t weight_sum = 0;
        for (size_t k = 0; k < channel_size; ++k) {
            weight_sum += (int32_t)weight_i8[ch * channel_size + k];
        }
        bias_fused[ch] = bias[ch] - zero_point * weight_sum;
    }
}

void fill_matrix_random(size_t rows, size_t cols, float *matrix, const float min, const float max)
{
    for (size_t i = 0; i < rows * cols; ++i) {
        float rand_val = min + (max - min) * ((float)rand() / (float)RAND_MAX);
        matrix[i] = rand_val;
    }
}

float calculate_cosine_similarity(size_t size, const float *ref, const float *act)
{
    double dot_product = 0.0;
    double ref_norm = 0.0;
    double act_norm = 0.0;

    for (size_t i = 0; i < size; ++i) {
        double ref_val = (double)ref[i];
        double act_val = (double)act[i];

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
    tqt_conv_params conv_params = {};
    conv_params.input_channel = 3;
    conv_params.output_channel = 16;
    conv_params.groups = 1;
    conv_params.input_shape[0] = 1;
    conv_params.input_shape[1] = 8;
    conv_params.input_shape[2] = 8;
    conv_params.kernel_shape[0] = 1;
    conv_params.kernel_shape[1] = 3;
    conv_params.kernel_shape[2] = 3;
    conv_params.stride[0] = 1;
    conv_params.stride[1] = 1;
    conv_params.stride[2] = 1;
    conv_params.dilation[0] = 1;
    conv_params.dilation[1] = 1;
    conv_params.dilation[2] = 1;
    conv_params.pad[0] = 0;
    conv_params.pad[1] = 1;
    conv_params.pad[2] = 1;
    for (int i = 0; i < 3; ++i)
        conv_params.output_shape[i] =
            (conv_params.input_shape[i] + 2 * conv_params.pad[i] -
             conv_params.dilation[i] * (conv_params.kernel_shape[i] - 1) - 1) /
                conv_params.stride[i] +
            1;

    size_t IC_pg = conv_params.input_channel;
    size_t OC_pg = conv_params.output_channel;
    size_t OD = conv_params.output_shape[0], OH = conv_params.output_shape[1],
           OW = conv_params.output_shape[2];
    size_t ID = conv_params.input_shape[0], IH = conv_params.input_shape[1],
           IW = conv_params.input_shape[2];
    // CF: M = OC, N = OD*OH*OW, K = IC*KD*KH*KW
    size_t M = OC_pg;
    size_t N = OD * OH * OW;
    size_t K = IC_pg * conv_params.kernel_shape[0] * conv_params.kernel_shape[1] *
               conv_params.kernel_shape[2];

    // Allocate fp32 data
    float *activation_fp32 = (float *)malloc(IC_pg * ID * IH * IW * sizeof(float));
    float *weight_fp32 = (float *)malloc(OC_pg * K * sizeof(float));
    float *bias_fp32 = (float *)malloc(OC_pg * sizeof(float));
    float *D_ref_fp32 = (float *)malloc(M * N * sizeof(float));

    if (!activation_fp32 || !weight_fp32 || !bias_fp32 || !D_ref_fp32) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize fp32 matrices
    fill_matrix_random(IC_pg, ID * IH * IW, activation_fp32, -1.0f, 1.0f);
    fill_matrix_random(OC_pg, K, weight_fp32, -1.0f, 1.0f);
    fill_matrix_random(1, OC_pg, bias_fp32, -10.0f, 10.0f);

    // Run fp32 reference conv
    reference_conv_cf(activation_fp32, weight_fp32, bias_fp32, D_ref_fp32, &conv_params);

    // Quantization parameters
    // A (weight): per-channel symmetric (each row of weight [M,K] is a channel)
    float *scale_a = (float *)malloc(M * sizeof(float));
    for (size_t ch = 0; ch < M; ++ch) {
        get_quant_info_sym_i8(&weight_fp32[ch * K], K, &scale_a[ch]);
    }

    // B (activation after im2col): global asymmetric
    float scale_b;
    int32_t zp_b;
    get_quant_info_asym_i8(activation_fp32, IC_pg * ID * IH * IW, &scale_b, &zp_b);

    // D (output): global asymmetric
    float scale_d;
    int32_t zp_d;
    get_quant_info_asym_i8(D_ref_fp32, M * N, &scale_d, &zp_d);

    // Allocate quantized buffers
    int8_t *weight_i8 = (int8_t *)malloc(M * K * sizeof(int8_t));
    int8_t *activation_i8 = (int8_t *)malloc(IC_pg * ID * IH * IW * sizeof(int8_t));
    int32_t *bias_i32 = (int32_t *)malloc(M * sizeof(int32_t));
    int32_t *bias_fused = (int32_t *)malloc(M * sizeof(int32_t));
    int8_t *D_i8 = (int8_t *)malloc(M * N * sizeof(int8_t));
    float *D_deq = (float *)malloc(M * N * sizeof(float));

    if (!scale_a || !weight_i8 || !activation_i8 || !bias_i32 || !bias_fused || !D_i8 || !D_deq) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Quantize weight (per-channel symmetric)
    for (size_t ch = 0; ch < M; ++ch) {
        quantize_fp32_to_i8(&weight_fp32[ch * K], &weight_i8[ch * K], K, scale_a[ch], 0);
    }

    // Quantize activation (global asymmetric)
    quantize_fp32_to_i8(activation_fp32, activation_i8, IC_pg * ID * IH * IW, scale_b, zp_b);

    // Quantize bias (Mx1, scale = scale_a[ch] * scale_b, zp = 0)
    for (size_t ch = 0; ch < M; ++ch) {
        float bias_scale = scale_a[ch] * scale_b;
        quantize_fp32_to_i32(&bias_fp32[ch], &bias_i32[ch], 1, bias_scale, 0);
    }

    // Fuse zero_point_b into bias: bias_fused[m] = bias[m] - zp_b * sum(weight_i8[m, 0..K-1])
    fuse_zp_to_bias(bias_fused, bias_i32, weight_i8, zp_b, M, K);

    // Set up quantization params
    struct tqt_qai8_qsi8dx_qai8_params qparams;
    qparams.scale_a = scale_a;
    qparams.quant_channel_a = (int32_t)M;
    qparams.scale_b = scale_b;
    qparams.zero_point_b = zp_b;
    qparams.scale_d = scale_d;
    qparams.zero_point_d = zp_d;

    int8_t clamp_min = -128;
    int8_t clamp_max = 127;

    memset(D_i8, 0, M * N * sizeof(int8_t));

    // Run micro-kernel implementation
    // im2col B (activation) [K, N]
    size_t im2col_size = ukernel.get_b_im2col_size(K, N);
    void *im2col_buf = malloc(im2col_size);

    ukernel.run_im2col(0, activation_i8, im2col_buf, &conv_params);

    // Run conv: A=weight[M,K], B=im2col[K,N]
    ukernel.run_conv(M, N, K, weight_i8, K, 0, im2col_buf, N, 0, NULL, N, D_i8, N, bias_fused,
                     clamp_min, clamp_max, &qparams);

    // Dequantize kernel output and compare with fp32 reference
    dequantize_i8_to_fp32(D_i8, D_deq, M * N, scale_d, zp_d);

    // Calculate cosine similarity
    const float cosine_sim = calculate_cosine_similarity(M * N, D_ref_fp32, D_deq);
    const bool passed = cosine_sim > 0.999f;

    printf("TEST[convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8]\n");
    printf("- ukernel: convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(activation_fp32);
    free(weight_fp32);
    free(bias_fp32);
    free(D_ref_fp32);
    free(D_deq);
    free(weight_i8);
    free(activation_i8);
    free(bias_i32);
    free(bias_fused);
    free(D_i8);
    free(scale_a);
    free(im2col_buf);

    return passed ? 0 : 1;
}
