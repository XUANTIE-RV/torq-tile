//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/reference_conv.h"

#include <algorithm>
#include <cmath>

namespace torq_tile
{
namespace test
{

template <typename T>
void reference_conv(size_t group_idx, const T *activation, const T *weight, const T *bias,
                    T *output, const tqt_conv_params *params, ConvLayout layout, T clamp_min,
                    T clamp_max)
{
    const float clamp_min_f = static_cast<float>(clamp_min);
    const float clamp_max_f = static_cast<float>(clamp_max);

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

    const size_t IC_per_group = params->input_channel / params->groups;
    const size_t OC_per_group = params->output_channel / params->groups;

    for (size_t oc = 0; oc < OC_per_group; ++oc) {
        for (size_t od = 0; od < OD; ++od) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    float acc = 0.0f;

                    for (size_t ic = 0; ic < IC_per_group; ++ic) {
                        for (size_t kd = 0; kd < KD; ++kd) {
                            for (size_t kh = 0; kh < KH; ++kh) {
                                for (size_t kw = 0; kw < KW; ++kw) {
                                    const int id_in = (int)(od * SD + kd * DD) - (int)PD;
                                    const int ih_in = (int)(oh * SH + kh * DH) - (int)PH;
                                    const int iw_in = (int)(ow * SW + kw * DW) - (int)PW;

                                    float input_val = 0.0f;
                                    if (id_in >= 0 && id_in < (int)ID && ih_in >= 0 &&
                                        ih_in < (int)IH && iw_in >= 0 && iw_in < (int)IW) {
                                        if (layout == ConvLayout::kChannelFirst) {
                                            // activation: [IC_total, ID, IH, IW]
                                            size_t ic_global = group_idx * IC_per_group + ic;
                                            input_val = static_cast<float>(
                                                activation[ic_global * ID * IH * IW +
                                                           (size_t)id_in * IH * IW +
                                                           (size_t)ih_in * IW + (size_t)iw_in]);
                                        } else {
                                            // activation: [ID, IH, IW, IC_total]
                                            size_t ic_global = group_idx * IC_per_group + ic;
                                            size_t C_total = params->input_channel;
                                            input_val = static_cast<float>(
                                                activation[((size_t)id_in * IH * IW +
                                                            (size_t)ih_in * IW + (size_t)iw_in) *
                                                               C_total +
                                                           ic_global]);
                                        }
                                    }

                                    // weight layout depends on conv layout:
                                    // CF: [OC_per_group, IC_per_group, KD, KH, KW]
                                    // CL: [OC_per_group, KD, KH, KW, IC_per_group]
                                    float weight_val;
                                    if (layout == ConvLayout::kChannelFirst) {
                                        weight_val = static_cast<float>(
                                            weight[oc * IC_per_group * KD * KH * KW +
                                                   ic * KD * KH * KW + kd * KH * KW + kh * KW +
                                                   kw]);
                                    } else {
                                        weight_val = static_cast<float>(
                                            weight[oc * KD * KH * KW * IC_per_group +
                                                   kd * KH * KW * IC_per_group +
                                                   kh * KW * IC_per_group + kw * IC_per_group +
                                                   ic]);
                                    }

                                    acc += input_val * weight_val;
                                }
                            }
                        }
                    }

                    if (bias) {
                        acc += static_cast<float>(bias[oc]);
                    }

                    acc = std::fmax(acc, clamp_min_f);
                    acc = std::fmin(acc, clamp_max_f);

                    if (layout == ConvLayout::kChannelFirst) {
                        // output: [OC_per_group, OD, OH, OW]
                        output[oc * OD * OH * OW + od * OH * OW + oh * OW + ow] =
                            static_cast<T>(acc);
                    } else {
                        // output: [OD, OH, OW, OC_per_group]
                        output[(od * OH * OW + oh * OW + ow) * OC_per_group + oc] =
                            static_cast<T>(acc);
                    }
                }
            }
        }
    }
}

// Explicit instantiations
template void reference_conv<float>(size_t, const float *, const float *, const float *, float *,
                                    const tqt_conv_params *, ConvLayout, float, float);
template void reference_conv<float16_t>(size_t, const float16_t *, const float16_t *,
                                        const float16_t *, float16_t *, const tqt_conv_params *,
                                        ConvLayout, float16_t, float16_t);

}  // namespace test
}  // namespace torq_tile
