//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/reference_dwconv.h"

#include <algorithm>
#include <cmath>

namespace torq_tile
{
namespace test
{

template <typename T>
void reference_dwconv(const T *input, const T *weight, const T *bias, T *output,
                      const tqt_conv_params *params, ConvLayout layout, T clamp_min, T clamp_max)
{
    const float clamp_min_f = static_cast<float>(clamp_min);
    const float clamp_max_f = static_cast<float>(clamp_max);

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

    // dwconv constraint: groups == IC. PyTorch ordering oc = ic + m*IC -> ic = oc % IC.
    for (size_t oc = 0; oc < OC; ++oc) {
        const size_t ic = oc % IC;

        for (size_t od = 0; od < OD; ++od) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    float acc = 0.0f;

                    for (size_t kd = 0; kd < KD; ++kd) {
                        for (size_t kh = 0; kh < KH; ++kh) {
                            for (size_t kw = 0; kw < KW; ++kw) {
                                const int id_in = (int)(od * SD + kd * DD) - (int)PD;
                                const int ih_in = (int)(oh * SH + kh * DH) - (int)PH;
                                const int iw_in = (int)(ow * SW + kw * DW) - (int)PW;

                                if (id_in < 0 || id_in >= (int)ID || ih_in < 0 ||
                                    ih_in >= (int)IH || iw_in < 0 || iw_in >= (int)IW) {
                                    continue;
                                }

                                float input_val;
                                float weight_val;

                                if (layout == ConvLayout::kChannelFirst) {
                                    // input [IC, ID, IH, IW]
                                    input_val = static_cast<float>(
                                        input[ic * ID * IH * IW + (size_t)id_in * IH * IW +
                                              (size_t)ih_in * IW + (size_t)iw_in]);
                                    // weight [OC, 1, KD, KH, KW]
                                    weight_val = static_cast<float>(
                                        weight[oc * KD * KH * KW + kd * KH * KW + kh * KW + kw]);
                                } else {
                                    // input [ID, IH, IW, IC]
                                    input_val = static_cast<float>(
                                        input[((size_t)id_in * IH * IW + (size_t)ih_in * IW +
                                               (size_t)iw_in) *
                                                  IC +
                                              ic]);
                                    // weight [KD, KH, KW, 1, OC]
                                    weight_val = static_cast<float>(
                                        weight[(kd * KH * KW + kh * KW + kw) * OC + oc]);
                                }

                                acc += input_val * weight_val;
                            }
                        }
                    }

                    if (bias) {
                        acc += static_cast<float>(bias[oc]);
                    }

                    acc = std::fmax(acc, clamp_min_f);
                    acc = std::fmin(acc, clamp_max_f);

                    if (layout == ConvLayout::kChannelFirst) {
                        // output [OC, OD, OH, OW]
                        output[oc * OD * OH * OW + od * OH * OW + oh * OW + ow] =
                            static_cast<T>(acc);
                    } else {
                        // output [OD, OH, OW, OC]
                        output[((size_t)od * OH * OW + oh * OW + ow) * OC + oc] =
                            static_cast<T>(acc);
                    }
                }
            }
        }
    }
}

// Explicit instantiations
template void reference_dwconv<float>(const float *, const float *, const float *, float *,
                                      const tqt_conv_params *, ConvLayout, float, float);
template void reference_dwconv<float16_t>(const float16_t *, const float16_t *, const float16_t *,
                                          float16_t *, const tqt_conv_params *, ConvLayout,
                                          float16_t, float16_t);

}  // namespace test
}  // namespace torq_tile
