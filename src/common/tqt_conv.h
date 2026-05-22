//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Convolution parameters (unified 3D, 1D/2D set higher dims to 1)
// ============================================================================

struct tqt_conv_params
{
    size_t input_shape[3];   // {input_d, input_h, input_w}
    size_t output_shape[3];  // {output_d, output_h, output_w} (caller pre-computes)
    size_t kernel_shape[3];  // {kernel_d, kernel_h, kernel_w}
    size_t stride[3];        // {stride_d, stride_h, stride_w}
    size_t dilation[3];      // {dilation_d, dilation_h, dilation_w}
    size_t pad[3];           // {pad_d, pad_h, pad_w}
    size_t input_channel;
    size_t output_channel;
    size_t groups;
};
// 1D conv: input_shape={1,1,IW}, kernel_shape={1,1,KW}, ...
// 2D conv: input_shape={1,IH,IW}, kernel_shape={1,KH,KW}, ...
// 3D conv: input_shape={ID,IH,IW}, kernel_shape={KD,KH,KW}, ...

#ifdef __cplusplus
}  // extern "C"
#endif

// ============================================================================
// im2col for channel-first (NCDHW) layout
// ============================================================================
//
// Activation layout: [N, C, D, H, W]  (per-group: [IC_per_group, ID, IH, IW])
// im2col output: [IC_per_group * KD * KH * KW, OD * OH * OW]  (column-major per output pixel)
//
// This is used by convcf where:
//   A = weight [OC_per_group, IC_per_group * KD * KH * KW]  (row-major)
//   B = im2col(activation) [IC_per_group * KD * KH * KW, OD * OH * OW]  (row-major)

template <typename T>
static inline void tqt_im2col_cf(size_t group_idx, const void *activation, void *im2col_buf,
                                 const tqt_conv_params *params)
{
    const T *input = (const T *)activation;
    T *col = (T *)im2col_buf;

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
    const size_t N_out = OD * OH * OW;  // number of output spatial locations

    // Input pointer offset for this group
    const T *input_group = input + group_idx * IC_per_group * ID * IH * IW;

    // im2col: rows = IC_per_group * KD * KH * KW, cols = OD * OH * OW
    // Layout: row-major [K, N] where K = IC_per_group * KD * KH * KW, N = OD * OH * OW
    size_t col_row = 0;
    for (size_t ic = 0; ic < IC_per_group; ++ic) {
        for (size_t kd = 0; kd < KD; ++kd) {
            for (size_t kh = 0; kh < KH; ++kh) {
                for (size_t kw = 0; kw < KW; ++kw) {
                    size_t col_col = 0;
                    for (size_t od = 0; od < OD; ++od) {
                        const size_t id_val = od * SD + kd * DD;
                        const int id_in = (int)id_val - (int)PD;
                        for (size_t oh = 0; oh < OH; ++oh) {
                            const size_t ih_val = oh * SH + kh * DH;
                            const int ih_in = (int)ih_val - (int)PH;
                            for (size_t ow = 0; ow < OW; ++ow) {
                                const size_t iw_val = ow * SW + kw * DW;
                                const int iw_in = (int)iw_val - (int)PW;

                                if (id_in >= 0 && id_in < (int)ID && ih_in >= 0 &&
                                    ih_in < (int)IH && iw_in >= 0 && iw_in < (int)IW) {
                                    col[col_row * N_out + col_col] =
                                        input_group[ic * ID * IH * IW + (size_t)id_in * IH * IW +
                                                    (size_t)ih_in * IW + (size_t)iw_in];
                                } else {
                                    col[col_row * N_out + col_col] = T(0);
                                }
                                col_col++;
                            }
                        }
                    }
                    col_row++;
                }
            }
        }
    }
}

// ============================================================================
// im2col for channel-last (NDHWC) layout
// ============================================================================
//
// Activation layout: [N, D, H, W, C]  (per-group: [ID, IH, IW, IC_per_group])
// im2col output: [OD * OH * OW, KD * KH * KW * IC_per_group]  (row-major)
//
// This is used by convcl where:
//   A = im2col(activation) [OD*OH*OW, KD*KH*KW*IC_per_group]  (row-major)
//   B = weight [OC_per_group, KD*KH*KW*IC_per_group]  (transposed, row-major)

template <typename T>
static inline void tqt_im2col_cl(size_t group_idx, const void *activation, void *im2col_buf,
                                 const tqt_conv_params *params)
{
    const T *input = (const T *)activation;
    T *col = (T *)im2col_buf;

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

    const size_t C_total = params->input_channel;
    const size_t IC_per_group = C_total / params->groups;
    const size_t K_col = KD * KH * KW * IC_per_group;  // columns of im2col output

    // For channel-last, input is [D, H, W, C], with all channels interleaved
    // Group offset is within the channel dimension
    const size_t c_offset = group_idx * IC_per_group;

    // im2col: rows = OD * OH * OW, cols = KD * KH * KW * IC_per_group
    size_t col_row = 0;
    for (size_t od = 0; od < OD; ++od) {
        for (size_t oh = 0; oh < OH; ++oh) {
            for (size_t ow = 0; ow < OW; ++ow) {
                size_t col_col = 0;
                for (size_t kd = 0; kd < KD; ++kd) {
                    const size_t id_val = od * SD + kd * DD;
                    const int id_in = (int)id_val - (int)PD;
                    for (size_t kh = 0; kh < KH; ++kh) {
                        const size_t ih_val = oh * SH + kh * DH;
                        const int ih_in = (int)ih_val - (int)PH;
                        for (size_t kw = 0; kw < KW; ++kw) {
                            const size_t iw_val = ow * SW + kw * DW;
                            const int iw_in = (int)iw_val - (int)PW;

                            if (id_in >= 0 && id_in < (int)ID && ih_in >= 0 && ih_in < (int)IH &&
                                iw_in >= 0 && iw_in < (int)IW) {
                                const T *src =
                                    input +
                                    ((size_t)id_in * IH * IW + (size_t)ih_in * IW + (size_t)iw_in) *
                                        C_total +
                                    c_offset;
                                memcpy(&col[col_row * K_col + col_col], src,
                                       IC_per_group * sizeof(T));
                            } else {
                                memset(&col[col_row * K_col + col_col], 0,
                                       IC_per_group * sizeof(T));
                            }
                            col_col += IC_per_group;
                        }
                    }
                }
                col_row++;
            }
        }
    }
}
