//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

namespace torq_tile
{
namespace test
{

enum class ConvLayout
{
    kChannelFirst,  // NCDHW
    kChannelLast,   // NDHWC
};

/// Reference convolution (naive, single-batch, single-group dispatch).
///
/// For channel-first:
///   activation: [IC, ID, IH, IW]  (per full input, all groups)
///   weight:     [OC_per_group, IC_per_group, KD, KH, KW]
///   bias:       [OC_per_group] or nullptr
///   output:     [OC_per_group, OD, OH, OW]
///
/// For channel-last:
///   activation: [ID, IH, IW, IC]  (per full input, all groups)
///   weight:     [OC_per_group, KD, KH, KW, IC_per_group]  (matches im2col_cl K ordering)
///   bias:       [OC_per_group] or nullptr
///   output:     [OD, OH, OW, OC_per_group]
template <typename T>
void reference_conv(size_t group_idx, const T *activation, const T *weight, const T *bias,
                    T *output, const tqt_conv_params *params, ConvLayout layout, T clamp_min,
                    T clamp_max);

extern template void reference_conv<float>(size_t, const float *, const float *, const float *,
                                           float *, const tqt_conv_params *, ConvLayout, float,
                                           float);
extern template void reference_conv<float16_t>(size_t, const float16_t *, const float16_t *,
                                               const float16_t *, float16_t *,
                                               const tqt_conv_params *, ConvLayout, float16_t,
                                               float16_t);

}  // namespace test
}  // namespace torq_tile
