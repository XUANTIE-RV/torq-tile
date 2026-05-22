//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"
#include "test/common/reference_conv.h"  // reuse ConvLayout enum

namespace torq_tile
{
namespace test
{

/// Reference depthwise convolution (naive, single-batch).
///
/// Constraints:
///   params->groups == params->input_channel  (dwconv requirement)
///   params->output_channel % params->input_channel == 0  (dm = OC / IC, integer)
///
/// PyTorch OC ordering:  oc = ic + m * IC,  m in [0, dm), ic in [0, IC).
/// Therefore: ic = oc % IC.
///
/// For channel-first (NCDHW):
///   input:  [IC, ID, IH, IW]
///   weight: [OC, 1, KD, KH, KW]   (PyTorch dwconv weight; the "1" is IC_per_group)
///   bias:   [OC] or nullptr
///   output: [OC, OD, OH, OW]
///
/// For channel-last (NDHWC):
///   input:  [ID, IH, IW, IC]
///   weight: [KD, KH, KW, 1, OC]   (HWIO style; the "1" is IC_per_group)
///   bias:   [OC] or nullptr
///   output: [OD, OH, OW, OC]
template <typename T>
void reference_dwconv(const T *input, const T *weight, const T *bias, T *output,
                      const tqt_conv_params *params, ConvLayout layout, T clamp_min, T clamp_max);

extern template void reference_dwconv<float>(const float *, const float *, const float *, float *,
                                             const tqt_conv_params *, ConvLayout, float, float);
extern template void reference_dwconv<float16_t>(const float16_t *, const float16_t *,
                                                 const float16_t *, float16_t *,
                                                 const tqt_conv_params *, ConvLayout, float16_t,
                                                 float16_t);

}  // namespace test
}  // namespace torq_tile
