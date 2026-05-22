//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "src/common/tqt_common.h"

namespace torq_tile
{
namespace test
{

/// Quantize fp16 data to int4 with per-block symmetric scale.
///
/// Algorithm reference: csi-nn2 block_quantize_q4
///   scale = max_value / -8.0f
///   q = clamp(round(value / scale) + 8, 0, 15)
///
/// @param n         Number of rows in src
/// @param k         Number of columns in src, must be a multiple of bl
/// @param bl        Block length for quantization, >= 2 and multiple of 2
/// @param src       Input matrix [n, k] fp16, row-major
/// @param dst       Output packed int4 [n, k/2] uint8, two nibbles per byte
/// @param scale     Output scale matrix [n, k/bl] fp16, row-major
/// @param ld_scale  Leading dimension of scale (= k/bl)
inline void quantize_fp16_to_int4(size_t n, size_t k, size_t bl, const float16_t *src, uint8_t *dst,
                                  float16_t *scale, size_t ld_scale)
{
    memset(dst, 0, n * (k / 2));

    for (size_t j = 0; j < n; ++j) {
        const size_t num_blocks = k / bl;
        for (size_t b = 0; b < num_blocks; ++b) {
            // Find the element with the largest absolute value (keep sign)
            float max_value = 0.0f;
            float abs_max_value = 0.0f;
            for (size_t t = 0; t < bl; ++t) {
                float value = static_cast<float>(src[j * k + b * bl + t]);
                if (fabsf(value) > abs_max_value) {
                    abs_max_value = fabsf(value);
                    max_value = value;
                }
            }

            // scale = max_value / -8 (reference: csi-nn2)
            float fp32_scale = max_value / -8.0f;
            float inv_scale = (fp32_scale != 0.0f) ? (1.0f / fp32_scale) : 0.0f;
            scale[j * ld_scale + b] = static_cast<float16_t>(fp32_scale);

            // Quantize and pack two nibbles per byte
            for (size_t t = 0; t < bl; t += 2) {
                float value0 = static_cast<float>(src[j * k + b * bl + t]);
                float value1 = static_cast<float>(src[j * k + b * bl + t + 1]);
                uint8_t q0 = static_cast<uint8_t>(
                    fminf(static_cast<float>(static_cast<int8_t>(value0 * inv_scale + 8.5f)), 15));
                uint8_t q1 = static_cast<uint8_t>(
                    fminf(static_cast<float>(static_cast<int8_t>(value1 * inv_scale + 8.5f)), 15));
                dst[j * (k / 2) + (b * bl + t) / 2] = q0 | static_cast<uint8_t>(q1 << 4);
            }
        }
    }
}

}  // namespace test
}  // namespace torq_tile
