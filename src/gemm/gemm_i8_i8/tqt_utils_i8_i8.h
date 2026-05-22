//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

static int64_t integer_from_exp(double input, int32_t *shift)
{
    uint64_t kSignMask = 0x8000000000000000LL;
    uint64_t kExponentMask = 0x7ff0000000000000LL;
    int32_t kExponentShift = 52;
    int32_t kExponentBias = 1023;
    uint32_t kExponentIsBadNum = 0x7ff;
    uint64_t kFractionMask = 0x000fffffffc00000LL;
    uint32_t kFractionShift = 22;
    uint32_t kFractionRoundingMask = 0x003fffff;
    uint32_t kFractionRoundingThreshold = 0x00200000;

    // We want to access the bits of the input double value directly, which is
    // tricky to do safely, so use a union to handle the casting.
    union
    {
        double double_value;
        uint64_t double_as_uint;
    } cast_union;

    cast_union.double_value = input;
    const uint64_t u = cast_union.double_as_uint;

    // If the bitfield is all zeros apart from the sign bit, this is a normalized
    // zero value, so return standard values for this special case.
    if ((u & ~kSignMask) == 0) {
        *shift = 0;
        return 0;
    }

    // Deal with NaNs and Infs, which are always indicated with a fixed pattern in
    // the exponent, and distinguished by whether the fractions are zero or
    // non-zero.
    const uint32_t exponent_part = ((u & kExponentMask) >> kExponentShift);
    if (exponent_part == kExponentIsBadNum) {
        *shift = 0x7fffffff;
        if (u & kFractionMask) {
            // NaN, so just return zero (with the exponent set to INT_MAX).
            return 0;
        } else {
            // Infinity, so return +/- INT_MAX.
            if (u & kSignMask) {
                return 0x8000000000000000;
            } else {
                return 0x7fffffffffffffff;
            }
        }
    }

    // The shift is fairly easy to extract from the high bits of the double value,
    // just by masking it out and applying a bias. The std::frexp() implementation
    // always returns values between 0.5 and 1.0 though, whereas the exponent
    // assumes 1.0 to 2.0 is the standard range, so I add on one to match that
    // interface.
    *shift = (exponent_part - kExponentBias) + 1;

    // There's an implicit high bit in the double format definition, so make sure
    // we include that at the top, and then reconstruct the rest of the fractional
    // value from the remaining fragments.
    int64_t fraction = 0x40000000 + ((u & kFractionMask) >> kFractionShift);

    // We're cutting off some bits at the bottom, so to exactly match the standard
    // frexp implementation here we'll apply rounding by adding one to the least
    // significant bit of the result if the discarded portion is over half of the
    // maximum.
    if ((u & kFractionRoundingMask) > kFractionRoundingThreshold) {
        fraction += 1;
    }
    // Negate the fraction if the sign bit was set.
    if (u & kSignMask) {
        fraction *= -1;
    }

    return fraction;
}

/// Convert a double multiplier to a quantized multiplier and shift.
/// This is used for requantization in int8 GEMM operations.
///
/// @param[in]  double_multiplier The floating-point multiplier
/// @param[out] quantized_multiplier The output quantized multiplier (int32)
/// @param[out] shift The output shift value
static inline void tqt_quantize_multiplier(double double_multiplier, int32_t *quantized_multiplier,
                                           int32_t *shift)
{
    if (double_multiplier == 0.) {
        *quantized_multiplier = 0;
        *shift = 0;
        return;
    }

    // If we're trying to avoid the use of floating-point instructions (for
    // example on microcontrollers) then use an alternative implementation
    // that only requires integer and bitwise operations. To enable this, you
    // need to set the define during the build process for your platform.
    int64_t q_fixed = integer_from_exp(double_multiplier, shift);

    if (q_fixed == (1ll << 31)) {
        q_fixed /= 2;
        ++*shift;
    }
    // A shift amount smaller than -31 would cause all bits to be shifted out
    // and thus all results would be zero. We implement that instead with
    // q_fixed==0, so as to avoid hitting issues with right-shift
    // operations with shift amounts greater than 31. Note that this happens
    // roughly when abs(double_multiplier) < 2^-31 and the present handling means
    // that we're effectively flushing tiny double_multiplier's to zero.
    // We could conceivably handle values in the range (roughly) [32, 63]
    // as 'denormals' i.e. (shift==0, q_fixed < 2^30). In that point of view
    // the present handling is just doing 'flush denormals to zero'. We could
    // reconsider and actually generate nonzero denormals if a need arises.
    if (*shift < -31) {
        *shift = 0;
        q_fixed = 0;
    }
    *quantized_multiplier = (int32_t)(q_fixed);
}

#ifdef __cplusplus
}  // extern "C"
#endif
