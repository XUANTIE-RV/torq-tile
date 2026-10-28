//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

static int64_t integer_from_exp(double input, int32_t *shift)
{
    const uint64_t sign_mask = 0x8000000000000000LL;
    const uint64_t exponent_mask = 0x7ff0000000000000LL;
    const int32_t exponent_shift = 52;
    const int32_t exponent_bias = 1023;
    const uint32_t exponent_is_bad_num = 0x7ff;
    const uint64_t fraction_mask = 0x000fffffffc00000LL;
    const uint32_t fraction_shift = 22;
    const uint32_t fraction_rounding_mask = 0x003fffff;
    const uint32_t fraction_rounding_threshold = 0x00200000;

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
    if ((u & ~sign_mask) == 0) {
        *shift = 0;
        return 0;
    }

    // Deal with NaNs and Infs, which are always indicated with a fixed pattern in
    // the exponent, and distinguished by whether the fractions are zero or
    // non-zero.
    const uint32_t exponent_part = ((u & exponent_mask) >> exponent_shift);
    if (exponent_part == exponent_is_bad_num) {
        *shift = 0x7fffffff;
        if (u & fraction_mask) {
            // NaN, so just return zero (with the exponent set to INT_MAX).
            return 0;
        } else {
            // Infinity, so return +/- INT_MAX.
            if (u & sign_mask) {
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
    *shift = (exponent_part - exponent_bias) + 1;

    // There's an implicit high bit in the double format definition, so make sure
    // we include that at the top, and then reconstruct the rest of the fractional
    // value from the remaining fragments.
    int64_t fraction = 0x40000000 + ((u & fraction_mask) >> fraction_shift);

    // We're cutting off some bits at the bottom, so to exactly match the standard
    // frexp implementation here we'll apply rounding by adding one to the least
    // significant bit of the result if the discarded portion is over half of the
    // maximum.
    if ((u & fraction_rounding_mask) > fraction_rounding_threshold) {
        fraction += 1;
    }
    // Negate the fraction if the sign bit was set.
    if (u & sign_mask) {
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
    *quantized_multiplier = (int32_t)(q_fixed);
}

#if defined(__riscv_flen) && __riscv_flen >= 64
/// Prepare Q31 parameters with explicit RNE, without reading or changing frm.
/// F32 inputs widen exactly; nonzero products and quotients fit the F64 normal
/// range for finite nonnegative input scales and a finite positive output scale.
static inline void tqt_quantize_multiplier_from_scales(float scale_a, float scale_b, float scale_d,
                                                       int32_t *quantized_multiplier,
                                                       int32_t *shift)
{
    double real_scale;
    __asm__ volatile("fmul.d %0, %1, %2, rne\n\tfdiv.d %0, %0, %3, rne"
                     : "=&f"(real_scale)
                     : "f"((double)scale_a), "f"((double)scale_b), "f"((double)scale_d));
    tqt_quantize_multiplier(real_scale, quantized_multiplier, shift);
}
#endif

/// Scale an I32 accumulator by multiplier * 2^(exponent - 31), round once to
/// nearest even, add the output zero point, then saturate. No CSR rounding mode
/// is consumed or changed. Valid quantization scales are finite and nonnegative.
static inline int8_t tqt_requantize_i32_i8(int32_t value, int32_t multiplier, int32_t exponent,
                                           int32_t zero_point)
{
    const int64_t product = (int64_t)value * multiplier;
    const uint64_t magnitude = product < 0 ? (uint64_t)(-product) : (uint64_t)product;
    uint64_t rounded;
    const int64_t right_shift = 31 - (int64_t)exponent;
    if (right_shift >= 64) {
        rounded = 0;
    } else if (right_shift > 0) {
        const uint64_t quotient = magnitude >> right_shift;
        const uint64_t remainder = magnitude & ((UINT64_C(1) << right_shift) - 1);
        const uint64_t half = UINT64_C(1) << (right_shift - 1);
        rounded = quotient + (remainder > half || (remainder == half && (quotient & 1)));
    } else if (magnitude == 0) {
        rounded = 0;
    } else if (right_shift <= -32 || magnitude > (UINT64_C(1) << 32) >> -right_shift) {
        rounded = UINT64_C(1) << 32;
    } else {
        rounded = magnitude << -right_shift;
    }
    // Clamping here also bounds the signed conversion and zero-point addition.
    if (rounded > (UINT64_C(1) << 32))
        rounded = UINT64_C(1) << 32;
    const int64_t scaled = product < 0 ? -(int64_t)rounded : (int64_t)rounded;
    const int64_t result = scaled + zero_point;
    return (int8_t)(result < INT8_MIN ? INT8_MIN : result > INT8_MAX ? INT8_MAX : result);
}

#ifdef __cplusplus
}  // extern "C"
#endif
