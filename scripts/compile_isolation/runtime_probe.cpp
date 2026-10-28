//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

#if defined(__riscv_ztt)
#include "src/gemm/gemm_f32_f32/tqt_kernel_gemm_f32_f32_1nlenxpnlen_ztt.h"
#if !defined(__riscv_zttuds128b) || !defined(__riscv_ztta4)
#define TQT_CHECK_ZTT_16BIT 1
#include "src/gemm/gemm_bf16_bf16/tqt_kernel_gemm_bf16_bf16_1nlenxpnlen_ztt.h"
#include "src/gemm/gemm_f16_f16/tqt_kernel_gemm_f16_f16_1nlenxpnlen_ztt.h"
#endif

static float from_bits(uint32_t bits)
{
    float value;
    TQT_MEMCPY(&value, &bits, sizeof(value));
    return value;
}

static uint32_t to_bits(float value)
{
    uint32_t bits;
    TQT_MEMCPY(&bits, &value, sizeof(bits));
    return bits;
}

template <float (*Min)(float, float), float (*Max)(float, float), float (*MinBits)(float, float),
          float (*MaxBits)(float, float)>
static int check_ztt_number_semantics()
{
    // Ordered values include both zeros, subnormals and infinities.
    const uint32_t ordered[] = {0xff800000U, 0xc0400000U, 0x80800000U, 0x80000001U, 0x80000000U,
                                0x00000000U, 0x00000001U, 0x00800000U, 0x40400000U, 0x7f800000U};
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            const float a = from_bits(ordered[i]), b = from_bits(ordered[j]);
            const uint32_t lo = ordered[i < j ? i : j], hi = ordered[i > j ? i : j];
            if (to_bits(Min(a, b)) != lo)
                return 1;
            if (to_bits(Max(a, b)) != hi)
                return 2;
            if (to_bits(MinBits(a, b)) != lo)
                return 3;
            if (to_bits(MaxBits(a, b)) != hi)
                return 4;
        }
    }
    const uint32_t nan_bits[] = {0x7fc00123U, 0xffc00456U, 0x7f800001U};
    for (uint32_t bits : nan_bits) {
        const float nan = from_bits(bits);
        for (uint32_t number : ordered) {
            const float value = from_bits(number);
            if (to_bits(Min(nan, value)) != number || to_bits(Min(value, nan)) != number ||
                to_bits(Max(nan, value)) != number || to_bits(Max(value, nan)) != number)
                return 5;
            if (to_bits(MinBits(nan, value)) != number || to_bits(MinBits(value, nan)) != number ||
                to_bits(MaxBits(nan, value)) != number || to_bits(MaxBits(value, nan)) != number)
                return 6;
        }
        if ((to_bits(Min(nan, nan)) & 0x7fffffffU) <= 0x7f800000U ||
            (to_bits(Max(nan, nan)) & 0x7fffffffU) <= 0x7f800000U ||
            (to_bits(MinBits(nan, nan)) & 0x7fffffffU) <= 0x7f800000U ||
            (to_bits(MaxBits(nan, nan)) & 0x7fffffffU) <= 0x7f800000U)
            return 7;
    }
    return 0;
}
#endif

#ifndef TQT_CHECK_FAILURE
#define TQT_CHECK_FAILURE 0
#endif

extern "C" int main()
{
#if TQT_CHECK_FAILURE == 1
    TQT_DEBUG_ASSERT(false);
    return 0;
#elif TQT_CHECK_FAILURE == 2
    TQT_ASSERT_ALWAYS(false);
    return 0;
#elif TQT_CHECK_FAILURE == 3
    TQT_ASSERT(false);
    return 0;
#else
    unsigned char src[37], dst[37];
    for (size_t i = 0; i < sizeof(src); ++i) {
        src[i] = (unsigned char)(i + 3);
    }
    volatile size_t count = sizeof(src);
    TQT_MEMSET(dst, 0x45, count);
    for (size_t i = 0; i < sizeof(dst); ++i) {
        if (dst[i] != 0x45)
            return 1;
    }
    TQT_MEMCPY(dst, src, count);
    for (size_t i = 0; i < sizeof(dst); ++i) {
        if (dst[i] != src[i])
            return 2;
    }
    const float input[] = {2, 3, 5};
    const float expected[] = {0, 2, 3, 5, 0};
    float output[5];
    tqt_conv_params params = {{1, 1, 3}, {1, 1, 5}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1},
                              {0, 0, 1}, 1,         1,         1};
    tqt_im2col_cl<float>(0, input, output, &params);
    for (size_t i = 0; i < 5; ++i) {
        if (output[i] != expected[i])
            return 3;
    }
    tqt_im2col_cf<float>(0, input, output, &params);
    for (size_t i = 0; i < 5; ++i) {
        if (output[i] != expected[i])
            return 4;
    }
    int debug_count = 0, condition_count = 0, message_count = 0, always_count = 0;
    TQT_DEBUG_ASSERT(++debug_count == 1);
    TQT_ASSERT_MSG(++condition_count == 1, (++message_count, "message"));
    TQT_ASSERT_ALWAYS(++always_count == 1);
#ifdef NDEBUG
    if (debug_count != 0 || message_count != 1)
        return 5;
#else
    if (debug_count != 1 || message_count != 0)
        return 6;
#endif
    if (condition_count != 1 || always_count != 1)
        return 7;
    if (tqt_clamp_is_enabled(-__builtin_inff(), __builtin_inff()))
        return 8;
    if (!tqt_clamp_is_enabled(-1, 1))
        return 9;
#if defined(__riscv_ztt)
        // Each family retains its own helpers; verify both scalar-F and bitwise paths.
#define TQT_CHECK_ZTT_NUMBERS(NS, BASE)                                                       \
    {                                                                                         \
        const int status =                                                                    \
            check_ztt_number_semantics<NS::min_number, NS::max_number, NS::number_bits<true>, \
                                       NS::number_bits<false>>();                             \
        if (status)                                                                           \
            return BASE + status;                                                             \
    }
    TQT_CHECK_ZTT_NUMBERS(tqt_gemm_f32_f32_1nlenxpnlen_ztt, 10)
#if defined(TQT_CHECK_ZTT_16BIT)
    TQT_CHECK_ZTT_NUMBERS(tqt_gemm_f16_f16_1nlenxpnlen_ztt, 20)
    TQT_CHECK_ZTT_NUMBERS(tqt_gemm_bf16_bf16_1nlenxpnlen_ztt, 30)
#endif
#undef TQT_CHECK_ZTT_NUMBERS
#endif
    return 0;
#endif
}

#undef TQT_CHECK_FAILURE
