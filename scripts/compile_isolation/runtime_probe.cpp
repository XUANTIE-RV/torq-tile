//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

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
    return 0;
#endif
}

#undef TQT_CHECK_FAILURE
