//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <stddef.h>
#include <stdint.h>

// Test-platform implementations, deliberately independent of compiler loop idioms.
void *memcpy(void *dst, const void *src, size_t size)
{
    volatile unsigned char *d = (volatile unsigned char *)dst;
    const volatile unsigned char *s = (const volatile unsigned char *)src;
    for (size_t i = 0; i < size; ++i) {
        d[i] = s[i];
    }
    return dst;
}

void *memset(void *dst, int value, size_t size)
{
    volatile unsigned char *d = (volatile unsigned char *)dst;
    for (size_t i = 0; i < size; ++i) {
        d[i] = (unsigned char)value;
    }
    return dst;
}

void *memmove(void *dst, const void *src, size_t size)
{
    volatile unsigned char *d = (volatile unsigned char *)dst;
    const volatile unsigned char *s = (const volatile unsigned char *)src;
    if ((uintptr_t)dst < (uintptr_t)src) {
        for (size_t i = 0; i < size; ++i) {
            d[i] = s[i];
        }
    } else {
        for (size_t i = size; i > 0; --i) {
            d[i - 1] = s[i - 1];
        }
    }
    return dst;
}

int memcmp(const void *a, const void *b, size_t size)
{
    const unsigned char *lhs = (const unsigned char *)a;
    const unsigned char *rhs = (const unsigned char *)b;
    for (size_t i = 0; i < size; ++i) {
        if (lhs[i] != rhs[i]) {
            return lhs[i] < rhs[i] ? -1 : 1;
        }
    }
    return 0;
}
