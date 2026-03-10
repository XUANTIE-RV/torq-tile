//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

#if defined(__riscv_xtheadmatrixmin)
#include <thead_matrix.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef __bf16 bfloat16_t;
typedef _Float16 float16_t;

typedef uint8_t float4_e2m1_t;
typedef uint8_t float8_e4m3_t;
typedef uint8_t float8_e5m2_t;
typedef uint8_t float8_e8m0_t;

#ifndef TORQ_TILE_ERROR_TRAP
#define TORQ_TILE_ERROR_TRAP 0
#endif

#ifndef TORQ_TILE_HAS_BUILTIN_UNREACHABLE
#define TORQ_TILE_HAS_BUILTIN_UNREACHABLE 0
#endif

#ifndef TORQ_TILE_HAS_BUILTIN_ASSUME0
#define TORQ_TILE_HAS_BUILTIN_ASSUME0 0
#endif

#if TORQ_TILE_ERROR_TRAP
#define TQT_ABORT() __builtin_trap()
#else
#define TQT_ABORT() abort()
#endif

#if TORQ_TILE_HAS_BUILTIN_UNREACHABLE
#define TQT_UNREACHABLE() __builtin_unreachable()
#elif TORQ_TILE_HAS_BUILTIN_ASSUME0
#define TQT_UNREACHABLE() __assume(0);
#else
#define TQT_UNREACHABLE()
#endif

#ifdef NDEBUG
#define TQT_ERROR(msg)   \
    do {                 \
        TQT_UNUSED(msg); \
        TQT_ABORT();     \
    } while (0)

#define TQT_ASSERT_MSG(cond, msg) \
    do {                          \
        TQT_UNUSED(msg);          \
        if (!(cond)) {            \
            TQT_UNREACHABLE();    \
        }                         \
    } while (0)
#else
#define TQT_ERROR(msg)                                          \
    do {                                                        \
        fflush(stdout);                                         \
        fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, msg); \
        TQT_ABORT();                                            \
    } while (0)

#define TQT_ASSERT_MSG(cond, msg) TQT_ASSERT_ALWAYS_MSG(cond, msg)
#endif  // NDEBUG

#define TQT_ASSERT_ALWAYS_MSG(cond, msg) \
    do {                                 \
        if (!(cond)) {                   \
            TQT_ERROR(msg);              \
        }                                \
    } while (0)

/// TQT_ASSERT* is used for logic sanity checking in the program
/// flow. Checks are optimized away in release builds same as
/// `assert`
#define TQT_ASSERT(cond) TQT_ASSERT_MSG(cond, #cond)
#define TQT_ASSERT_IF_MSG(precond, cond, msg) TQT_ASSERT_MSG(!(precond) || (cond), msg)
#define TQT_ASSERT_IF(precond, cond) TQT_ASSERT_IF_MSG(precond, cond, #precond " |-> " #cond)

/// `TQT_ASSERT_ALWAYS*` is same as `TQT_ASSERT*`, but doesn't get removed by `NDEBUG`
#define TQT_ASSERT_ALWAYS(cond) TQT_ASSERT_ALWAYS_MSG(cond, #cond)
#define TQT_ASSERT_ALWAYS_IF_MSG(precond, cond, msg) \
    TQT_ASSERT_ALWAYS_MSG(!(precond) || (cond), msg)
#define TQT_ASSERT_ALWAYS_IF(precond, cond) \
    TQT_ASSERT_ALWAYS_IF_MSG(precond, cond, #precond " |-> " #cond)

/// TQT_ASSUME* is used for function pre-condition checking, similar to `[[assume]]` in C++23.
/// So TQT_ASSUME should be used directly on the function parameters, rather than inside
/// function logic.
#define TQT_ASSUME_MSG TQT_ASSERT_MSG
#define TQT_ASSUME TQT_ASSERT
#define TQT_ASSUME_IF_MSG TQT_ASSERT_IF_MSG
#define TQT_ASSUME_IF TQT_ASSERT_IF

/// `TQT_ASSUME_ALWAYS*` is same as `TQT_ASSUME*`, but doesn't get removed by `NDEBUG`
#define TQT_ASSUME_ALWAYS_MSG TQT_ASSERT_ALWAYS_MSG
#define TQT_ASSUME_ALWAYS TQT_ASSERT_ALWAYS
#define TQT_ASSUME_ALWAYS_IF_MSG TQT_ASSERT_ALWAYS_IF_MSG
#define TQT_ASSUME_ALWAYS_IF TQT_ASSERT_ALWAYS_IF

/// Indicate that result of `x` is unused
#define TQT_UNUSED(x) (void)(x)

/// Return minimum or maximum of `a` and `b`
#define TQT_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define TQT_MAX(a, b) (((a) > (b)) ? (a) : (b))

#if defined(__riscv_vector)
static inline int csrr_vlenb(void)
{
    int a = 0;
    asm volatile("csrr %0, vlenb" : "=r"(a) : : "memory");
    return a;
}
#endif

#if defined(__riscv_xtheadmatrixmin)
static inline int csrr_xrlen(void)
{
    int a = 0;
    asm volatile("csrr %0, xrlenb" : "=r"(a) : : "memory");
    return a * 8;
}

static inline int csrr_xrlenb(void)
{
    int a = 0;
    asm volatile("csrr %0, xrlenb" : "=r"(a) : : "memory");
    return a;
}
#endif

#ifdef __cplusplus
}
#endif
