//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#ifdef TORQ_TILE_PLATFORM_HEADER
#error Remove TORQ_TILE_PLATFORM_HEADER; provide required runtime symbols when linking instead.
#endif

#if !defined(__STDC_HOSTED__) || (__STDC_HOSTED__ != 0 && __STDC_HOSTED__ != 1)
#error The compiler must describe its execution environment with __STDC_HOSTED__.
#endif

#ifndef TORQ_TILE_FREESTANDING
#if __STDC_HOSTED__
#define TORQ_TILE_FREESTANDING 0
#else
#define TORQ_TILE_FREESTANDING 1
#endif
#endif

// Reject CMake strings such as ON/AUTO rather than silently treating them as zero.
#define TQT_RUNTIME_BOOL_0_VALID 1
#define TQT_RUNTIME_BOOL_1_VALID 1
#define TQT_RUNTIME_VALID_BOOL_IMPL(value) TQT_RUNTIME_BOOL_##value##_VALID
#define TQT_RUNTIME_VALID_BOOL(value) TQT_RUNTIME_VALID_BOOL_IMPL(value)
#if !TQT_RUNTIME_VALID_BOOL(TORQ_TILE_FREESTANDING)
#error TORQ_TILE_FREESTANDING must be the numeric literal 0 or 1.
#endif
#undef TQT_RUNTIME_VALID_BOOL
#undef TQT_RUNTIME_VALID_BOOL_IMPL
#undef TQT_RUNTIME_BOOL_1_VALID
#undef TQT_RUNTIME_BOOL_0_VALID

#if TORQ_TILE_FREESTANDING != (1 - __STDC_HOSTED__)
#error TORQ_TILE_FREESTANDING must match the compiler mode; use -ffreestanding for device code.
#endif

#if TORQ_TILE_FREESTANDING
// Builtins may still emit external calls; the platform supplies their implementations.
#define TQT_MEMCPY __builtin_memcpy
#define TQT_MEMSET __builtin_memset
#else
#include <string.h>
#define TQT_MEMCPY memcpy
#define TQT_MEMSET memset
#endif
