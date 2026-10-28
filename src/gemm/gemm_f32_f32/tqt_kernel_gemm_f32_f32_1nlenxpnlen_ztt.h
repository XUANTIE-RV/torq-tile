//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(__riscv) || !defined(__riscv_ztt)
#error This file requires Ztt support.
#endif

#include <riscv_ztt.h>

#include "src/common/tqt_common.h"

// Keep the fixed-register traversal and packing contract in sync with:
// ../gemm_f16_f16/tqt_kernel_gemm_f16_f16_1nlenxpnlen_ztt.h
// ../gemm_bf16_bf16/tqt_kernel_gemm_bf16_bf16_1nlenxpnlen_ztt.h
namespace tqt_gemm_f32_f32_1nlenxpnlen_ztt
{

#if TORQ_TILE_FREESTANDING
// IEEE binary32 number semantics, including signed zero, without scalar F or libm.
template <bool Minimum>
static inline float number_bits(float a, float b)
{
    uint32_t a_bits, b_bits;
    TQT_MEMCPY(&a_bits, &a, sizeof(a));
    TQT_MEMCPY(&b_bits, &b, sizeof(b));
    const bool a_nan = (a_bits & 0x7fffffffU) > 0x7f800000U;
    const bool b_nan = (b_bits & 0x7fffffffU) > 0x7f800000U;
    uint32_t result;
    if (a_nan || b_nan) {
        result = a_nan ? (b_nan ? 0x7fc00000U : b_bits) : a_bits;
    } else {
        const uint32_t a_key = (a_bits & 0x80000000U) ? ~a_bits : (a_bits ^ 0x80000000U);
        const uint32_t b_key = (b_bits & 0x80000000U) ? ~b_bits : (b_bits ^ 0x80000000U);
        result = (Minimum ? a_key < b_key : a_key > b_key) ? a_bits : b_bits;
    }
    float value;
    TQT_MEMCPY(&value, &result, sizeof(value));
    return value;
}

static inline float min_number(float a, float b)
{
#if defined(__riscv_flen) && __riscv_flen >= 32
    float value;
    __asm__ volatile("fmin.s %0, %1, %2" : "=f"(value) : "f"(a), "f"(b));
    return value;
#else
    return number_bits<true>(a, b);
#endif
}

static inline float max_number(float a, float b)
{
#if defined(__riscv_flen) && __riscv_flen >= 32
    float value;
    __asm__ volatile("fmax.s %0, %1, %2" : "=f"(value) : "f"(a), "f"(b));
    return value;
#else
    return number_bits<false>(a, b);
#endif
}
#else
static inline float min_number(float a, float b)
{
    return fminf(a, b);
}

static inline float max_number(float a, float b)
{
    return fmaxf(a, b);
}
#endif

using T = float;
constexpr uint32_t dtype = 0x20300120;

#if defined(__riscv_zttuds8b)
constexpr size_t uds = 8;
#elif defined(__riscv_zttuds16b)
constexpr size_t uds = 16;
#elif defined(__riscv_zttuds32b)
constexpr size_t uds = 32;
#elif defined(__riscv_zttuds64b)
constexpr size_t uds = 64;
#elif defined(__riscv_zttuds128b)
constexpr size_t uds = 128;
#else
#error Unsupported Ztt UDS profile.
#endif

#if defined(__riscv_ztta4)
constexpr size_t acc_count = 4;
#elif defined(__riscv_ztta8)
constexpr size_t acc_count = 8;
#elif defined(__riscv_ztta16)
constexpr size_t acc_count = 16;
#else
#error Unsupported Ztt accumulator profile.
#endif

static inline size_t checked_mul(size_t a, size_t b)
{
    TQT_ASSUME_ALWAYS_MSG(b == 0 || a <= SIZE_MAX / b, "Ztt buffer size overflow");
    return a * b;
}

static inline size_t checked_add(size_t a, size_t b)
{
    TQT_ASSUME_ALWAYS_MSG(a <= SIZE_MAX - b, "Ztt offset overflow");
    return a + b;
}

static inline size_t round_up(size_t value, size_t step)
{
    return checked_add(value, step - 1) / step * step;
}

struct geometry
{
    static constexpr size_t p = uds > sizeof(T) * 8 ? uds / (sizeof(T) * 8) : 1;
    static_assert(p <= acc_count, "Ztt adaptive GEMM needs P accumulator registers");

    static size_t m_step()
    {
        const size_t s = tqt_get_amenlen();
        TQT_ASSUME_ALWAYS_MSG(s != 0 && (s & (s - 1)) == 0, "Invalid Ztt square size");
        TQT_ASSUME_ALWAYS_MSG(tqt_get_ameudsz() == uds, "Ztt binary and hardware UDS must match");
        checked_mul(checked_mul(checked_mul(s, s), p), sizeof(float));
        return s;
    }

    static size_t n_step() { return checked_mul(m_step(), p); }

    static size_t size(size_t rows, size_t cols, bool b = false)
    {
        if (rows == 0 || cols == 0) {
            return 0;
        }
        const size_t s = m_step();
        return checked_mul(checked_mul(round_up(rows, b ? s * p : s), round_up(cols, s * p)),
                           sizeof(T));
    }

    static size_t offset(size_t row, size_t col, size_t ld)
    {
        return checked_mul(checked_add(checked_mul(row, ld), col), sizeof(T));
    }
};

// A/C/D: block-row-major. B: block-column-major. Every square is row-major.
static inline size_t packed_index(size_t i, size_t j, size_t padded_rows, size_t padded_cols,
                                  size_t s, bool b)
{
    const size_t block =
        b ? (j / s) * (padded_rows / s) + i / s : (i / s) * (padded_cols / s) + j / s;
    return block * s * s + (i % s) * s + j % s;
}

template <bool B = false, bool Transposed = false>
static inline void pack(size_t rows, size_t cols, size_t ld, const void *src, void *dst)
{
    if (rows == 0 || cols == 0) {
        return;
    }
    TQT_ASSUME_ALWAYS(src != nullptr && dst != nullptr && src != dst);
    TQT_ASSUME_ALWAYS(ld >= (Transposed ? rows : cols));
    checked_mul(checked_mul(Transposed ? cols : rows, ld), sizeof(T));
    const size_t bytes = geometry::size(rows, cols, B);
    const size_t s = geometry::m_step(), step = s * geometry::p;
    const size_t pr = round_up(rows, B ? step : s), pc = round_up(cols, step);
    TQT_MEMSET(dst, 0, bytes);
    const T *input = static_cast<const T *>(src);
    T *output = static_cast<T *>(dst);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            output[packed_index(i, j, pr, pc, s, B)] = input[Transposed ? j * ld + i : i * ld + j];
        }
    }
}

static inline void unpack(size_t m, size_t n, size_t ld, const void *src, void *dst)
{
    if (m == 0 || n == 0) {
        return;
    }
    TQT_ASSUME_ALWAYS(src != nullptr && dst != nullptr && src != dst && ld >= n);
    checked_mul(checked_mul(m, ld), sizeof(T));
    geometry::size(m, n);
    const size_t s = geometry::m_step(), pn = round_up(n, s * geometry::p);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            static_cast<T *>(dst)[i * ld + j] =
                static_cast<const T *>(src)[packed_index(i, j, 0, pn, s, false)];
        }
    }
}

struct args
{
    size_t m, n, k, s;
    const T *a;
    const T *b;
    const T *c;
    T *d;
    const T *bias;
    size_t lda, ldb, ldc, ldd;
};

// These scalar-only helpers are always inlined into
// tqt_squares_kernel_gemm_f32_f32_1nlenxpnlen_ztt(). No compiler-managed M/ACC value may coexist
// with the fixed-register assembly. The current Clang does not accept M/ACC clobber names;
// isolation at the noinline, call-clobbered ABI boundary is essential. All memory effects are
// visible to the compiler.
template <unsigned Slot, bool Packed, bool TransposedB>
__attribute__((always_inline)) static inline void tqt_multiply_kernel_gemm_f32_f32_1nlenxpnlen_ztt(
    const T *b, size_t stride)
{
    if constexpr (Packed) {
        asm volatile("mls.rm m4, %0" : : "r"(b) : "memory");
    } else if constexpr (TransposedB) {
        asm volatile("mls.st m4, (%0), %1" : : "r"(b), "r"(stride) : "memory");
    } else {
        asm volatile("mls.tst m4, (%0), %1" : : "r"(b), "r"(stride) : "memory");
    }
#define TQT_ZTT_MAC_SLOT(I)                                                \
    if constexpr (Slot == I) {                                             \
        if constexpr (Packed) {                                            \
            asm volatile("mmulacc.2d acc" #I ", m0, m4" : : : "memory");   \
        } else {                                                           \
            asm volatile("mmulbtacc.2d acc" #I ", m0, m4" : : : "memory"); \
        }                                                                  \
    }
    TQT_ZTT_MAC_SLOT(0)
    TQT_ZTT_MAC_SLOT(1)
    TQT_ZTT_MAC_SLOT(2)
    TQT_ZTT_MAC_SLOT(3)
    TQT_ZTT_MAC_SLOT(4)
    TQT_ZTT_MAC_SLOT(5)
    TQT_ZTT_MAC_SLOT(6)
    TQT_ZTT_MAC_SLOT(7)
#undef TQT_ZTT_MAC_SLOT
}

template <bool Wide, bool Packed, bool TransposedB>
__attribute__((noinline)) static void tqt_squares_kernel_gemm_f32_f32_1nlenxpnlen_ztt(
    const args *input)
{
    constexpr size_t p = geometry::p;
    constexpr size_t abits = Wide ? 32 : sizeof(T) * 8;
    constexpr size_t ap = uds > abits ? uds / abits : 1;
    constexpr uint32_t adtype = Wide ? 0x20300120 : dtype;
    const auto &v = *input;
    for (size_t i = 0; i < v.m; i += v.s) {
        for (size_t j = 0; j < v.n; j += p * v.s) {
            asm volatile(
                "msettyp m0, %0\n\tmsettyp m4, %0\n\tmsettyp m8, %1\n\t"
                "asettyp acc0, %1"
                :
                : "r"(dtype), "r"(adtype)
                : "memory");
            if constexpr (Wide && p > ap) {
                asm volatile("msettyp m9, %0" : : "r"(adtype) : "memory");
            }
            if constexpr (p >= 2) {
                asm volatile("asettyp acc1, %0" : : "r"(adtype) : "memory");
            }
            if constexpr (p >= 4) {
                asm volatile("asettyp acc2, %0\n\tasettyp acc3, %0" : : "r"(adtype) : "memory");
            }
            if constexpr (p >= 8) {
                asm volatile(
                    "asettyp acc4, %0\n\tasettyp acc5, %0\n\t"
                    "asettyp acc6, %0\n\tasettyp acc7, %0"
                    :
                    : "r"(adtype)
                    : "memory");
            }
            if (v.c != nullptr) {
                if constexpr (Packed) {
                    asm volatile("mls.rm m0, %0" : : "r"(v.c + i * v.n + j * v.s) : "memory");
                } else {
                    asm volatile("mls.st m0, (%0), %1"
                                 :
                                 : "r"(v.c + i * v.ldc + j), "r"(v.ldc * sizeof(T))
                                 : "memory");
                }
                if constexpr (Wide) {
                    asm volatile("mconv.ew m8, m0" : : : "memory");
                } else {
                    asm volatile("mmov.m.m m8, m0" : : : "memory");
                }
                asm volatile("mmov.a.m acc0, m8" : : : "memory");
                if constexpr (Wide && p > ap) {
                    if constexpr (ap == 1) {
                        asm volatile("mmov.a.m acc1, m9" : : : "memory");
                    } else if constexpr (ap == 2) {
                        asm volatile("mmov.a.m acc2, m9" : : : "memory");
                    } else {
                        asm volatile("mmov.a.m acc4, m9" : : : "memory");
                    }
                }
            }
            for (size_t k = 0; k < v.k; k += p * v.s) {
                if constexpr (Packed) {
                    asm volatile("mls.rm m0, %0" : : "r"(v.a + i * v.k + k * v.s) : "memory");
                } else {
                    asm volatile("mls.st m0, (%0), %1"
                                 :
                                 : "r"(v.a + i * v.lda + k), "r"(v.lda * sizeof(T))
                                 : "memory");
                }
#define TQT_ZTT_MULTIPLY(I)                                                              \
    if constexpr (p > I) {                                                               \
        const size_t col = j + I * v.s;                                                  \
        const T *b = v.b + (Packed ? col * v.k + k * v.s                                 \
                                   : (TransposedB ? col * v.ldb + k : k * v.ldb + col)); \
        tqt_multiply_kernel_gemm_f32_f32_1nlenxpnlen_ztt<I, Packed, TransposedB>(        \
            b, v.ldb * sizeof(T));                                                       \
    }
                TQT_ZTT_MULTIPLY(0)
                TQT_ZTT_MULTIPLY(1)
                TQT_ZTT_MULTIPLY(2)
                TQT_ZTT_MULTIPLY(3)
                TQT_ZTT_MULTIPLY(4)
                TQT_ZTT_MULTIPLY(5)
                TQT_ZTT_MULTIPLY(6)
                TQT_ZTT_MULTIPLY(7)
#undef TQT_ZTT_MULTIPLY
            }
            asm volatile("mmov.m.a m8, acc0" : : : "memory");
            if constexpr (Wide && p > ap) {
                if constexpr (ap == 1) {
                    asm volatile("mmov.m.a m9, acc1" : : : "memory");
                } else if constexpr (ap == 2) {
                    asm volatile("mmov.m.a m9, acc2" : : : "memory");
                } else {
                    asm volatile("mmov.m.a m9, acc4" : : : "memory");
                }
            }
            if (v.bias != nullptr) {
                asm volatile("mls.st m0, (%0), zero" : : "r"(v.bias + j) : "memory");
                if constexpr (Wide) {
                    asm volatile("msettyp m4, %0" : : "r"(adtype) : "memory");
                    if constexpr (p > ap) {
                        asm volatile("msettyp m5, %0" : : "r"(adtype) : "memory");
                    }
                    asm volatile("mconv.ew m4, m0\n\tmadd.ew m8, m8, m4" : : : "memory");
                    if constexpr (p > ap) {
                        asm volatile("madd.ew m9, m9, m5" : : : "memory");
                    }
                } else {
                    asm volatile("madd.ew m8, m8, m0" : : : "memory");
                }
            }
            T *d = v.d + (Packed ? i * v.n + j * v.s : i * v.ldd + j);
            if constexpr (Wide) {
                asm volatile("mconv.ew m0, m8" : : : "memory");
                if constexpr (Packed) {
                    asm volatile("mss.rm m0, %0" : : "r"(d) : "memory");
                } else {
                    asm volatile("mss.st m0, (%0), %1"
                                 :
                                 : "r"(d), "r"(v.ldd * sizeof(T))
                                 : "memory");
                }
            } else {
                if constexpr (Packed) {
                    asm volatile("mss.rm m8, %0" : : "r"(d) : "memory");
                } else {
                    asm volatile("mss.st m8, (%0), %1"
                                 :
                                 : "r"(d), "r"(v.ldd * sizeof(T))
                                 : "memory");
                }
            }
        }
    }
}

template <bool Wide, bool Packed, bool TransposedB>
static inline void tqt_run_kernel_gemm_f32_f32_1nlenxpnlen_ztt(args v, T lower, T upper)
{
    if (v.m == 0 || v.n == 0) {
        return;
    }
    v.s = geometry::m_step();
    const size_t step = v.s * geometry::p;
    TQT_ASSUME_ALWAYS(v.d != nullptr && (v.k == 0 || (v.a != nullptr && v.b != nullptr)));
    if constexpr (Packed) {
        geometry::size(v.m, v.k);
        geometry::size(v.k, v.n, true);
        geometry::size(v.m, v.n);
        v.m = round_up(v.m, v.s);
        v.n = round_up(v.n, step);
        v.k = round_up(v.k, step);
    } else {
        TQT_ASSUME_ALWAYS_MSG(v.m % v.s == 0 && v.n % step == 0 && v.k % step == 0,
                              "Ztt nonpacked GEMM requires aligned M, N and K");
        TQT_ASSUME_ALWAYS(v.ldd >= v.n && (v.c == nullptr || v.ldc >= v.n));
        TQT_ASSUME_ALWAYS(v.c != v.d || v.ldc == v.ldd);
        checked_mul(checked_mul(v.m, v.ldd), sizeof(T));
        if (v.c != nullptr) {
            checked_mul(checked_mul(v.m, v.ldc), sizeof(T));
        }
        if (v.k != 0) {
            TQT_ASSUME_ALWAYS(v.lda >= v.k && v.ldb >= (TransposedB ? v.k : v.n));
            checked_mul(checked_mul(v.m, v.lda), sizeof(T));
            checked_mul(checked_mul(TransposedB ? v.n : v.k, v.ldb), sizeof(T));
        }
    }
    const bool acquired = (__riscv_ztt_get_ameown() & 1) == 0;
    if (acquired) {
        TQT_ASSUME_ALWAYS_MSG((__riscv_ztt_ame_acquire(0) & 1) != 0,
                              "Unable to acquire Ztt backend");
    } else {
        TQT_ASSUME_ALWAYS_MSG((__riscv_ztt_get_amestatus() & 1) == 0,
                              "Pre-existing Ztt unsupported status");
    }
    tqt_squares_kernel_gemm_f32_f32_1nlenxpnlen_ztt<Wide, Packed, TransposedB>(&v);
    const size_t status = __riscv_ztt_get_amestatus();
    if (acquired) {
        __riscv_ztt_ame_release();
    }
    TQT_ASSUME_ALWAYS_MSG((status & 1) == 0, "Unsupported Ztt operation/datatype tuple");
    const float lo = static_cast<float>(lower), hi = static_cast<float>(upper);
    if (tqt_clamp_is_enabled(lo, hi)) {
        for (size_t i = 0; i < v.m; ++i) {
            for (size_t j = 0; j < v.n; ++j) {
                T &x = v.d[Packed ? i * v.n + j : i * v.ldd + j];
                x = static_cast<T>(min_number(max_number(static_cast<float>(x), lo), hi));
            }
        }
    }
}

}  // namespace tqt_gemm_f32_f32_1nlenxpnlen_ztt
