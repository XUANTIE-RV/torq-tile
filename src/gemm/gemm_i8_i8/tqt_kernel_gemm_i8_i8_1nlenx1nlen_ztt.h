//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(__riscv) || !defined(__riscv_ztt) || !defined(__riscv_flen) || __riscv_flen < 32
#error This file requires Ztt and scalar F support.
#endif

#if (!defined(__riscv_zttm16) && !defined(__riscv_zttm32)) || \
    (!defined(__riscv_ztta4) && !defined(__riscv_ztta8) && !defined(__riscv_ztta16))
#error This file requires M16/M32 and A4/A8/A16 register profiles.
#endif

#include <riscv_ztt.h>

#include "src/common/tqt_common.h"

// Keep geometry, packing and fixed-register traversal in sync with:
// ../gemm_i8_u4/tqt_kernel_gemm_i8_u4_1nlenx1nlen_ztt.h
namespace tqt_gemm_i8_i8_1nlenx1nlen_ztt
{

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

constexpr bool asymmetric = false;
constexpr size_t bits = asymmetric ? 4 : 8;
constexpr size_t group = asymmetric ? 32 : 16;
constexpr size_t p32 = uds > 32 ? uds / 32 : 1;
constexpr uintptr_t i8_type = 0x40000008;
constexpr uintptr_t b_type = 0x40000000 | bits;
constexpr uintptr_t i32_type = 0x40000020;
constexpr uintptr_t f32_type = 0x20300120;

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

static inline size_t round_up(size_t a, size_t b)
{
    return checked_mul(checked_add(a, b - 1) / b, b);
}

struct geometry
{
    static size_t step()
    {
        const size_t s = tqt_get_amenlen();
        TQT_ASSUME_ALWAYS_MSG(s != 0 && s <= 128 && (s & (s - 1)) == 0,
                              "Ztt quantized GEMM requires power-of-two NLEN <= 128");
        TQT_ASSUME_ALWAYS_MSG(tqt_get_ameudsz() == uds, "Ztt binary and hardware UDS must match");
        return s;
    }

    static size_t padded_k(size_t k) { return round_up(k, uds / bits * step()); }

    static size_t size(size_t rows, size_t k, bool b)
    {
        if (rows == 0 || k == 0)
            return 0;
        TQT_ASSUME_ALWAYS(k % 256 == 0);
        return checked_mul(round_up(rows, step()), padded_k(k)) / (b && asymmetric ? 2 : 1);
    }

    static size_t offset(size_t row, size_t k, size_t ld, bool b)
    {
        const size_t s = step();
        TQT_ASSUME_ALWAYS(row % s == 0 && k % 256 == 0 && ld % 256 == 0 && k <= ld);
        return checked_add(checked_mul(row, padded_k(ld)), checked_mul(k, s)) /
               (b && asymmetric ? 2 : 1);
    }
};

// Input pointers and destination pointers are already advanced to k_idx. Only the
// requested K interval and final padding are written, even inside a shared panel.
template <bool B>
static inline void pack(size_t rows, size_t k, size_t ld, size_t packed_ld, size_t k_idx,
                        const void *src, void *dst)
{
    if (rows == 0 || k == 0)
        return;
    const size_t end = checked_add(k_idx, k);
    TQT_ASSUME_ALWAYS(k % 256 == 0 && k_idx % 256 == 0 && packed_ld % 256 == 0);
    TQT_ASSUME_ALWAYS(src != nullptr && dst != nullptr && src != dst);
    TQT_ASSUME_ALWAYS(ld >= k && packed_ld >= end);
    if constexpr (B && asymmetric)
        TQT_ASSUME_ALWAYS(ld % 2 == 0);
    const size_t s = geometry::step(), pk = geometry::padded_k(packed_ld);
    geometry::size(rows, packed_ld, B);
    checked_mul(rows, ld);
    const size_t stop = end == packed_ld ? pk : end;
    const uint8_t *input = static_cast<const uint8_t *>(src);
    uint8_t *output = static_cast<uint8_t *>(dst);
    for (size_t i = 0; i < round_up(rows, s); ++i) {
        for (size_t q = k_idx; q < stop; ++q) {
            const size_t index = (i / s) * s * pk + (q / s) * s * s + (i % s) * s + q % s;
            const size_t local = index - k_idx * s;
            if constexpr (B && asymmetric) {
                uint8_t value = 0;
                if (i < rows && q < end) {
                    const size_t source = i * ld + q - k_idx;
                    value = ((input[source / 2] >> ((source % 2) * 4)) & 15) ^ 8;
                }
                const size_t shift = (local % 2) * 4;
                output[local / 2] = (output[local / 2] & ~(15U << shift)) | (value << shift);
            } else {
                output[local] = i < rows && q < end ? input[i * ld + q - k_idx] : 0;
            }
        }
    }
}

struct args
{
    size_t m, n, k, k_idx, lda, ldb;
    const uint8_t *a;
    const uint8_t *b;
    const float *sa;
    size_t ldsa;
    const int16_t *sums;
    size_t ldsums;
    const float *sb;
    const float *zb;
    size_t ldsb;
    const float *c;
    size_t ldc;
    float *d;
    size_t ldd;
    const float *bias;
};

// Scalar-only ABI boundary: no compiler-managed M/ACC values or calls are allowed
// while fixed registers are live. Current Clang does not accept M/ACC clobbers.
// m0/m1: retained A; m2: retained B; m4..m7: masks then FP32 term;
// m8..m11: I32 dot then metadata; m12..m15: running FP32 output.
// Volatile metadata vectors prevent loop-to-libcall lowering inside this region.
template <size_t S>
__attribute__((noinline)) static void tqt_squares_kernel_gemm_i8_i8_1nlenx1nlen_ztt(const args &v)
{
    constexpr size_t h = uds / bits * S;
    constexpr size_t a_span = uds / 8 * S * S;
    constexpr size_t scratch_size = S * S + (p32 - 1) * S;
    int8_t mask[2 * h + group];
    float output[scratch_size];
    volatile float sa[S * p32], sb[S * p32], sum_sa[S * p32], zb[S * p32];
    volatile int32_t correction[S * p32];
    if constexpr (h > group) {
        for (size_t q = 0; q < 2 * h + group; ++q)
            mask[q] = q >= h && q < h + group ? -1 : 0;
    }
    for (size_t i = 0; i < v.m; i += S) {
        const size_t mr = v.m - i < S ? v.m - i : S;
        for (size_t j = 0; j < v.n; j += S) {
            const size_t nr = v.n - j < S ? v.n - j : S;
            asm volatile("msettyp m12, %0" : : "r"(f32_type) : "memory");
            size_t loaded_panel = SIZE_MAX;
            for (size_t g = v.k_idx; g < v.k_idx + v.k; g += group) {
                asm volatile("asettyp acc0, %0" : : "r"(i32_type) : "memory");
                if constexpr (p32 >= 2)
                    asm volatile("asettyp acc1, %0" : : "r"(i32_type) : "memory");
                if constexpr (p32 >= 4)
                    asm volatile("asettyp acc2, %0\n\tasettyp acc3, %0"
                                 :
                                 : "r"(i32_type)
                                 : "memory");
                // H <= group: accumulate whole panels without masks. Otherwise
                // preserve a panel across groups and mask only the A operand.
                for (size_t q = g; q < g + group; q += h) {
                    const size_t panel = q / h * h;
                    if (panel != loaded_panel) {
                        const uint8_t *ap = v.a - v.k_idx * S + i * v.lda + panel * S;
                        const uint8_t *bp = v.b - v.k_idx * S * bits / 8 + j * v.ldb * bits / 8 +
                                            panel * S * bits / 8;
                        asm volatile(
                            "msettyp m0, %0\n\tmsettyp m1, %0\n\tmsettyp m2, %1\n\t"
                            "mls.rm m0, %2\n\tmls.rm m2, %3"
                            :
                            : "r"(i8_type), "r"(b_type), "r"(ap), "r"(bp)
                            : "memory");
                        if constexpr (asymmetric)
                            asm volatile("mls.rm m1, %0" : : "r"(ap + a_span) : "memory");
                        loaded_panel = panel;
                    }
                    if constexpr (h > group) {
                        asm volatile(
                            "msettyp m4, %0\n\tmsettyp m5, %0\n\t"
                            "msettyp m6, %0\n\tmsettyp m7, %0\n\t"
                            "mls.st m6, (%1), zero\n\tmand.ew m4, m0, m6"
                            :
                            : "r"(i8_type), "r"(mask + h - (g - panel))
                            : "memory");
                        if constexpr (asymmetric)
                            asm volatile("mls.st m7, (%0), zero\n\tmand.ew m5, m1, m7"
                                         :
                                         : "r"(mask + h - (g - panel) + uds / 8 * S)
                                         : "memory");
                        asm volatile("mmulbtacc.2d acc0, m4, m2" : : : "memory");
                    } else {
                        asm volatile("mmulbtacc.2d acc0, m0, m2" : : : "memory");
                    }
                }
                for (size_t r = 0; r < S * p32; ++r) {
                    sa[r] = r < mr ? v.sa[(i + r) * v.ldsa + g / 256] : 0.0f;
                    sb[r] = r < nr ? v.sb[(j + r) * v.ldsb + g / group] : 0.0f;
                    if constexpr (asymmetric) {
                        const int32_t sum = r < mr ? (int32_t)v.sums[(i + r) * v.ldsums + g / 16] +
                                                         v.sums[(i + r) * v.ldsums + g / 16 + 1]
                                                   : 0;
                        correction[r] = 8 * sum;
                        sum_sa[r] = (float)sum * sa[r];
                        zb[r] = r < nr ? v.zb[(j + r) * v.ldsb + g / group] : 0.0f;
                    }
                }
                asm volatile("msettyp m8, %0\n\tmmov.m.a m8, acc0" : : "r"(i32_type) : "memory");
                if constexpr (asymmetric) {
                    asm volatile("msettyp m4, %0\n\tmls.tst m4, (%1), zero\n\tmadd.ew m8, m8, m4"
                                 :
                                 : "r"(i32_type), "r"(correction)
                                 : "memory");
                }
                asm volatile(
                    "msettyp m4, %0\n\tmconv.ew m4, m8\n\tmsettyp m8, %0\n\t"
                    "mls.st m8, (%1), zero\n\tmmul.ew m4, m4, m8\n\t"
                    "mls.tst m8, (%2), zero\n\tmmul.ew m4, m4, m8\n\tmadd.ew m12, m12, m4"
                    :
                    : "r"(f32_type), "r"(sb), "r"(sa)
                    : "memory");
                if constexpr (asymmetric) {
                    // Ztt subtraction computes source 2 minus source 1.
                    asm volatile(
                        "mls.tst m4, (%0), zero\n\tmls.st m8, (%1), zero\n\t"
                        "mmul.ew m4, m4, m8\n\tmsub.ew m12, m4, m12"
                        :
                        : "r"(sum_sa), "r"(zb)
                        : "memory");
                }
            }
            // Wide UDS stores have overlapping row strides; extra lanes land in
            // scratch only. The first S columns of every row are the valid tile.
            asm volatile("mss.st m12, (%0), %1" : : "r"(output), "r"(S * sizeof(float)) : "memory");
            for (size_t r = 0; r < mr; ++r) {
                for (size_t c = 0; c < nr; ++c) {
                    float value = output[r * S + c];
                    if (v.c)
                        value += v.c[(i + r) * v.ldc + j + c];
                    if (v.bias)
                        value += v.bias[j + c];
                    v.d[(i + r) * v.ldd + j + c] = value;
                }
            }
        }
    }
}

static inline void run(args v, float lower, float upper)
{
    if (v.m == 0 || v.n == 0)
        return;
    const size_t s = geometry::step();
    TQT_ASSUME_ALWAYS(v.d != nullptr && v.ldd >= v.n && (!v.c || v.ldc >= v.n));
    TQT_ASSUME_ALWAYS(v.c != v.d || v.ldc == v.ldd);
    TQT_ASSUME_ALWAYS(v.k % 256 == 0 && v.k_idx % 256 == 0);
    const size_t end = checked_add(v.k_idx, v.k);
    checked_mul(checked_mul(v.m, v.ldd), sizeof(float));
    if (v.c)
        checked_mul(checked_mul(v.m, v.ldc), sizeof(float));
    if (v.k != 0) {
        TQT_ASSUME_ALWAYS(v.a && v.b && v.sa && v.sb);
        TQT_ASSUME_ALWAYS(v.lda >= end && v.ldb >= end);
        TQT_ASSUME_ALWAYS(v.lda % 256 == 0 && v.ldb % 256 == 0);
        TQT_ASSUME_ALWAYS(v.ldsa >= end / 256 && v.ldsb >= end / group);
        if constexpr (asymmetric)
            TQT_ASSUME_ALWAYS(v.sums && v.zb && v.ldsums >= end / 16);
        checked_mul(checked_mul(v.m, v.ldsa), sizeof(float));
        checked_mul(checked_mul(v.n, v.ldsb), sizeof(float));
        if constexpr (asymmetric)
            checked_mul(checked_mul(v.m, v.ldsums), sizeof(int16_t));
        geometry::size(v.m, v.lda, false);
        geometry::size(v.n, v.ldb, true);
        v.lda = geometry::padded_k(v.lda);
        v.ldb = geometry::padded_k(v.ldb);
    }
    const bool acquired = (__riscv_ztt_get_ameown() & 1) == 0;
    if (acquired) {
        TQT_ASSUME_ALWAYS_MSG((__riscv_ztt_ame_acquire(0) & 1) != 0,
                              "Unable to acquire Ztt backend");
    } else {
        TQT_ASSUME_ALWAYS_MSG((__riscv_ztt_get_amestatus() & 1) == 0,
                              "Pre-existing Ztt unsupported status");
    }
#define TQT_ZTT_QUANT_DISPATCH(S)                            \
    case S:                                                  \
        tqt_squares_kernel_gemm_i8_i8_1nlenx1nlen_ztt<S>(v); \
        break
    switch (s) {
        TQT_ZTT_QUANT_DISPATCH(1);
        TQT_ZTT_QUANT_DISPATCH(2);
        TQT_ZTT_QUANT_DISPATCH(4);
        TQT_ZTT_QUANT_DISPATCH(8);
        TQT_ZTT_QUANT_DISPATCH(16);
        TQT_ZTT_QUANT_DISPATCH(32);
        TQT_ZTT_QUANT_DISPATCH(64);
        TQT_ZTT_QUANT_DISPATCH(128);
    }
#undef TQT_ZTT_QUANT_DISPATCH
    const size_t status = __riscv_ztt_get_amestatus();
    if (acquired)
        __riscv_ztt_ame_release();
    TQT_ASSUME_ALWAYS_MSG((status & 1) == 0, "Unsupported Ztt operation/datatype tuple");
    if (tqt_clamp_is_enabled(lower, upper)) {
        for (size_t i = 0; i < v.m; ++i) {
            for (size_t j = 0; j < v.n; ++j) {
                float &value = v.d[i * v.ldd + j];
                asm volatile("fmax.s %0, %0, %1\n\tfmin.s %0, %0, %2"
                             : "+f"(value)
                             : "f"(lower), "f"(upper));
            }
        }
    }
}

}  // namespace tqt_gemm_i8_i8_1nlenx1nlen_ztt
