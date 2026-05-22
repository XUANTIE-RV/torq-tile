//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "src/common/tqt_common.h"

// ============================================================================
// Internal Pack Kernel Templates
// ============================================================================

/// Pack kernel for A matrix: [m, k] -> [m/MR, k, MR]
/// Uses vsseg instructions to interleave MR rows
/// MR: number of rows (2-8)
template <size_t MR>
static inline void tqt_gemm_pack_mxk_e32_kernel(size_t k, const float *src, size_t lda, float *dst)
{
    static_assert(MR >= 2 && MR <= 8, "MR must be in range [2, 8]");

    size_t size = k;
    while (size > 0) {
        size_t vl = __riscv_vsetvl_e32m1(size);

        vfloat32m1_t v0, v1, v2, v3, v4, v5, v6, v7;

        // Load MR rows
        if constexpr (MR >= 1)
            v0 = __riscv_vle32_v_f32m1(src + 0 * lda, vl);
        if constexpr (MR >= 2)
            v1 = __riscv_vle32_v_f32m1(src + 1 * lda, vl);
        if constexpr (MR >= 3)
            v2 = __riscv_vle32_v_f32m1(src + 2 * lda, vl);
        if constexpr (MR >= 4)
            v3 = __riscv_vle32_v_f32m1(src + 3 * lda, vl);
        if constexpr (MR >= 5)
            v4 = __riscv_vle32_v_f32m1(src + 4 * lda, vl);
        if constexpr (MR >= 6)
            v5 = __riscv_vle32_v_f32m1(src + 5 * lda, vl);
        if constexpr (MR >= 7)
            v6 = __riscv_vle32_v_f32m1(src + 6 * lda, vl);
        if constexpr (MR >= 8)
            v7 = __riscv_vle32_v_f32m1(src + 7 * lda, vl);

        // Store with vsseg (interleaved)
        if constexpr (MR == 2) {
            vfloat32m1x2_t src = __riscv_vcreate_v_f32m1x2(v0, v1);
            __riscv_vsseg2e32_v_f32m1x2(dst, src, vl);
        } else if constexpr (MR == 3) {
            vfloat32m1x3_t src = __riscv_vcreate_v_f32m1x3(v0, v1, v2);
            __riscv_vsseg3e32_v_f32m1x3(dst, src, vl);
        } else if constexpr (MR == 4) {
            vfloat32m1x4_t src = __riscv_vcreate_v_f32m1x4(v0, v1, v2, v3);
            __riscv_vsseg4e32_v_f32m1x4(dst, src, vl);
        } else if constexpr (MR == 5) {
            vfloat32m1x5_t src = __riscv_vcreate_v_f32m1x5(v0, v1, v2, v3, v4);
            __riscv_vsseg5e32_v_f32m1x5(dst, src, vl);
        } else if constexpr (MR == 6) {
            vfloat32m1x6_t src = __riscv_vcreate_v_f32m1x6(v0, v1, v2, v3, v4, v5);
            __riscv_vsseg6e32_v_f32m1x6(dst, src, vl);
        } else if constexpr (MR == 7) {
            vfloat32m1x7_t src = __riscv_vcreate_v_f32m1x7(v0, v1, v2, v3, v4, v5, v6);
            __riscv_vsseg7e32_v_f32m1x7(dst, src, vl);
        } else if constexpr (MR == 8) {
            vfloat32m1x8_t src = __riscv_vcreate_v_f32m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
            __riscv_vsseg8e32_v_f32m1x8(dst, src, vl);
        }

        src += vl;
        dst += vl * MR;
        size -= vl;
    }
}

/// Pack kernel for single row A matrix: [1, k] -> [1, k]
/// Special case for MR = 1, no interleaving needed
/// Simply copies the row from src to dst
static inline void tqt_gemm_pack_mxk_e32_kernel_1row(size_t k, const float *src, float *dst)
{
    size_t size = k;
    while (size > 0) {
        size_t vl = __riscv_vsetvl_e32m1(size);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(src, vl);
        __riscv_vse32_v_f32m1(dst, v, vl);
        src += vl;
        dst += vl;
        size -= vl;
    }
}

/// Pack kernel for A matrix: [m, k] -> [m/MR, k, MR]
/// Uses vsseg instructions to interleave MR rows
/// MR: number of rows (2-8)
template <size_t MR>
static inline void tqt_gemm_pack_mxk_e16_kernel(size_t k, const float16_t *src, size_t lda,
                                                float16_t *dst)
{
    static_assert(MR >= 2 && MR <= 8, "MR must be in range [2, 8]");

    size_t size = k;
    while (size > 0) {
        size_t vl = __riscv_vsetvl_e16m1(size);

        vfloat16m1_t v0, v1, v2, v3, v4, v5, v6, v7;

        // Load MR rows
        if constexpr (MR >= 1)
            v0 = __riscv_vle16_v_f16m1(src + 0 * lda, vl);
        if constexpr (MR >= 2)
            v1 = __riscv_vle16_v_f16m1(src + 1 * lda, vl);
        if constexpr (MR >= 3)
            v2 = __riscv_vle16_v_f16m1(src + 2 * lda, vl);
        if constexpr (MR >= 4)
            v3 = __riscv_vle16_v_f16m1(src + 3 * lda, vl);
        if constexpr (MR >= 5)
            v4 = __riscv_vle16_v_f16m1(src + 4 * lda, vl);
        if constexpr (MR >= 6)
            v5 = __riscv_vle16_v_f16m1(src + 5 * lda, vl);
        if constexpr (MR >= 7)
            v6 = __riscv_vle16_v_f16m1(src + 6 * lda, vl);
        if constexpr (MR >= 8)
            v7 = __riscv_vle16_v_f16m1(src + 7 * lda, vl);

        // Store with vsseg (interleaved)
        if constexpr (MR == 2) {
            vfloat16m1x2_t src = __riscv_vcreate_v_f16m1x2(v0, v1);
            __riscv_vsseg2e16_v_f16m1x2(dst, src, vl);
        } else if constexpr (MR == 3) {
            vfloat16m1x3_t src = __riscv_vcreate_v_f16m1x3(v0, v1, v2);
            __riscv_vsseg3e16_v_f16m1x3(dst, src, vl);
        } else if constexpr (MR == 4) {
            vfloat16m1x4_t src = __riscv_vcreate_v_f16m1x4(v0, v1, v2, v3);
            __riscv_vsseg4e16_v_f16m1x4(dst, src, vl);
        } else if constexpr (MR == 5) {
            vfloat16m1x5_t src = __riscv_vcreate_v_f16m1x5(v0, v1, v2, v3, v4);
            __riscv_vsseg5e16_v_f16m1x5(dst, src, vl);
        } else if constexpr (MR == 6) {
            vfloat16m1x6_t src = __riscv_vcreate_v_f16m1x6(v0, v1, v2, v3, v4, v5);
            __riscv_vsseg6e16_v_f16m1x6(dst, src, vl);
        } else if constexpr (MR == 7) {
            vfloat16m1x7_t src = __riscv_vcreate_v_f16m1x7(v0, v1, v2, v3, v4, v5, v6);
            __riscv_vsseg7e16_v_f16m1x7(dst, src, vl);
        } else if constexpr (MR == 8) {
            vfloat16m1x8_t src = __riscv_vcreate_v_f16m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
            __riscv_vsseg8e16_v_f16m1x8(dst, src, vl);
        }

        src += vl;
        dst += vl * MR;
        size -= vl;
    }
}

/// Pack kernel for single row A matrix: [1, k] -> [1, k]
/// Special case for MR = 1, no interleaving needed
/// Simply copies the row from src to dst
static inline void tqt_gemm_pack_mxk_e16_kernel_1row(size_t k, const float16_t *src, float16_t *dst)
{
    size_t size = k;
    while (size > 0) {
        size_t vl = __riscv_vsetvl_e16m1(size);
        vfloat16m1_t v = __riscv_vle16_v_f16m1(src, vl);
        __riscv_vse16_v_f16m1(dst, v, vl);
        src += vl;
        dst += vl;
        size -= vl;
    }
}

/// Pack kernel for non-transposed B matrix: [k, n] -> [n/(NR*vl), k, NR*vl]
/// NR: number of vector registers (1, 2, or 3)
template <size_t NR>
static inline void tqt_gemm_pack_kxn_kernel(size_t n, size_t k, const uint8_t *src, size_t ldb,
                                            uint8_t *dst, size_t vl, size_t bytes_per_elem)
{
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");
    size_t stride_b_bytes = ldb * bytes_per_elem;
    size_t n_bytes = n * bytes_per_elem;
    size_t vl_e8m1 = vl * bytes_per_elem;
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(vl_e8m1) :);

    for (size_t k_idx = 0; k_idx < k; k_idx++) {
        if constexpr (NR >= 1) {
            asm volatile(
                "vle8.v v0, (%[src])\n\t"
                "vse8.v v0, (%[dst])\n\t"
                :
                : [src] "r"(src + 0 * vl_e8m1), [dst] "r"(dst + 0 * vl_e8m1)
                : "v0", "memory");
        }
        if constexpr (NR >= 2) {
            asm volatile(
                "vle8.v v1, (%[src])\n\t"
                "vse8.v v1, (%[dst])\n\t"
                :
                : [src] "r"(src + 1 * vl_e8m1), [dst] "r"(dst + 1 * vl_e8m1)
                : "v1", "memory");
        }
        if constexpr (NR >= 3) {
            asm volatile(
                "vle8.v v2, (%[src])\n\t"
                "vse8.v v2, (%[dst])\n\t"
                :
                : [src] "r"(src + 2 * vl_e8m1), [dst] "r"(dst + 2 * vl_e8m1)
                : "v2", "memory");
        }
        src += stride_b_bytes;
        dst += n_bytes;
    }
}

/// Pack kernel for transposed B matrix: [n, k] -> [n/(NR*vl), k, NR*vl]
/// NR: number of vector registers (1, 2, or 3)
template <size_t NR>
static inline void tqt_gemm_pack_nxk_e32_kernel(size_t n, size_t k, const uint32_t *src, size_t ldb,
                                                uint32_t *dst, size_t vl)
{
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");
    const size_t stride_b = ldb * 4;
    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(vl) :);

    for (size_t k_idx = 0; k_idx < k; k_idx++) {
        if constexpr (NR >= 1) {
            asm volatile(
                "vlse32.v v0, (%[src]), %[stride]\n\t"
                "vse32.v  v0, (%[dst])\n\t"
                :
                : [src] "r"(src + 0 * vl * ldb), [stride] "r"(stride_b), [dst] "r"(dst + 0 * vl)
                : "v0", "memory");
        }
        if constexpr (NR >= 2) {
            asm volatile(
                "vlse32.v v1, (%[src]), %[stride]\n\t"
                "vse32.v  v1, (%[dst])\n\t"
                :
                : [src] "r"(src + 1 * vl * ldb), [stride] "r"(stride_b), [dst] "r"(dst + 1 * vl)
                : "v1", "memory");
        }
        if constexpr (NR >= 3) {
            asm volatile(
                "vlse32.v v2, (%[src]), %[stride]\n\t"
                "vse32.v  v2, (%[dst])\n\t"
                :
                : [src] "r"(src + 2 * vl * ldb), [stride] "r"(stride_b), [dst] "r"(dst + 2 * vl)
                : "v2", "memory");
        }
        src += 1;
        dst += n;
    }
}

/// Pack kernel for transposed B matrix: [n, k] -> [n/(NR*vl), k, NR*vl]
/// NR: number of vector registers (1, 2, or 3)
template <size_t NR>
static inline void tqt_gemm_pack_nxk_e16_kernel(size_t n, size_t k, const uint16_t *src, size_t ldb,
                                                uint16_t *dst, size_t vl)
{
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");
    const size_t stride_b = ldb * 2;
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" : : "r"(vl) :);

    for (size_t k_idx = 0; k_idx < k; k_idx++) {
        if constexpr (NR >= 1) {
            asm volatile(
                "vlse16.v v0, (%[src]), %[stride]\n\t"
                "vse16.v  v0, (%[dst])\n\t"
                :
                : [src] "r"(src + 0 * vl * ldb), [stride] "r"(stride_b), [dst] "r"(dst + 0 * vl)
                : "v0", "memory");
        }
        if constexpr (NR >= 2) {
            asm volatile(
                "vlse16.v v1, (%[src]), %[stride]\n\t"
                "vse16.v  v1, (%[dst])\n\t"
                :
                : [src] "r"(src + 1 * vl * ldb), [stride] "r"(stride_b), [dst] "r"(dst + 1 * vl)
                : "v1", "memory");
        }
        if constexpr (NR >= 3) {
            asm volatile(
                "vlse16.v v2, (%[src]), %[stride]\n\t"
                "vse16.v  v2, (%[dst])\n\t"
                :
                : [src] "r"(src + 2 * vl * ldb), [stride] "r"(stride_b), [dst] "r"(dst + 2 * vl)
                : "v2", "memory");
        }
        src += 1;
        dst += n;
    }
}

// ============================================================================
// Public Pack Functions
// ============================================================================

/// Pack A matrix: [m, k] -> [m/8, k, 8]
/// A is stored in row-major with shape [m, k]
/// A_packed will be stored as [m/8, k, 8]
static void tqt_gemm_pack_mxk_8x1_e32_rvv(size_t m, size_t k, size_t lda, size_t lda_packed,
                                          size_t k_idx, const void *A, void *A_packed)
{
    const size_t m_step = 8;

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        const size_t actual_m = TQT_MIN(m - m_idx, m_step);
        const float *a_ptr = (const float *)A + m_idx * lda;
        float *a_packed_ptr;
        if (m_idx == 0 || k_idx == 0 || actual_m == m_step) {
            a_packed_ptr = (float *)A_packed + m_idx * lda_packed;
        } else {
            a_packed_ptr = (float *)A_packed + m_idx * lda_packed - k_idx * (m_step - actual_m);
        }

        switch (actual_m) {
            case 8:
                tqt_gemm_pack_mxk_e32_kernel<8>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 7:
                tqt_gemm_pack_mxk_e32_kernel<7>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 6:
                tqt_gemm_pack_mxk_e32_kernel<6>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 5:
                tqt_gemm_pack_mxk_e32_kernel<5>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 4:
                tqt_gemm_pack_mxk_e32_kernel<4>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 3:
                tqt_gemm_pack_mxk_e32_kernel<3>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 2:
                tqt_gemm_pack_mxk_e32_kernel<2>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 1:
                tqt_gemm_pack_mxk_e32_kernel_1row(k, a_ptr, a_packed_ptr);
                break;
            default:
                break;
        }
    }
}

/// Pack A matrix: [m, k] -> [m/8, k, 8]
/// A is stored in row-major with shape [m, k]
/// A_packed will be stored as [m/8, k, 8]
static void tqt_gemm_pack_mxk_8x1_e16_rvv(size_t m, size_t k, size_t lda, size_t lda_packed,
                                          size_t k_idx, const void *A, void *A_packed)
{
    const size_t m_step = 8;

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        const size_t actual_m = TQT_MIN(m - m_idx, m_step);
        const float16_t *a_ptr = (const float16_t *)A + m_idx * lda;
        float16_t *a_packed_ptr;
        if (m_idx == 0 || k_idx == 0 || actual_m == m_step) {
            a_packed_ptr = (float16_t *)A_packed + m_idx * lda_packed;
        } else {
            a_packed_ptr = (float16_t *)A_packed + m_idx * lda_packed - k_idx * (m_step - actual_m);
        }

        switch (actual_m) {
            case 8:
                tqt_gemm_pack_mxk_e16_kernel<8>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 7:
                tqt_gemm_pack_mxk_e16_kernel<7>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 6:
                tqt_gemm_pack_mxk_e16_kernel<6>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 5:
                tqt_gemm_pack_mxk_e16_kernel<5>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 4:
                tqt_gemm_pack_mxk_e16_kernel<4>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 3:
                tqt_gemm_pack_mxk_e16_kernel<3>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 2:
                tqt_gemm_pack_mxk_e16_kernel<2>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 1:
                tqt_gemm_pack_mxk_e16_kernel_1row(k, a_ptr, a_packed_ptr);
                break;
            default:
                break;
        }
    }
}

/// Pack non-transposed B matrix: [k, n] -> [n/(3*vl), k, 3*vl]
/// B is stored in row-major with shape [k, n]
/// B_packed will be stored as [n/(3*vl), k, 3*vl]
static void tqt_gemm_pack_kxn_1x3vl_rvv(size_t n, size_t k, size_t ldb, size_t ldb_packed,
                                        size_t k_idx, const void *B, void *B_packed,
                                        size_t bit_size)
{
    const size_t bytes_per_elem = bit_size / 8;
    const size_t vlmax = csrr_vlenb() / bytes_per_elem;
    const size_t n_step = 3 * vlmax;

    const uint8_t *B_u8 = (const uint8_t *)B;
    uint8_t *B_packed_u8 = (uint8_t *)B_packed;

    for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
        const size_t actual_n = TQT_MIN(n - n_idx, n_step);
        const uint8_t *b_ptr = B_u8 + n_idx * bytes_per_elem;
        uint8_t *b_packed_ptr;
        if (n_idx == 0 || k_idx == 0 || actual_n == n_step) {
            b_packed_ptr = B_packed_u8 + (n_idx * ldb_packed) * bytes_per_elem;
        } else {
            b_packed_ptr =
                B_packed_u8 + (n_idx * ldb_packed - k_idx * (n_step - actual_n)) * bytes_per_elem;
        }

        size_t size_n = 0;
        if (actual_n >= vlmax) {
            if (actual_n == 3 * vlmax) {
                size_n = 3 * vlmax;
                tqt_gemm_pack_kxn_kernel<3>(size_n, k, b_ptr, ldb, b_packed_ptr, vlmax,
                                            bytes_per_elem);
            } else if (actual_n >= 2 * vlmax) {
                size_n = 2 * vlmax;
                tqt_gemm_pack_kxn_kernel<2>(size_n, k, b_ptr, ldb, b_packed_ptr, vlmax,
                                            bytes_per_elem);
            } else {
                size_n = vlmax;
                tqt_gemm_pack_kxn_kernel<1>(size_n, k, b_ptr, ldb, b_packed_ptr, vlmax,
                                            bytes_per_elem);
            }
        }
        size_t remain_n = actual_n - size_n;
        if (remain_n > 0) {
            b_ptr += size_n * bytes_per_elem;
            b_packed_ptr += size_n * ldb_packed * bytes_per_elem;
            tqt_gemm_pack_kxn_kernel<1>(remain_n, k, b_ptr, ldb, b_packed_ptr, remain_n,
                                        bytes_per_elem);
        }
    }
}

/// Pack transposed B matrix: [n, k] -> [n/(3*vl), k, 3*vl]
/// B is stored in row-major with shape [n, k]
/// B_packed will be stored as [n/(3*vl), k, 3*vl]
static void tqt_gemm_pack_nxk_1x3vl_e32_rvv(size_t n, size_t k, size_t ldb, size_t ldb_packed,
                                            size_t k_idx, const void *B, void *B_packed)
{
    const size_t vlmax = csrr_vlenb() / 4;
    const size_t n_step = 3 * vlmax;

    const uint32_t *B_u32 = (const uint32_t *)B;
    uint32_t *B_packed_u32 = (uint32_t *)B_packed;

    for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
        const size_t actual_n = TQT_MIN(n - n_idx, n_step);
        const uint32_t *b_ptr = B_u32 + n_idx * ldb;
        uint32_t *b_packed_ptr;
        if (n_idx == 0 || k_idx == 0 || actual_n == n_step) {
            b_packed_ptr = B_packed_u32 + n_idx * ldb_packed;
        } else {
            b_packed_ptr = B_packed_u32 + n_idx * ldb_packed - k_idx * (n_step - actual_n);
        }

        size_t size_n = 0;
        if (actual_n >= vlmax) {
            if (actual_n == 3 * vlmax) {
                size_n = 3 * vlmax;
                tqt_gemm_pack_nxk_e32_kernel<3>(size_n, k, b_ptr, ldb, b_packed_ptr, vlmax);
            } else if (actual_n >= 2 * vlmax) {
                size_n = 2 * vlmax;
                tqt_gemm_pack_nxk_e32_kernel<2>(size_n, k, b_ptr, ldb, b_packed_ptr, vlmax);
            } else {
                size_n = vlmax;
                tqt_gemm_pack_nxk_e32_kernel<1>(size_n, k, b_ptr, ldb, b_packed_ptr, vlmax);
            }
        }
        size_t remain_n = actual_n - size_n;
        if (remain_n > 0) {
            b_ptr += size_n * ldb;
            b_packed_ptr += size_n * ldb_packed;
            tqt_gemm_pack_nxk_e32_kernel<1>(remain_n, k, b_ptr, ldb, b_packed_ptr, remain_n);
        }
    }
}

/// Pack transposed B matrix: [n, k] -> [n/(3*vl), k, 3*vl]
/// B is stored in row-major with shape [n, k]
/// B_packed will be stored as [n/(3*vl), k, 3*vl]
static void tqt_gemm_pack_nxk_1x3vl_e16_rvv(size_t n, size_t k, size_t ldb, size_t ldb_packed,
                                            size_t k_idx, const void *B, void *B_packed)
{
    const size_t vlmax = csrr_vlenb() / 2;
    const size_t n_step = 3 * vlmax;

    const uint16_t *B_u16 = (const uint16_t *)B;
    uint16_t *B_packed_u16 = (uint16_t *)B_packed;

    for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
        const size_t actual_n = TQT_MIN(n - n_idx, n_step);
        const uint16_t *b_ptr = B_u16 + n_idx * ldb;
        uint16_t *b_packed_ptr;
        if (n_idx == 0 || k_idx == 0 || actual_n == n_step) {
            b_packed_ptr = B_packed_u16 + n_idx * ldb_packed;
        } else {
            b_packed_ptr = B_packed_u16 + n_idx * ldb_packed - k_idx * (n_step - actual_n);
        }

        size_t size_n = 0;
        if (actual_n >= vlmax) {
            if (actual_n == 3 * vlmax) {
                size_n = 3 * vlmax;
                tqt_gemm_pack_nxk_e16_kernel<3>(size_n, k, b_ptr, ldb, b_packed_ptr, vlmax);
            } else if (actual_n >= 2 * vlmax) {
                size_n = 2 * vlmax;
                tqt_gemm_pack_nxk_e16_kernel<2>(size_n, k, b_ptr, ldb, b_packed_ptr, vlmax);
            } else {
                size_n = vlmax;
                tqt_gemm_pack_nxk_e16_kernel<1>(size_n, k, b_ptr, ldb, b_packed_ptr, vlmax);
            }
        }
        size_t remain_n = actual_n - size_n;
        if (remain_n > 0) {
            b_ptr += size_n * ldb;
            b_packed_ptr += size_n * ldb_packed;
            tqt_gemm_pack_nxk_e16_kernel<1>(remain_n, k, b_ptr, ldb, b_packed_ptr, remain_n);
        }
    }
}

/// Pack non-transposed B matrix: [k, n] -> [n/vl, k, vl]
/// B is stored in row-major with shape [k, n]
/// B_packed will be stored as [n/vl, k, vl]
static void tqt_gemm_pack_kxn_1x1vl_rvv(size_t n, size_t k, size_t ldb, size_t ldb_packed,
                                        size_t k_idx, const void *B, void *B_packed,
                                        size_t bit_size)
{
    const size_t bytes_per_elem = bit_size / 8;
    const size_t vlmax = csrr_vlenb() / bytes_per_elem;
    const size_t n_step = vlmax;

    const uint8_t *B_u8 = (const uint8_t *)B;
    uint8_t *B_packed_u8 = (uint8_t *)B_packed;

    for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
        const size_t actual_n = TQT_MIN(n - n_idx, n_step);
        const uint8_t *b_ptr = B_u8 + n_idx * bytes_per_elem;
        uint8_t *b_packed_ptr;
        if (n_idx == 0 || k_idx == 0 || actual_n == n_step) {
            b_packed_ptr = B_packed_u8 + (n_idx * ldb_packed) * bytes_per_elem;
        } else {
            b_packed_ptr =
                B_packed_u8 + (n_idx * ldb_packed - k_idx * (n_step - actual_n)) * bytes_per_elem;
        }

        tqt_gemm_pack_kxn_kernel<1>(actual_n, k, b_ptr, ldb, b_packed_ptr, actual_n,
                                    bytes_per_elem);
    }
}

/// Pack transposed B matrix: [n, k] -> [n/vl, k, vl]
/// B is stored in row-major with shape [n, k]
/// B_packed will be stored as [n/vl, k, vl]
static void tqt_gemm_pack_nxk_1x1vl_e16_rvv(size_t n, size_t k, size_t ldb, size_t ldb_packed,
                                            size_t k_idx, const void *B, void *B_packed)
{
    const size_t vlmax = csrr_vlenb() / 2;
    const size_t n_step = vlmax;

    const uint16_t *B_u16 = (const uint16_t *)B;
    uint16_t *B_packed_u16 = (uint16_t *)B_packed;

    for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
        const size_t actual_n = TQT_MIN(n - n_idx, n_step);
        const uint16_t *b_ptr = B_u16 + n_idx * ldb;
        uint16_t *b_packed_ptr;
        if (n_idx == 0 || k_idx == 0 || actual_n == n_step) {
            b_packed_ptr = B_packed_u16 + n_idx * ldb_packed;
        } else {
            b_packed_ptr = B_packed_u16 + n_idx * ldb_packed - k_idx * (n_step - actual_n);
        }

        tqt_gemm_pack_nxk_e16_kernel<1>(actual_n, k, b_ptr, ldb, b_packed_ptr, actual_n);
    }
}

// ============================================================================
// INT8 Pack Kernels
// ============================================================================

/// Pack kernel for A matrix: [m, k] -> [m/MR, k, MR]
/// Uses vsseg instructions to interleave MR rows
/// MR: number of rows (2-4)
template <size_t MR>
static inline void tqt_gemm_pack_mxk_e8_kernel(size_t k, const int8_t *src, size_t lda, int8_t *dst)
{
    static_assert(MR >= 2 && MR <= 4, "MR must be in range [2, 4]");

    size_t size = k;
    while (size > 0) {
        size_t vl = __riscv_vsetvl_e8m1(size);

        vint8m1_t v0, v1, v2, v3;

        // Load MR rows
        if constexpr (MR >= 1)
            v0 = __riscv_vle8_v_i8m1(src + 0 * lda, vl);
        if constexpr (MR >= 2)
            v1 = __riscv_vle8_v_i8m1(src + 1 * lda, vl);
        if constexpr (MR >= 3)
            v2 = __riscv_vle8_v_i8m1(src + 2 * lda, vl);
        if constexpr (MR >= 4)
            v3 = __riscv_vle8_v_i8m1(src + 3 * lda, vl);

        // Store with vsseg (interleaved)
        if constexpr (MR == 2) {
            vint8m1x2_t src = __riscv_vcreate_v_i8m1x2(v0, v1);
            __riscv_vsseg2e8_v_i8m1x2(dst, src, vl);
        } else if constexpr (MR == 3) {
            vint8m1x3_t src = __riscv_vcreate_v_i8m1x3(v0, v1, v2);
            __riscv_vsseg3e8_v_i8m1x3(dst, src, vl);
        } else if constexpr (MR == 4) {
            vint8m1x4_t src = __riscv_vcreate_v_i8m1x4(v0, v1, v2, v3);
            __riscv_vsseg4e8_v_i8m1x4(dst, src, vl);
        }

        src += vl;
        dst += vl * MR;
        size -= vl;
    }
}

/// Pack kernel for single row A matrix: [1, k] -> [1, k]
/// Special case for MR = 1, no interleaving needed
/// Simply copies the row from src to dst
static inline void tqt_gemm_pack_mxk_e8_kernel_1row(size_t k, const int8_t *src, int8_t *dst)
{
    size_t size = k;
    while (size > 0) {
        size_t vl = __riscv_vsetvl_e8m1(size);
        vint8m1_t v = __riscv_vle8_v_i8m1(src, vl);
        __riscv_vse8_v_i8m1(dst, v, vl);
        src += vl;
        dst += vl;
        size -= vl;
    }
}

/// Pack kernel for transposed B matrix: [n, k] -> [n/(NR*vl), k, NR*vl]
/// NR: number of vector registers (1)
template <size_t NR>
static inline void tqt_gemm_pack_nxk_e8_kernel(size_t n, size_t k, const uint8_t *src, size_t ldb,
                                               uint8_t *dst, size_t vl)
{
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");
    const size_t stride_b = ldb;
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(vl) :);

    for (size_t k_idx = 0; k_idx < k; k_idx++) {
        if constexpr (NR >= 1) {
            asm volatile(
                "vlse8.v v0, (%[src]), %[stride]\n\t"
                "vse8.v  v0, (%[dst])\n\t"
                :
                : [src] "r"(src + 0 * vl * ldb), [stride] "r"(stride_b), [dst] "r"(dst + 0 * vl)
                : "v0", "memory");
        }
        if constexpr (NR >= 2) {
            asm volatile(
                "vlse8.v v1, (%[src]), %[stride]\n\t"
                "vse8.v  v1, (%[dst])\n\t"
                :
                : [src] "r"(src + 1 * vl * ldb), [stride] "r"(stride_b), [dst] "r"(dst + 1 * vl)
                : "v1", "memory");
        }
        if constexpr (NR >= 3) {
            asm volatile(
                "vlse8.v v2, (%[src]), %[stride]\n\t"
                "vse8.v  v2, (%[dst])\n\t"
                :
                : [src] "r"(src + 2 * vl * ldb), [stride] "r"(stride_b), [dst] "r"(dst + 2 * vl)
                : "v2", "memory");
        }
        src += 1;
        dst += n;
    }
}

/// Pack kernel for transposed B matrix (e8 mf2 granularity): same as e8_kernel but uses LMUL=mf2.
/// Max VL = VLENB/2 = vlmax_e16. Used for sub-packing partial N-tiles in W4A16 kernels.
template <size_t NR>
static inline void tqt_gemm_pack_nxk_e8mf2_kernel(size_t n, size_t k, const uint8_t *src,
                                                  size_t ldb, uint8_t *dst, size_t vl)
{
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");
    const size_t stride_b = ldb;
    asm volatile("vsetvli zero, %0, e8, mf2, ta, ma" : : "r"(vl) :);

    for (size_t k_idx = 0; k_idx < k; k_idx++) {
        if constexpr (NR >= 1) {
            asm volatile(
                "vlse8.v v0, (%[src]), %[stride]\n\t"
                "vse8.v  v0, (%[dst])\n\t"
                :
                : [src] "r"(src + 0 * vl * ldb), [stride] "r"(stride_b), [dst] "r"(dst + 0 * vl)
                : "v0", "memory");
        }
        if constexpr (NR >= 2) {
            asm volatile(
                "vlse8.v v1, (%[src]), %[stride]\n\t"
                "vse8.v  v1, (%[dst])\n\t"
                :
                : [src] "r"(src + 1 * vl * ldb), [stride] "r"(stride_b), [dst] "r"(dst + 1 * vl)
                : "v1", "memory");
        }
        if constexpr (NR >= 3) {
            asm volatile(
                "vlse8.v v2, (%[src]), %[stride]\n\t"
                "vse8.v  v2, (%[dst])\n\t"
                :
                : [src] "r"(src + 2 * vl * ldb), [stride] "r"(stride_b), [dst] "r"(dst + 2 * vl)
                : "v2", "memory");
        }
        src += 1;
        dst += n;
    }
}

/// Pack A matrix: [m, k] -> [m/4, k, 4]
/// A is stored in row-major with shape [m, k]
/// A_packed will be stored as [m/4, k, 4]
static void tqt_gemm_pack_mxk_4x1_e8_rvv(size_t m, size_t k, size_t lda, size_t lda_packed,
                                         size_t k_idx, const void *A, void *A_packed)
{
    const size_t m_step = 4;

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        const size_t actual_m = TQT_MIN(m - m_idx, m_step);
        const int8_t *a_ptr = (const int8_t *)A + m_idx * lda;
        int8_t *a_packed_ptr;
        if (m_idx == 0 || k_idx == 0 || actual_m == m_step) {
            a_packed_ptr = (int8_t *)A_packed + m_idx * lda_packed;
        } else {
            a_packed_ptr = (int8_t *)A_packed + m_idx * lda_packed - k_idx * (m_step - actual_m);
        }

        switch (actual_m) {
            case 4:
                tqt_gemm_pack_mxk_e8_kernel<4>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 3:
                tqt_gemm_pack_mxk_e8_kernel<3>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 2:
                tqt_gemm_pack_mxk_e8_kernel<2>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 1:
                tqt_gemm_pack_mxk_e8_kernel_1row(k, a_ptr, a_packed_ptr);
                break;
            default:
                break;
        }
    }
}

/// Pack transposed B matrix: [n, k] -> [n/vl, k, vl]
/// B is stored in row-major with shape [n, k]
/// B_packed will be stored as [n/vl, k, vl]
static void tqt_gemm_pack_nxk_1x1vl_e8_rvv(size_t n, size_t k, size_t ldb, size_t ldb_packed,
                                           size_t k_idx, const void *B, void *B_packed)
{
    const size_t vlmax = csrr_vlenb();
    const size_t n_step = vlmax;

    const uint8_t *B_u8 = (const uint8_t *)B;
    uint8_t *B_packed_u8 = (uint8_t *)B_packed;

    for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
        const size_t actual_n = TQT_MIN(n - n_idx, n_step);
        const uint8_t *b_ptr = B_u8 + n_idx * ldb;
        uint8_t *b_packed_ptr;
        if (n_idx == 0 || k_idx == 0 || actual_n == n_step) {
            b_packed_ptr = B_packed_u8 + n_idx * ldb_packed;
        } else {
            b_packed_ptr = B_packed_u8 + n_idx * ldb_packed - k_idx * (n_step - actual_n);
        }

        tqt_gemm_pack_nxk_e8_kernel<1>(actual_n, k, b_ptr, ldb, b_packed_ptr, actual_n);
    }
}

/// Pack transposed B matrix: [n, k] -> [n/(2*vl_mf2), k, 2*vl_mf2]
/// Uses e8 mf2 as minimum pack granularity (vl_mf2 = VLENB/2 = vlmax_e16).
/// n_step = 2 * vl_mf2 = VLENB (same tiling as 1x1vl_e8).
///
/// For full tiles (actual_n >= n_step): uses e8 m1 kernel (all elements in one pass).
/// For partial tiles (vl_mf2 < actual_n < n_step): splits into two e8 mf2 sub-packs,
///   so each sub-pack's width matches the kernel's NR=1 stride.
/// For tail (actual_n <= vl_mf2): single e8 mf2 call.
static void tqt_gemm_pack_nxk_1x2vl_e8mf2_rvv(size_t n, size_t k, size_t ldb, size_t ldb_packed,
                                              size_t k_idx, const void *B, void *B_packed)
{
    const size_t vlmax_mf2 = csrr_vlenb() / 2;
    const size_t n_step = 2 * vlmax_mf2;

    const uint8_t *B_u8 = (const uint8_t *)B;
    uint8_t *B_packed_u8 = (uint8_t *)B_packed;

    for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
        const size_t actual_n = TQT_MIN(n - n_idx, n_step);
        const uint8_t *b_ptr = B_u8 + n_idx * ldb;
        uint8_t *b_packed_ptr;
        if (n_idx == 0 || k_idx == 0 || actual_n == n_step) {
            b_packed_ptr = B_packed_u8 + n_idx * ldb_packed;
        } else {
            b_packed_ptr = B_packed_u8 + n_idx * ldb_packed - k_idx * (n_step - actual_n);
        }

        if (actual_n >= n_step) {
            tqt_gemm_pack_nxk_e8_kernel<1>(actual_n, k, b_ptr, ldb, b_packed_ptr, actual_n);
        } else if (actual_n > vlmax_mf2) {
            const size_t remain = actual_n - vlmax_mf2;
            tqt_gemm_pack_nxk_e8mf2_kernel<1>(vlmax_mf2, k, b_ptr, ldb, b_packed_ptr, vlmax_mf2);
            tqt_gemm_pack_nxk_e8mf2_kernel<1>(remain, k, b_ptr + vlmax_mf2 * ldb, ldb,
                                              b_packed_ptr + vlmax_mf2 * ldb_packed, remain);
        } else {
            tqt_gemm_pack_nxk_e8mf2_kernel<1>(actual_n, k, b_ptr, ldb, b_packed_ptr, actual_n);
        }
    }
}

/// Pack transposed B matrix for int4: [n, k] int4 -> packed layout
/// This is a wrapper that treats 2 int4 as 1 uint8 byte.
/// Uses e8 mf2 as minimum pack granularity for correct partial-tile handling
/// in W4A16 kernels where NR is defined in e16 (fp16) terms.
///
/// @param[in]  n         Number of rows in B (N dimension).
/// @param[in]  k         Number of int4 columns in B (K dimension, must be even).
/// @param[in]  ldb       Leading dimension of B in int4 elements (= k).
/// @param[in]  ldb_packed Leading dimension of packed B in int4 elements.
/// @param[in]  k_idx     Starting k index for packing (in int4 elements, must be even).
/// @param[in]  B         Input matrix B [n, k/2] uint8 (row-major).
/// @param[out] B_packed  Packed matrix B buffer.
static void tqt_gemm_pack_nxk_2x1vl_e4mf2_rvv(size_t n, size_t k, size_t ldb, size_t ldb_packed,
                                              size_t k_idx, const void *B, void *B_packed)
{
    tqt_gemm_pack_nxk_1x2vl_e8mf2_rvv(n, k / 2, ldb / 2, ldb_packed / 2, k_idx / 2, B, B_packed);
}
