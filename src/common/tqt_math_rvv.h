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
// Shared RVV Math Primitives
// ============================================================================
//
// Vectorised transcendental helpers for RVV micro-kernels. RVV has no
// transcendental instructions, so every function here is a polynomial or
// rational evaluation written with plain RVV intrinsics.
//
// Only the LMUL variants actually consumed by a shipped micro-kernel are
// instantiated; adding another one is a single macro invocation at the end of
// the corresponding section.
// ============================================================================

// ----------------------------------------------------------------------------
// exp
// ----------------------------------------------------------------------------
//
// exp(x) = 2^n * (1 + poly(r)), r = x - n * ln(2). The coefficients and grouped
// degree-5 evaluation follow Arm's optimized expf, also used by GGML. Adding
// 0x1.8p23f rounds x * log2(e) to an integer under RNE and encodes that integer
// in the significand, avoiding numeric float/integer conversions.
// Reference: Arm optimized-routines, math/aarch64/advsimd/v_expf_inline.h.
//
// Contract of the generated tqt_vexp_f32m<LMUL>:
//   - Requires round-to-nearest, ties-to-even (RNE); does not change frm.
//   - exp(+/-0) = 1, exp(-Inf) = +0, exp(+Inf) = +Inf; NaNs propagate.
//   - Split scaling preserves subnormal results and permits overflow to +Inf.
//     Finite inputs are not clamped to a finite saturation value.
//   - Accuracy is validated through the consuming micro-kernel tests.
//     Correct rounding and exhaustive all-f32 accuracy are not guaranteed.
//   - NaN payloads and floating-point exception flags are not specified. The
//     exceptional path can evaluate intermediate values discarded by a merge.
//
// Select and prepare scaling before polynomial evaluation. This keeps exponent
// temporaries out of the FMA chain and leaves room for the caller's SiLU gate.

#define TQT_RVV_EXP_SHIFT 0x1.8p23f
#define TQT_RVV_EXP_LOG2EF 0x1.715476p+0f
#define TQT_RVV_EXP_C1 0x1.62e4p-1f
#define TQT_RVV_EXP_C2 0x1.7f7d1cp-20f

#define TQT_RVV_EXP_P0 0x1.ffffecp-1f
#define TQT_RVV_EXP_P1 0x1.fffdb6p-2f
#define TQT_RVV_EXP_P2 0x1.555e66p-3f
#define TQT_RVV_EXP_P3 0x1.573e2ep-5f
#define TQT_RVV_EXP_P4 0x1.0e4020p-7f

/// Generates `tqt_vexp_f32m<LMUL>(vfloat32m<LMUL>_t x, size_t vl)`.
///
/// @param LMUL Vector length multiplier of the operand group (1, 2, 4 or 8).
/// @param MLEN Mask length matching LMUL for 32-bit elements (32 / LMUL).
#define TQT_RVV_F32_EXP_OP(LMUL, MLEN)                                                             \
    static inline vfloat32m##LMUL##_t tqt_vexp_poly_f32m##LMUL(vfloat32m##LMUL##_t x, float c1,    \
                                                               float c3, size_t vl)                \
    {                                                                                              \
        /* poly(x) = P0*x + x^2 * (P1 + P2*x + x^2 * (P3 + P4*x)). */                              \
        vfloat32m##LMUL##_t u = __riscv_vfmul_vv_f32m##LMUL(x, x, vl);                             \
        vfloat32m##LMUL##_t p = __riscv_vfmv_v_f_f32m##LMUL(c3, vl);                               \
        p = __riscv_vfmacc_vf_f32m##LMUL(p, TQT_RVV_EXP_P4, x, vl);                                \
        vfloat32m##LMUL##_t q = __riscv_vfmv_v_f_f32m##LMUL(c1, vl);                               \
        q = __riscv_vfmacc_vf_f32m##LMUL(q, TQT_RVV_EXP_P2, x, vl);                                \
        q = __riscv_vfmacc_vv_f32m##LMUL(q, p, u, vl);                                             \
        p = __riscv_vfmul_vf_f32m##LMUL(x, TQT_RVV_EXP_P0, vl);                                    \
        return __riscv_vfmacc_vv_f32m##LMUL(p, q, u, vl);                                          \
    }                                                                                              \
    static inline vfloat32m##LMUL##_t tqt_vexp_f32m##LMUL(vfloat32m##LMUL##_t x, size_t vl)        \
    {                                                                                              \
        vfloat32m##LMUL##_t z = __riscv_vfmv_v_f_f32m##LMUL(TQT_RVV_EXP_SHIFT, vl);                \
        z = __riscv_vfmacc_vf_f32m##LMUL(z, TQT_RVV_EXP_LOG2EF, x, vl);                            \
        vfloat32m##LMUL##_t n = __riscv_vfsub_vf_f32m##LMUL(z, TQT_RVV_EXP_SHIFT, vl);             \
        x = __riscv_vfnmsac_vf_f32m##LMUL(x, TQT_RVV_EXP_C1, n, vl);                               \
        x = __riscv_vfnmsac_vf_f32m##LMUL(x, TQT_RVV_EXP_C2, n, vl);                               \
        vfloat32m##LMUL##_t abs_n = __riscv_vfabs_v_f32m##LMUL(n, vl);                             \
        vbool##MLEN##_t special = __riscv_vmfgt_vf_f32m##LMUL##_b##MLEN(abs_n, 126.0f, vl);        \
        vuint32m##LMUL##_t e =                                                                     \
            __riscv_vsll_vx_u32m##LMUL(__riscv_vreinterpret_v_f32m##LMUL##_u32m##LMUL(z), 23, vl); \
        vfloat32m##LMUL##_t scale = __riscv_vreinterpret_v_u32m##LMUL##_f32m##LMUL(                \
            __riscv_vadd_vx_u32m##LMUL(e, 0x3f800000, vl));                                        \
        if (!__riscv_vcpop_m_b##MLEN(special, vl)) {                                               \
            vfloat32m##LMUL##_t p =                                                                \
                tqt_vexp_poly_f32m##LMUL(x, TQT_RVV_EXP_P1, TQT_RVV_EXP_P3, vl);                   \
            return __riscv_vfmacc_vv_f32m##LMUL(scale, p, scale, vl);                              \
        }                                                                                          \
                                                                                                   \
        /* Keep GCC from speculating exceptional scales into the normal path. */                   \
        __asm__ volatile("" : "+vr"(n), "+vr"(e));                                                 \
        /* Prepare both scales before the polynomial so n/e/abs_n die early at LMUL=4. */          \
        vbool##MLEN##_t extreme = __riscv_vmfgt_vf_f32m##LMUL##_b##MLEN(abs_n, 192.0f, vl);        \
        vint32m##LMUL##_t sign =                                                                   \
            __riscv_vsra_vx_i32m##LMUL(__riscv_vreinterpret_v_f32m##LMUL##_i32m##LMUL(n), 31, vl); \
        vuint32m##LMUL##_t d = __riscv_vand_vx_u32m##LMUL(                                         \
            __riscv_vreinterpret_v_i32m##LMUL##_u32m##LMUL(sign), 0x82000000u, vl);                \
        vfloat32m##LMUL##_t s1 = __riscv_vreinterpret_v_u32m##LMUL##_f32m##LMUL(                   \
            __riscv_vadd_vx_u32m##LMUL(d, 0x7f000000, vl));                                        \
        vfloat32m##LMUL##_t s2 =                                                                   \
            __riscv_vreinterpret_v_u32m##LMUL##_f32m##LMUL(__riscv_vsub_vv_u32m##LMUL(e, d, vl));  \
        s2 = __riscv_vmerge_vvm_f32m##LMUL(scale, s2, special, vl);                                \
        s1 = __riscv_vfmerge_vfm_f32m##LMUL(s1, 1.0f, __riscv_vmnot_m_b##MLEN(special, vl), vl);   \
        /* Extreme lanes evaluate s1*s1 (zero or infinity), without keeping a late mask. */        \
        s2 = __riscv_vmerge_vvm_f32m##LMUL(s2, s1, extreme, vl);                                   \
        x = __riscv_vfmerge_vfm_f32m##LMUL(x, 0.0f, extreme, vl);                                  \
        /* Keep polynomial broadcasts after scale preparation, without a runtime fence. */         \
        float c1 = TQT_RVV_EXP_P1;                                                                 \
        float c3 = TQT_RVV_EXP_P3;                                                                 \
        __asm__ volatile("" : "+vr"(x), "+f"(c1), "+f"(c3) : "vr"(s1), "vr"(s2));                  \
        vfloat32m##LMUL##_t p = tqt_vexp_poly_f32m##LMUL(x, c1, c3, vl);                           \
        s2 = __riscv_vfmacc_vv_f32m##LMUL(s2, p, s2, vl);                                          \
        return __riscv_vfmul_vv_f32m##LMUL(s2, s1, vl);                                            \
    }

// Instantiated LMULs. LMUL=4 serves both RVV SiLU variants: the FP32 kernel
// operates on f32m4 directly, the FP16 kernel widens f16m2 into f32m4.
TQT_RVV_F32_EXP_OP(4, 8)

// ----------------------------------------------------------------------------
// tanh
// ----------------------------------------------------------------------------
//
// Odd-over-even rational approximation evaluated on |x|. Inputs are saturated
// symmetrically at 9 because tanh rounds to +/-1 outside that interval in f32.
// Very small inputs are returned directly to avoid losing their low bits in the
// rational evaluation. The coefficients are the same minimax approximation used
// by the NCNN RVV math helper; the Horner evaluation here is written independently
// in TORQ-Tile's intrinsic style.

#define TQT_RVV_TANH_TINY 1.0e-4f
#define TQT_RVV_TANH_HI 9.0f

#define TQT_RVV_TANH_ALPHA_1 4.89352455891786e-3f
#define TQT_RVV_TANH_ALPHA_3 6.37261928875436e-4f
#define TQT_RVV_TANH_ALPHA_5 1.48572235717979e-5f
#define TQT_RVV_TANH_ALPHA_7 5.12229709037114e-8f
#define TQT_RVV_TANH_ALPHA_9 -8.60467152213735e-11f
#define TQT_RVV_TANH_ALPHA_11 2.00018790482477e-13f
#define TQT_RVV_TANH_ALPHA_13 -2.76076847742355e-16f

#define TQT_RVV_TANH_BETA_0 4.89352518554385e-3f
#define TQT_RVV_TANH_BETA_2 2.26843463243900e-3f
#define TQT_RVV_TANH_BETA_4 1.18534705686654e-4f
#define TQT_RVV_TANH_BETA_6 1.19825839466702e-6f

/// Generates `tqt_vtanh_f32m<LMUL>(vfloat32m<LMUL>_t x, size_t vl)`.
///
/// @param LMUL Vector length multiplier of the operand group.
/// @param MLEN Mask length matching LMUL for 32-bit elements (32 / LMUL).
#define TQT_RVV_F32_TANH_OP(LMUL, MLEN)                                                            \
    static inline vfloat32m##LMUL##_t tqt_vtanh_f32m##LMUL(vfloat32m##LMUL##_t x, size_t vl)       \
    {                                                                                              \
        vfloat32m##LMUL##_t ax = __riscv_vfsgnj_vf_f32m##LMUL(x, 1.0f, vl);                        \
        vbool##MLEN##_t regular =                                                                  \
            __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(ax, TQT_RVV_TANH_TINY, vl);                      \
        ax = __riscv_vfmin_vf_f32m##LMUL(ax, TQT_RVV_TANH_HI, vl);                                 \
                                                                                                   \
        vfloat32m##LMUL##_t z = __riscv_vfmul_vv_f32m##LMUL(ax, ax, vl);                           \
                                                                                                   \
        vfloat32m##LMUL##_t numerator = __riscv_vfmul_vf_f32m##LMUL(z, TQT_RVV_TANH_ALPHA_13, vl); \
        numerator = __riscv_vfadd_vf_f32m##LMUL(numerator, TQT_RVV_TANH_ALPHA_11, vl);             \
        numerator = __riscv_vfmul_vv_f32m##LMUL(numerator, z, vl);                                 \
        numerator = __riscv_vfadd_vf_f32m##LMUL(numerator, TQT_RVV_TANH_ALPHA_9, vl);              \
        numerator = __riscv_vfmul_vv_f32m##LMUL(numerator, z, vl);                                 \
        numerator = __riscv_vfadd_vf_f32m##LMUL(numerator, TQT_RVV_TANH_ALPHA_7, vl);              \
        numerator = __riscv_vfmul_vv_f32m##LMUL(numerator, z, vl);                                 \
        numerator = __riscv_vfadd_vf_f32m##LMUL(numerator, TQT_RVV_TANH_ALPHA_5, vl);              \
        numerator = __riscv_vfmul_vv_f32m##LMUL(numerator, z, vl);                                 \
        numerator = __riscv_vfadd_vf_f32m##LMUL(numerator, TQT_RVV_TANH_ALPHA_3, vl);              \
        numerator = __riscv_vfmul_vv_f32m##LMUL(numerator, z, vl);                                 \
        numerator = __riscv_vfadd_vf_f32m##LMUL(numerator, TQT_RVV_TANH_ALPHA_1, vl);              \
        numerator = __riscv_vfmul_vv_f32m##LMUL(numerator, ax, vl);                                \
                                                                                                   \
        vfloat32m##LMUL##_t denominator = __riscv_vfmul_vf_f32m##LMUL(z, TQT_RVV_TANH_BETA_6, vl); \
        denominator = __riscv_vfadd_vf_f32m##LMUL(denominator, TQT_RVV_TANH_BETA_4, vl);           \
        denominator = __riscv_vfmul_vv_f32m##LMUL(denominator, z, vl);                             \
        denominator = __riscv_vfadd_vf_f32m##LMUL(denominator, TQT_RVV_TANH_BETA_2, vl);           \
        denominator = __riscv_vfmul_vv_f32m##LMUL(denominator, z, vl);                             \
        denominator = __riscv_vfadd_vf_f32m##LMUL(denominator, TQT_RVV_TANH_BETA_0, vl);           \
                                                                                                   \
        vfloat32m##LMUL##_t y = __riscv_vfdiv_vv_f32m##LMUL(numerator, denominator, vl);           \
        y = __riscv_vfsgnj_vv_f32m##LMUL(y, x, vl);                                                \
        return __riscv_vmerge_vvm_f32m##LMUL(x, y, regular, vl);                                   \
    }

// LMUL=4 serves both RVV GeLU variants. The FP16 kernel widens f16m2 into
// f32m4 before evaluating the approximation.
TQT_RVV_F32_TANH_OP(4, 8)
