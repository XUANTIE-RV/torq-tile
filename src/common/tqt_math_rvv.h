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
// Cephes single-precision exponential: exp(x) = 2^n * exp(g), with
// n = floor(x * log2(e) + 0.5) and g = x - n * ln(2) in [-0.35, 0.35], a range a
// degree-5 polynomial covers. ln(2) is split into C1 + C2 so the range reduction
// keeps its low bits.
//
// Contract of the generated tqt_vexp_f32m<LMUL>:
//   - The argument is saturated into [-88.376, 88.376] first, so no argument
//     overflows to infinity: above that range the result saturates around
//     2.4e38 instead. Callers that divide by the result therefore see a
//     denominator that stops growing, which bounds how small their quotient can
//     get - see the negative-tail note in the RVV SiLU kernel headers.
//   - The 2^n step writes a biased exponent field directly, so it cannot
//     produce a subnormal: every argument <= -87.6832 returns +0.0. That
//     includes the [-88.376, -87.6832] sliver whose true value is a
//     representable f32 subnormal, where the relative error is therefore 100%.
//     Over [-87.6832, 88.376] the relative error is on the order of 1e-7.
//   - NaN and +Inf are NOT handled, and specifically NaN does not propagate:
//     vfmin / vfmax follow IEEE minimumNumber / maximumNumber, so both are
//     replaced by the clamp bound in the very first instruction and the function
//     returns ~2.4e38. -Inf happens to give the correct +0.0.
//
// The evaluation keeps up to five vector groups live (the caller's argument, the
// reduced x, n, and the two polynomial temporaries), so an LMUL=8 instantiation
// would need every architectural vector register and spill. LMUL=4 is the widest
// group that leaves the register allocator room.

#define TQT_RVV_EXP_HI 88.3762626647949f
#define TQT_RVV_EXP_LO -88.3762626647949f

#define TQT_RVV_EXP_LOG2EF 1.44269504088896341f
#define TQT_RVV_EXP_C1 0.693359375f
#define TQT_RVV_EXP_C2 -2.12194440e-4f

#define TQT_RVV_EXP_P0 1.9875691500E-4f
#define TQT_RVV_EXP_P1 1.3981999507E-3f
#define TQT_RVV_EXP_P2 8.3334519073E-3f
#define TQT_RVV_EXP_P3 4.1665795894E-2f
#define TQT_RVV_EXP_P4 1.6666665459E-1f
#define TQT_RVV_EXP_P5 5.0000001201E-1f

/// Generates `tqt_vexp_f32m<LMUL>(vfloat32m<LMUL>_t x, size_t vl)`.
///
/// @param LMUL Vector length multiplier of the operand group (1, 2, 4 or 8).
/// @param MLEN Mask length matching LMUL for 32-bit elements (32 / LMUL).
#define TQT_RVV_F32_EXP_OP(LMUL, MLEN)                                                      \
    static inline vfloat32m##LMUL##_t tqt_vexp_f32m##LMUL(vfloat32m##LMUL##_t x, size_t vl) \
    {                                                                                       \
        vfloat32m##LMUL##_t fx, tmp, y, z;                                                  \
                                                                                            \
        /* Saturate so the 2^n reconstruction can neither overflow nor wrap. */             \
        x = __riscv_vfmin_vf_f32m##LMUL(x, TQT_RVV_EXP_HI, vl);                             \
        x = __riscv_vfmax_vf_f32m##LMUL(x, TQT_RVV_EXP_LO, vl);                             \
                                                                                            \
        /* n = floor(x * log2(e) + 0.5): round to nearest, then correct down. */            \
        fx = __riscv_vfmv_v_f_f32m##LMUL(0.5f, vl);                                         \
        fx = __riscv_vfmacc_vf_f32m##LMUL(fx, TQT_RVV_EXP_LOG2EF, x, vl);                   \
        tmp = __riscv_vfcvt_f_x_v_f32m##LMUL(__riscv_vfcvt_x_f_v_i32m##LMUL(fx, vl), vl);   \
        vbool##MLEN##_t mask = __riscv_vmfgt_vv_f32m##LMUL##_b##MLEN(tmp, fx, vl);          \
        fx = __riscv_vfsub_vf_f32m##LMUL##_mu(mask, tmp, tmp, 1.0f, vl);                    \
                                                                                            \
        /* g = x - n * ln(2), with ln(2) split into C1 + C2. */                             \
        tmp = __riscv_vfmul_vf_f32m##LMUL(fx, TQT_RVV_EXP_C1, vl);                          \
        z = __riscv_vfmul_vf_f32m##LMUL(fx, TQT_RVV_EXP_C2, vl);                            \
        x = __riscv_vfsub_vv_f32m##LMUL(x, tmp, vl);                                        \
        x = __riscv_vfsub_vv_f32m##LMUL(x, z, vl);                                          \
                                                                                            \
        /* exp(g) = 1 + g + g^2 * poly5(g). */                                              \
        y = __riscv_vfmul_vf_f32m##LMUL(x, TQT_RVV_EXP_P0, vl);                             \
        z = __riscv_vfmul_vv_f32m##LMUL(x, x, vl);                                          \
        y = __riscv_vfadd_vf_f32m##LMUL(y, TQT_RVV_EXP_P1, vl);                             \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                          \
        y = __riscv_vfadd_vf_f32m##LMUL(y, TQT_RVV_EXP_P2, vl);                             \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                          \
        y = __riscv_vfadd_vf_f32m##LMUL(y, TQT_RVV_EXP_P3, vl);                             \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                          \
        y = __riscv_vfadd_vf_f32m##LMUL(y, TQT_RVV_EXP_P4, vl);                             \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                          \
        y = __riscv_vfadd_vf_f32m##LMUL(y, TQT_RVV_EXP_P5, vl);                             \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                                          \
        y = __riscv_vfadd_vv_f32m##LMUL(y, x, vl);                                          \
        y = __riscv_vfadd_vf_f32m##LMUL(y, 1.0f, vl);                                       \
                                                                                            \
        /* 2^n by writing the biased exponent field directly. */                            \
        vint32m##LMUL##_t mm = __riscv_vfcvt_x_f_v_i32m##LMUL(fx, vl);                      \
        mm = __riscv_vadd_vx_i32m##LMUL(mm, 0x7f, vl);                                      \
        mm = __riscv_vsll_vx_i32m##LMUL(mm, 23, vl);                                        \
        vfloat32m##LMUL##_t pow2n = __riscv_vreinterpret_v_i32m##LMUL##_f32m##LMUL(mm);     \
                                                                                            \
        return __riscv_vfmul_vv_f32m##LMUL(y, pow2n, vl);                                   \
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
