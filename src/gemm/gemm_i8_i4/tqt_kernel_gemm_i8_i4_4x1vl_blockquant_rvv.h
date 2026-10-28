//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include <stddef.h>
#include <stdint.h>

#include "src/common/tqt_common.h"
// Pull in shared epilogue helpers (add_bias/add_c/clamp/store for f32/f16/bf16).
#include "src/gemm/gemm_i8_i4/tqt_kernel_gemm_i8_i4_4x1vl_rvv.h"

/// Register allocation for 4x1vl blockquant W4A8 GEMM kernel (full fp32 acc):
///   v0:      B lo nibble buffer (uint8 m1)         -- K-pair scratch
///   v1:      B hi nibble buffer (uint8 m1)         -- K-pair scratch
///   v2:      B raw byte load destination (uint8 m1) -- K-pair scratch
///   v3:      LUT lookup result (int8 m1)
///   v4-v7:   offset_b vector or scratch (e32 m4)       -- K-pair scratch
///   v8-v11:  B dequant fp32 lane (e32 m4)          -- K-pair scratch (lo/hi
///                                                    serialized through this
///                                                    slot, fmacc consumes
///                                                    before the next dequant)
///   v12-v15: scale_b vector (e32 m4)               -- block-constant, loaded
///                                                    once per K-block
///   v16-v19: fp32 global accumulator row 0 (e32 m4)
///   v20-v23: fp32 global accumulator row 1 (e32 m4)
///   v24-v27: fp32 global accumulator row 2 (e32 m4)
///   v28-v31: fp32 global accumulator row 3 (e32 m4)
///
/// Accumulator is fp32 throughout. The K loop body dequantizes one nibble
/// position at a time into v8-v11 and immediately folds it into v16-v31 via
/// vfmacc.vf with a per-row a-scalar already cast to fp32 (and, for the qai8dx
/// family, offset by -zp_a[m]; for the qsi8d2 family, multiplied by the
/// per-block scale_a[m, blk]).
///
/// Epilogue uses the shared helpers from tqt_kernel_gemm_i8_i4_4x1vl_rvv.h:
///   tqt_add_1xnbias_{f32,f16,bf16}_kernel_gemm_i8_i4_4x1vl_rvv<MR>
///   tqt_add_c_{f32,f16,bf16}_kernel_gemm_i8_i4_4x1vl_rvv<MR>
///   tqt_clamp_f32_kernel_gemm_i8_i4_4x1vl_rvv<MR>
///   tqt_store_f32_kernel_gemm_i8_i4_4x1vl_rvv<MR>
///   tqt_store_narrow_{f16,bf16}_kernel_gemm_i8_i4_4x1vl_rvv<MR>

// ============================================================================
// Initialize fp32 global accumulators to zero (v16/v20/v24/v28, e32 m4)
// ============================================================================

template <size_t MR>
static inline void tqt_init_fzero_acc_kernel_gemm_i8_i4_4x1vl_blockquant_rvv(size_t vl)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    if constexpr (MR >= 1)
        asm volatile("vmv.v.i v16, 0\n\t" : : : "v16", "v17", "v18", "v19");
    if constexpr (MR >= 2)
        asm volatile("vmv.v.i v20, 0\n\t" : : : "v20", "v21", "v22", "v23");
    if constexpr (MR >= 3)
        asm volatile("vmv.v.i v24, 0\n\t" : : : "v24", "v25", "v26", "v27");
    if constexpr (MR >= 4)
        asm volatile("vmv.v.i v28, 0\n\t" : : : "v28", "v29", "v30", "v31");
}

// ============================================================================
// Load the per-block scale_b vector into v12-v15.
//
// scale_b is laid out as [n, k/bl] fp32 row-major. For a given block index
// blk, the kernel needs the lane (n_lane, blk) for each n in the current tile.
// Loads with vlse32.v at stride = ldsb * sizeof(float).
// ============================================================================

static inline void tqt_load_scale_b_block_kernel_gemm_i8_i4_4x1vl_blockquant_rvv(
    size_t vl, const float *scale_b_lane, size_t scale_byte_stride)
{
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);
    asm volatile("vlse32.v v12, (%0), %1"
                 :
                 : "r"(scale_b_lane), "r"(scale_byte_stride)
                 : "v12", "v13", "v14", "v15", "memory");
}

// ============================================================================
// Dequantize and accumulate both nibbles while retaining the raw B byte and,
// for qai4, the zero-point vector across the complete K pair.
// ============================================================================

#define TQT_VFMACC_LO_ROWS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV \
    "vfmacc.vf v16, %[a_lo_0], v8\n\t"                            \
    ".if %[mr] >= 2\n\t"                                          \
    "vfmacc.vf v20, %[a_lo_1], v8\n\t"                            \
    ".endif\n\t"                                                  \
    ".if %[mr] >= 3\n\t"                                          \
    "vfmacc.vf v24, %[a_lo_2], v8\n\t"                            \
    ".endif\n\t"                                                  \
    ".if %[mr] >= 4\n\t"                                          \
    "vfmacc.vf v28, %[a_lo_3], v8\n\t"                            \
    ".endif\n\t"

#define TQT_VFMACC_HI_ROWS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV \
    "vfmacc.vf v16, %[a_hi_0], v8\n\t"                            \
    ".if %[mr] >= 2\n\t"                                          \
    "vfmacc.vf v20, %[a_hi_1], v8\n\t"                            \
    ".endif\n\t"                                                  \
    ".if %[mr] >= 3\n\t"                                          \
    "vfmacc.vf v24, %[a_hi_2], v8\n\t"                            \
    ".endif\n\t"                                                  \
    ".if %[mr] >= 4\n\t"                                          \
    "vfmacc.vf v28, %[a_hi_3], v8\n\t"                            \
    ".endif\n\t"

#define TQT_QSI4_PAIR_BODY_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV(B_LOAD)                \
    "vsetvli zero, %[vl], e8, m1, ta, ma\n\t" B_LOAD                                     \
    "vand.vi v0, v2, 0xf\n\t"                                                            \
    "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"                                           \
    "vzext.vf4 v8, v0\n\t"                                                               \
    "vfcvt.f.x.v v8, v8\n\t"                                                             \
    "vfadd.vf v8, v8, %[negative_eight]\n\t"                                             \
    "vfmul.vv v8, v8, v12\n\t" TQT_VFMACC_LO_ROWS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV \
    "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"                                            \
    "vsrl.vi v1, v2, 4\n\t"                                                              \
    "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"                                           \
    "vzext.vf4 v8, v1\n\t"                                                               \
    "vfcvt.f.x.v v8, v8\n\t"                                                             \
    "vfadd.vf v8, v8, %[negative_eight]\n\t"                                             \
    "vfmul.vv v8, v8, v12\n\t" TQT_VFMACC_HI_ROWS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV

#define TQT_LUT4_PAIR_BODY_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV(B_LOAD)                \
    "vsetvli zero, %[vl], e8, m1, ta, ma\n\t" B_LOAD                                     \
    "vand.vi v0, v2, 0xf\n\t"                                                            \
    "vluxei8.v v3, (%[lut]), v0\n\t"                                                     \
    "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"                                           \
    "vsext.vf4 v8, v3\n\t"                                                               \
    "vfcvt.f.x.v v8, v8\n\t"                                                             \
    "vfmul.vv v8, v8, v12\n\t" TQT_VFMACC_LO_ROWS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV \
    "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"                                            \
    "vsrl.vi v1, v2, 4\n\t"                                                              \
    "vluxei8.v v3, (%[lut]), v1\n\t"                                                     \
    "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"                                           \
    "vsext.vf4 v8, v3\n\t"                                                               \
    "vfcvt.f.x.v v8, v8\n\t"                                                             \
    "vfmul.vv v8, v8, v12\n\t" TQT_VFMACC_HI_ROWS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV

#define TQT_QAI4_PAIR_BODY_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV(B_LOAD)               \
    "vsetvli zero, %[vl], e8, m1, ta, ma\n\t" B_LOAD                                    \
    "vand.vi v0, v2, 0xf\n\t"                                                           \
    "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"                                          \
    "vlse32.v v4, (%[offset_b_lane]), %[offset_byte_stride]\n\t"                        \
    "vzext.vf4 v8, v0\n\t"                                                              \
    "vfcvt.f.x.v v8, v8\n\t"                                                            \
    "vfmul.vv v8, v8, v12\n\t"                                                          \
    "vfadd.vv v8, v8, v4\n\t" TQT_VFMACC_LO_ROWS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV \
    "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"                                           \
    "vsrl.vi v1, v2, 4\n\t"                                                             \
    "vsetvli zero, %[vl], e32, m4, ta, ma\n\t"                                          \
    "vzext.vf4 v8, v1\n\t"                                                              \
    "vfcvt.f.x.v v8, v8\n\t"                                                            \
    "vfmul.vv v8, v8, v12\n\t"                                                          \
    "vfadd.vv v8, v8, v4\n\t" TQT_VFMACC_HI_ROWS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV

#define TQT_PAIR_ACTIVATION_INPUTS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV(MR, A_LO, A_HI) \
    [mr] "i"(MR), [a_lo_0] "f"((A_LO)[0]), [a_lo_1] "f"((A_LO)[MR >= 2 ? 1 : 0]),         \
        [a_lo_2] "f"((A_LO)[MR >= 3 ? 2 : 0]), [a_lo_3] "f"((A_LO)[MR >= 4 ? 3 : 0]),     \
        [a_hi_0] "f"((A_HI)[0]), [a_hi_1] "f"((A_HI)[MR >= 2 ? 1 : 0]),                   \
        [a_hi_2] "f"((A_HI)[MR >= 3 ? 2 : 0]), [a_hi_3] "f"((A_HI)[MR >= 4 ? 3 : 0])

#define TQT_QSI4_PAIR_CLOBBERS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV                           \
    "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", \
        "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "memory"

#define TQT_QAI4_PAIR_CLOBBERS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV                              \
    "v0", "v1", "v2", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18",       \
        "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", \
        "memory"

template <size_t MR>
static inline void tqt_dot_bt_pair_fp32_qsi4c2_kernel_gemm_i8_i4_4x1vl_blockquant_rvv(
    size_t vl, const uint8_t *b_col, size_t b_byte_stride, const float a_lo[MR],
    const float a_hi[MR], const int8_t *lut)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");
    if (lut) {
        asm volatile(
            TQT_LUT4_PAIR_BODY_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV(
                "vlse8.v v2, (%[b_col]), %[b_byte_stride]\n\t")
            :
            : [vl] "r"(vl), [b_col] "r"(b_col), [b_byte_stride] "r"(b_byte_stride), [lut] "r"(lut),
              TQT_PAIR_ACTIVATION_INPUTS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV(MR, a_lo, a_hi)
            : TQT_QSI4_PAIR_CLOBBERS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV);
    } else {
        asm volatile(
            TQT_QSI4_PAIR_BODY_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV(
                "vlse8.v v2, (%[b_col]), %[b_byte_stride]\n\t")
            :
            : [vl] "r"(vl), [b_col] "r"(b_col), [b_byte_stride] "r"(b_byte_stride),
              [negative_eight] "f"(-8.0f),
              TQT_PAIR_ACTIVATION_INPUTS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV(MR, a_lo, a_hi)
            : TQT_QSI4_PAIR_CLOBBERS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV);
    }
}

template <size_t MR>
static inline void tqt_dot_b_pair_fp32_qsi4c2_kernel_gemm_i8_i4_4x1vl_blockquant_rvv(
    size_t vl, const uint8_t *b_col, const float a_lo[MR], const float a_hi[MR], const int8_t *lut)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");
    if (lut) {
        asm volatile(
            TQT_LUT4_PAIR_BODY_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV("vle8.v v2, (%[b_col])\n\t")
            :
            : [vl] "r"(vl), [b_col] "r"(b_col), [lut] "r"(lut),
              TQT_PAIR_ACTIVATION_INPUTS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV(MR, a_lo, a_hi)
            : TQT_QSI4_PAIR_CLOBBERS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV);
    } else {
        asm volatile(
            TQT_QSI4_PAIR_BODY_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV("vle8.v v2, (%[b_col])\n\t")
            :
            : [vl] "r"(vl), [b_col] "r"(b_col), [negative_eight] "f"(-8.0f),
              TQT_PAIR_ACTIVATION_INPUTS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV(MR, a_lo, a_hi)
            : TQT_QSI4_PAIR_CLOBBERS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV);
    }
}

template <size_t MR>
static inline void tqt_dot_bt_pair_fp32_qai4c2_kernel_gemm_i8_i4_4x1vl_blockquant_rvv(
    size_t vl, const uint8_t *b_col, size_t b_byte_stride, const float *offset_b_lane,
    size_t offset_byte_stride, const float a_lo[MR], const float a_hi[MR])
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");
    asm volatile(TQT_QAI4_PAIR_BODY_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV(
                     "vlse8.v v2, (%[b_col]), %[b_byte_stride]\n\t")
                 :
                 : [vl] "r"(vl), [b_col] "r"(b_col), [b_byte_stride] "r"(b_byte_stride),
                   [offset_b_lane] "r"(offset_b_lane), [offset_byte_stride] "r"(offset_byte_stride),
                   TQT_PAIR_ACTIVATION_INPUTS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV(MR, a_lo, a_hi)
                 : TQT_QAI4_PAIR_CLOBBERS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV);
}

template <size_t MR>
static inline void tqt_dot_b_pair_fp32_qai4c2_kernel_gemm_i8_i4_4x1vl_blockquant_rvv(
    size_t vl, const uint8_t *b_col, const float *offset_b_lane, size_t offset_byte_stride,
    const float a_lo[MR], const float a_hi[MR])
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");
    asm volatile(
        TQT_QAI4_PAIR_BODY_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV("vle8.v v2, (%[b_col])\n\t")
        :
        : [vl] "r"(vl), [b_col] "r"(b_col), [offset_b_lane] "r"(offset_b_lane),
          [offset_byte_stride] "r"(offset_byte_stride),
          TQT_PAIR_ACTIVATION_INPUTS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV(MR, a_lo, a_hi)
        : TQT_QAI4_PAIR_CLOBBERS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV);
}

#undef TQT_QAI4_PAIR_CLOBBERS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV
#undef TQT_QSI4_PAIR_CLOBBERS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV
#undef TQT_PAIR_ACTIVATION_INPUTS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV
#undef TQT_QAI4_PAIR_BODY_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV
#undef TQT_LUT4_PAIR_BODY_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV
#undef TQT_QSI4_PAIR_BODY_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV
#undef TQT_VFMACC_HI_ROWS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV
#undef TQT_VFMACC_LO_ROWS_KERNEL_GEMM_I8_I4_4X1VL_BLOCKQUANT_RVV

// ============================================================================
// Multiply each accumulator row by the per-row scale_a[r] (fp32). Used by
// qai8dx-input kernels where scale_a is per-row and the K loop deferred this
// multiply until the epilogue.
// ============================================================================

template <size_t MR>
static inline void tqt_apply_scale_a_per_row_kernel_gemm_i8_i4_4x1vl_blockquant_rvv(
    size_t vl, const float scale_a[MR])
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma\n\t" : : "r"(vl) :);

    if constexpr (MR >= 1)
        asm volatile("vfmul.vf v16, v16, %0" : : "f"(scale_a[0]) : "v16", "v17", "v18", "v19");
    if constexpr (MR >= 2)
        asm volatile("vfmul.vf v20, v20, %0" : : "f"(scale_a[1]) : "v20", "v21", "v22", "v23");
    if constexpr (MR >= 3)
        asm volatile("vfmul.vf v24, v24, %0" : : "f"(scale_a[2]) : "v24", "v25", "v26", "v27");
    if constexpr (MR >= 4)
        asm volatile("vfmul.vf v28, v28, %0" : : "f"(scale_a[3]) : "v28", "v29", "v30", "v31");
}
