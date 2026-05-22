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
#include "src/gemm/gemm_i8_i8/tqt_kernel_gemm_i8_i8_4x1vl_rvv.h"

/// Igemmcf 1vlx4 kernel — additional function for strided D store.
///
/// Register allocation (reuses GEMM i32_i8_i8 4x1vl layout with M/N roles swapped):
///   v0:      Weight vector (int8 LMUL=1)
///   a0-a3:   Input scalars for NR spatial positions
///   v16-v19: Accumulator row 0 (int32 LMUL=4)
///   v20-v23: Accumulator row 1 (int32 LMUL=4)
///   v24-v27: Accumulator row 2 (int32 LMUL=4)
///   v28-v31: Accumulator row 3 (int32 LMUL=4)
///   v4-v7:   Narrowed int8 results after requantize (int8 LMUL=1)
///
/// Template parameters:
///   MR = number of OC vector groups (always 1 for i8 4x1vl)
///   NR = number of scalar spatial positions (1-4)
///
/// Reused from tqt_kernel_gemm_i8_i8_4x1vl_rvv.h (call with swapped <NR>):
///   tqt_init_zero_kernel_gemm<NR>          -- zero accumulators
///   tqt_load_b_kernel_gemm                 -- load packed weight vector into v0
///   tqt_load_bt_kernel_gemm                -- load unpacked weight vector (strided)
///   tqt_vmacc_kernel_gemm<NR>              -- widening multiply-accumulate
///   tqt_add_mx1bias_kernel_gemm<NR>        -- per-spatial bias (scalar broadcast add)
///   tqt_requantize_mx1_kernel_gemm<NR>     -- per-row requantize (each spatial position)
///   tqt_clamp_i8_kernel_gemm<NR>           -- clamp all i8 results

/// Store int8 results to D matrix with strided stores.
/// D layout: [OC, OD*OH*OW] (NCDHW channel-first), consecutive OC values are stride_d apart.
/// After requantize, results are in v4 (row 0), v5 (row 1), v6 (row 2), v7 (row 3) — LMUL=1.
/// For each spatial position nr:
///   vsse8.v v{4+nr}, (D + nr), stride_d * sizeof(int8_t)
template <size_t MR, size_t NR>
static inline void tqt_store_i8_strided_kernel_igemmcf_i8_i8_1vlx4_rvv(size_t vl, int8_t *D,
                                                                       size_t stride_d)
{
    static_assert(MR == 1, "MR must be 1 for i8 igemmcf");
    static_assert(NR >= 1 && NR <= 4, "NR must be in range [1, 4]");

    size_t stride_bytes = stride_d * sizeof(int8_t);
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma\n\t" : : "r"(vl) :);

    // nr=0: v4
    if constexpr (NR >= 1)
        asm volatile("vsse8.v v4, (%0), %1" : : "r"(D + 0), "r"(stride_bytes) : "memory");
    // nr=1: v5
    if constexpr (NR >= 2)
        asm volatile("vsse8.v v5, (%0), %1" : : "r"(D + 1), "r"(stride_bytes) : "memory");
    // nr=2: v6
    if constexpr (NR >= 3)
        asm volatile("vsse8.v v6, (%0), %1" : : "r"(D + 2), "r"(stride_bytes) : "memory");
    // nr=3: v7
    if constexpr (NR >= 4)
        asm volatile("vsse8.v v7, (%0), %1" : : "r"(D + 3), "r"(stride_bytes) : "memory");
}
