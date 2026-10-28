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

// Every live vector value belongs to one extended asm block. No register state
// crosses a C++ call or a separately scheduled asm statement.
// Vector register groups (the second row is used only when Rows == 2):
// v0-v3: decode and parameter temporaries, including f32 super-block metadata;
// v4-v5 / v6-v7: i16 group dots; v8-v11 / v12-v15: f32 results;
// v16-v19 / v20-v23: i32 scaled dots. All vector registers are conservatively clobbered.
// a0/a1: packed A/B; a2: group scales; s1: super scales;
// s3/s4: activation scales; ft0/ft1: loaded row scales.
// A 16-term i8*i3 dot is in [-8128, 8192]. Signed i8 scales fit an i32
// super-block accumulator. Low/high planes are reused with immediate MAC.
// Each 16-value group has four low2 byte planes followed by two high1 byte planes.
// For slot s, low2 comes from plane s / 4 and high1 from plane 4 + s / 8.
// Decode: (low2 | (high1 << 2)) - 4, equivalent to low2 - ((high1 ^ 1) << 2).
// A cleared high bit selects [-4, -1]; a set high bit selects [0, 3].
// Machine-word argument block consumed by RV64 assembly via offsetof and ld.
// Address fields retain the pointee types of the typed wrapper parameters.
// Row 0/1 below denotes the current row pair, not the whole M tile.
struct tqt_q3_row_args
{
    uintptr_t a;        // const int8_t*: packed A at row 0 and the current K origin.
    uintptr_t b;        // const uint8_t*: packed B at the current K origin.
    uintptr_t sa0;      // const float*: row-0 scales, one per 256 K values.
    uintptr_t sa1;      // const float*: row-1 scales.
    uintptr_t qs;       // const int8_t*: group scales at lane 0 and the first K group.
    uintptr_t ss;       // const float*: super scales at lane 0 and the first K block.
    uintptr_t c0;       // const float*: optional row-0 accumulation input.
    uintptr_t c1;       // const float*: optional row-1 accumulation input.
    uintptr_t d0;       // float*: row-0 output.
    uintptr_t d1;       // float*: row-1 output.
    uintptr_t bias;     // const float*: optional bias shared by both rows.
    uintptr_t blocks;   // Number of complete 256-value K blocks in this call.
    uintptr_t mr;       // Rows in the whole M tile; byte stride between packed A K values.
    uintptr_t nr;       // Active N lanes; byte stride between packed B byte planes.
    uintptr_t qstride;  // Byte stride between N lanes in group scale metadata.
    uintptr_t sstride;  // Byte stride between N lanes in super scale metadata.
    uintptr_t clamp;    // Machine-word Boolean: 0 skips clamp, 1 applies clamp.
};

template <size_t Rows>
static inline void tqt_q3_rows_rvv(const tqt_q3_row_args &args, float clamp_min, float clamp_max)
{
    static_assert(Rows == 1 || Rows == 2, "Only one or two rows are supported");
    static_assert(sizeof(uintptr_t) == 8, "This kernel requires RV64");
    asm volatile(
        // Load row-pair parameters and initialize the f32 results, including K == 0.
        "ld a0, %c[a](%[args])\n\t"
        "ld a1, %c[b](%[args])\n\t"
        "ld a2, %c[qs](%[args])\n\t"
        "ld a7, %c[blocks](%[args])\n\t"
        "ld t3, %c[mr](%[args])\n\t"
        "ld t4, %c[nr](%[args])\n\t"
        "ld t5, %c[qstride](%[args])\n\t"
        "ld t6, %c[sstride](%[args])\n\t"
        "ld s1, %c[ss](%[args])\n\t"
        "ld s3, %c[sa0](%[args])\n\t"
        "ld s4, %c[sa1](%[args])\n\t"
        "vsetvli zero, t4, e32, m4, ta, ma\n\t"
        "vmv.v.i v8, 0\n\t"
        ".if %c[pair]\n\t"
        "vmv.v.i v12, 0\n\t"
        ".endif\n\t"
        "beqz a7, 3f\n\t"
        // Super-block entry and its backedge both retain e32/m4.
        "1:\n\t"
        "vmv.v.i v16, 0\n\t"
        ".if %c[pair]\n\t"
        "vmv.v.i v20, 0\n\t"
        ".endif\n\t"
        "li a6, 16\n\t"
        // Set e16/m2 once per super-block; the group backedge retains this vtype.
        "vsetvli zero, t4, e16, m2, ta, ma\n\t"
        "2:\n\t"
        // Decode 16 i3 values and reuse each B vector for both row dots.
        // Each group advances packed A by 16 * mr bytes and B by 6 * nr bytes.
        "vmv.v.i v4, 0\n\t"
        ".if %c[pair]\n\t"
        "vmv.v.i v6, 0\n\t"
        ".endif\n\t"
        "vsetvli zero, t4, e8, m1, ta, ma\n\t"
        // Slots 0-7 share the first high1 plane and consume two low2 planes.
        "mv t2, a1\n\t"
        "add t2, t2, t4\n\t"
        "add t2, t2, t4\n\t"
        "add t2, t2, t4\n\t"
        "add t2, t2, t4\n\t"
        "vle8.v v2, (t2)\n\t"
        "vxor.vi v2, v2, -1\n\t"
        "vle8.v v0, (a1)\n\t"
        "add a1, a1, t4\n\t"
        "vsrl.vi v1, v0, 0\n\t"
        "vand.vi v1, v1, 3\n\t"
        "vsrl.vi v3, v2, 0\n\t"
        "vand.vi v3, v3, 1\n\t"
        "vsll.vi v3, v3, 2\n\t"
        "vsub.vv v1, v1, v3\n\t"
        "lb t0, 0(a0)\n\t"
        "vwmacc.vx v4, t0, v1\n\t"
        ".if %c[pair]\n\t"
        "lb t1, 1(a0)\n\t"
        "vwmacc.vx v6, t1, v1\n\t"
        ".endif\n\t"
        "add a0, a0, t3\n\t"
        "vsrl.vi v1, v0, 2\n\t"
        "vand.vi v1, v1, 3\n\t"
        "vsrl.vi v3, v2, 1\n\t"
        "vand.vi v3, v3, 1\n\t"
        "vsll.vi v3, v3, 2\n\t"
        "vsub.vv v1, v1, v3\n\t"
        "lb t0, 0(a0)\n\t"
        "vwmacc.vx v4, t0, v1\n\t"
        ".if %c[pair]\n\t"
        "lb t1, 1(a0)\n\t"
        "vwmacc.vx v6, t1, v1\n\t"
        ".endif\n\t"
        "add a0, a0, t3\n\t"
        "vsrl.vi v1, v0, 4\n\t"
        "vand.vi v1, v1, 3\n\t"
        "vsrl.vi v3, v2, 2\n\t"
        "vand.vi v3, v3, 1\n\t"
        "vsll.vi v3, v3, 2\n\t"
        "vsub.vv v1, v1, v3\n\t"
        "lb t0, 0(a0)\n\t"
        "vwmacc.vx v4, t0, v1\n\t"
        ".if %c[pair]\n\t"
        "lb t1, 1(a0)\n\t"
        "vwmacc.vx v6, t1, v1\n\t"
        ".endif\n\t"
        "add a0, a0, t3\n\t"
        "vsrl.vi v1, v0, 6\n\t"
        "vand.vi v1, v1, 3\n\t"
        "vsrl.vi v3, v2, 3\n\t"
        "vand.vi v3, v3, 1\n\t"
        "vsll.vi v3, v3, 2\n\t"
        "vsub.vv v1, v1, v3\n\t"
        "lb t0, 0(a0)\n\t"
        "vwmacc.vx v4, t0, v1\n\t"
        ".if %c[pair]\n\t"
        "lb t1, 1(a0)\n\t"
        "vwmacc.vx v6, t1, v1\n\t"
        ".endif\n\t"
        "add a0, a0, t3\n\t"
        "vle8.v v0, (a1)\n\t"
        "add a1, a1, t4\n\t"
        "vsrl.vi v1, v0, 0\n\t"
        "vand.vi v1, v1, 3\n\t"
        "vsrl.vi v3, v2, 4\n\t"
        "vand.vi v3, v3, 1\n\t"
        "vsll.vi v3, v3, 2\n\t"
        "vsub.vv v1, v1, v3\n\t"
        "lb t0, 0(a0)\n\t"
        "vwmacc.vx v4, t0, v1\n\t"
        ".if %c[pair]\n\t"
        "lb t1, 1(a0)\n\t"
        "vwmacc.vx v6, t1, v1\n\t"
        ".endif\n\t"
        "add a0, a0, t3\n\t"
        "vsrl.vi v1, v0, 2\n\t"
        "vand.vi v1, v1, 3\n\t"
        "vsrl.vi v3, v2, 5\n\t"
        "vand.vi v3, v3, 1\n\t"
        "vsll.vi v3, v3, 2\n\t"
        "vsub.vv v1, v1, v3\n\t"
        "lb t0, 0(a0)\n\t"
        "vwmacc.vx v4, t0, v1\n\t"
        ".if %c[pair]\n\t"
        "lb t1, 1(a0)\n\t"
        "vwmacc.vx v6, t1, v1\n\t"
        ".endif\n\t"
        "add a0, a0, t3\n\t"
        "vsrl.vi v1, v0, 4\n\t"
        "vand.vi v1, v1, 3\n\t"
        "vsrl.vi v3, v2, 6\n\t"
        "vand.vi v3, v3, 1\n\t"
        "vsll.vi v3, v3, 2\n\t"
        "vsub.vv v1, v1, v3\n\t"
        "lb t0, 0(a0)\n\t"
        "vwmacc.vx v4, t0, v1\n\t"
        ".if %c[pair]\n\t"
        "lb t1, 1(a0)\n\t"
        "vwmacc.vx v6, t1, v1\n\t"
        ".endif\n\t"
        "add a0, a0, t3\n\t"
        "vsrl.vi v1, v0, 6\n\t"
        "vand.vi v1, v1, 3\n\t"
        "vsrl.vi v3, v2, 7\n\t"
        "vand.vi v3, v3, 1\n\t"
        "vsll.vi v3, v3, 2\n\t"
        "vsub.vv v1, v1, v3\n\t"
        "lb t0, 0(a0)\n\t"
        "vwmacc.vx v4, t0, v1\n\t"
        ".if %c[pair]\n\t"
        "lb t1, 1(a0)\n\t"
        "vwmacc.vx v6, t1, v1\n\t"
        ".endif\n\t"
        "add a0, a0, t3\n\t"
        // Slots 8-15 share the second high1 plane and consume the remaining low2 planes.
        "mv t2, a1\n\t"
        "add t2, t2, t4\n\t"
        "add t2, t2, t4\n\t"
        "add t2, t2, t4\n\t"
        "vle8.v v2, (t2)\n\t"
        "vxor.vi v2, v2, -1\n\t"
        "vle8.v v0, (a1)\n\t"
        "add a1, a1, t4\n\t"
        "vsrl.vi v1, v0, 0\n\t"
        "vand.vi v1, v1, 3\n\t"
        "vsrl.vi v3, v2, 0\n\t"
        "vand.vi v3, v3, 1\n\t"
        "vsll.vi v3, v3, 2\n\t"
        "vsub.vv v1, v1, v3\n\t"
        "lb t0, 0(a0)\n\t"
        "vwmacc.vx v4, t0, v1\n\t"
        ".if %c[pair]\n\t"
        "lb t1, 1(a0)\n\t"
        "vwmacc.vx v6, t1, v1\n\t"
        ".endif\n\t"
        "add a0, a0, t3\n\t"
        "vsrl.vi v1, v0, 2\n\t"
        "vand.vi v1, v1, 3\n\t"
        "vsrl.vi v3, v2, 1\n\t"
        "vand.vi v3, v3, 1\n\t"
        "vsll.vi v3, v3, 2\n\t"
        "vsub.vv v1, v1, v3\n\t"
        "lb t0, 0(a0)\n\t"
        "vwmacc.vx v4, t0, v1\n\t"
        ".if %c[pair]\n\t"
        "lb t1, 1(a0)\n\t"
        "vwmacc.vx v6, t1, v1\n\t"
        ".endif\n\t"
        "add a0, a0, t3\n\t"
        "vsrl.vi v1, v0, 4\n\t"
        "vand.vi v1, v1, 3\n\t"
        "vsrl.vi v3, v2, 2\n\t"
        "vand.vi v3, v3, 1\n\t"
        "vsll.vi v3, v3, 2\n\t"
        "vsub.vv v1, v1, v3\n\t"
        "lb t0, 0(a0)\n\t"
        "vwmacc.vx v4, t0, v1\n\t"
        ".if %c[pair]\n\t"
        "lb t1, 1(a0)\n\t"
        "vwmacc.vx v6, t1, v1\n\t"
        ".endif\n\t"
        "add a0, a0, t3\n\t"
        "vsrl.vi v1, v0, 6\n\t"
        "vand.vi v1, v1, 3\n\t"
        "vsrl.vi v3, v2, 3\n\t"
        "vand.vi v3, v3, 1\n\t"
        "vsll.vi v3, v3, 2\n\t"
        "vsub.vv v1, v1, v3\n\t"
        "lb t0, 0(a0)\n\t"
        "vwmacc.vx v4, t0, v1\n\t"
        ".if %c[pair]\n\t"
        "lb t1, 1(a0)\n\t"
        "vwmacc.vx v6, t1, v1\n\t"
        ".endif\n\t"
        "add a0, a0, t3\n\t"
        "vle8.v v0, (a1)\n\t"
        "add a1, a1, t4\n\t"
        "vsrl.vi v1, v0, 0\n\t"
        "vand.vi v1, v1, 3\n\t"
        "vsrl.vi v3, v2, 4\n\t"
        "vand.vi v3, v3, 1\n\t"
        "vsll.vi v3, v3, 2\n\t"
        "vsub.vv v1, v1, v3\n\t"
        "lb t0, 0(a0)\n\t"
        "vwmacc.vx v4, t0, v1\n\t"
        ".if %c[pair]\n\t"
        "lb t1, 1(a0)\n\t"
        "vwmacc.vx v6, t1, v1\n\t"
        ".endif\n\t"
        "add a0, a0, t3\n\t"
        "vsrl.vi v1, v0, 2\n\t"
        "vand.vi v1, v1, 3\n\t"
        "vsrl.vi v3, v2, 5\n\t"
        "vand.vi v3, v3, 1\n\t"
        "vsll.vi v3, v3, 2\n\t"
        "vsub.vv v1, v1, v3\n\t"
        "lb t0, 0(a0)\n\t"
        "vwmacc.vx v4, t0, v1\n\t"
        ".if %c[pair]\n\t"
        "lb t1, 1(a0)\n\t"
        "vwmacc.vx v6, t1, v1\n\t"
        ".endif\n\t"
        "add a0, a0, t3\n\t"
        "vsrl.vi v1, v0, 4\n\t"
        "vand.vi v1, v1, 3\n\t"
        "vsrl.vi v3, v2, 6\n\t"
        "vand.vi v3, v3, 1\n\t"
        "vsll.vi v3, v3, 2\n\t"
        "vsub.vv v1, v1, v3\n\t"
        "lb t0, 0(a0)\n\t"
        "vwmacc.vx v4, t0, v1\n\t"
        ".if %c[pair]\n\t"
        "lb t1, 1(a0)\n\t"
        "vwmacc.vx v6, t1, v1\n\t"
        ".endif\n\t"
        "add a0, a0, t3\n\t"
        "vsrl.vi v1, v0, 6\n\t"
        "vand.vi v1, v1, 3\n\t"
        "vsrl.vi v3, v2, 7\n\t"
        "vand.vi v3, v3, 1\n\t"
        "vsll.vi v3, v3, 2\n\t"
        "vsub.vv v1, v1, v3\n\t"
        "lb t0, 0(a0)\n\t"
        "vwmacc.vx v4, t0, v1\n\t"
        ".if %c[pair]\n\t"
        "lb t1, 1(a0)\n\t"
        "vwmacc.vx v6, t1, v1\n\t"
        ".endif\n\t"
        "add a0, a0, t3\n\t"
        // Skip the two high1 planes already consumed through t2.
        "add a1, a1, t4\n\t"
        "add a1, a1, t4\n\t"
        // Scale the i16 group dots into i32 super-block accumulators.
        "vlse8.v v0, (a2), t5\n\t"
        "vsetvli zero, t4, e16, m2, ta, ma\n\t"
        "vsext.vf2 v2, v0\n\t"
        "vwmacc.vv v16, v4, v2\n\t"
        ".if %c[pair]\n\t"
        "vwmacc.vv v20, v6, v2\n\t"
        ".endif\n\t"
        // Advance K groups by one byte; strided vector loads traverse N lanes.
        "addi a2, a2, 1\n\t"
        "addi a6, a6, -1\n\t"
        "bnez a6, 2b\n\t"
        // Finish this super-block in f32 and advance its float metadata by one element.
        "vsetvli zero, t4, e32, m4, ta, ma\n\t"
        "flw ft0, 0(s3)\n\t"
        "addi s3, s3, 4\n\t"
        "vfcvt.f.x.v v16, v16\n\t"
        "vfmul.vf v16, v16, ft0\n\t"
        "vlse32.v v0, (s1), t6\n\t"
        "vfmacc.vv v8, v16, v0\n\t"
        ".if %c[pair]\n\t"
        "flw ft1, 0(s4)\n\t"
        "addi s4, s4, 4\n\t"
        "vfcvt.f.x.v v20, v20\n\t"
        "vfmul.vf v20, v20, ft1\n\t"
        "vfmacc.vv v12, v20, v0\n\t"
        ".endif\n\t"
        "addi s1, s1, 4\n\t"
        "addi a7, a7, -1\n\t"
        "bnez a7, 1b\n\t"
        // Epilogue: optional C, shared bias, optional clamp, then row stores.
        "3:\n\t"
        "ld t2, %c[c0](%[args])\n\t"
        "beqz t2, 4f\n\t"
        "vle32.v v0, (t2)\n\t"
        "vfadd.vv v8, v8, v0\n\t"
        ".if %c[pair]\n\t"
        "ld t2, %c[c1](%[args])\n\t"
        "vle32.v v0, (t2)\n\t"
        "vfadd.vv v12, v12, v0\n\t"
        ".endif\n\t"
        "4:\n\t"
        "ld t2, %c[bias](%[args])\n\t"
        "beqz t2, 5f\n\t"
        "vle32.v v0, (t2)\n\t"
        "vfadd.vv v8, v8, v0\n\t"
        ".if %c[pair]\n\t"
        "vfadd.vv v12, v12, v0\n\t"
        ".endif\n\t"
        "5:\n\t"
        "ld t2, %c[clamp](%[args])\n\t"
        "beqz t2, 6f\n\t"
        "vfmax.vf v8, v8, %[lo]\n\t"
        "vfmin.vf v8, v8, %[hi]\n\t"
        ".if %c[pair]\n\t"
        "vfmax.vf v12, v12, %[lo]\n\t"
        "vfmin.vf v12, v12, %[hi]\n\t"
        ".endif\n\t"
        "6:\n\t"
        "ld t2, %c[d0](%[args])\n\t"
        "vse32.v v8, (t2)\n\t"
        ".if %c[pair]\n\t"
        "ld t2, %c[d1](%[args])\n\t"
        "vse32.v v12, (t2)\n\t"
        ".endif\n\t"
        :
        : [args] "r"(&args), [pair] "i"(Rows == 2), [lo] "f"(clamp_min), [hi] "f"(clamp_max),
          [a] "i"(offsetof(tqt_q3_row_args, a)), [b] "i"(offsetof(tqt_q3_row_args, b)),
          [sa0] "i"(offsetof(tqt_q3_row_args, sa0)), [sa1] "i"(offsetof(tqt_q3_row_args, sa1)),
          [qs] "i"(offsetof(tqt_q3_row_args, qs)), [ss] "i"(offsetof(tqt_q3_row_args, ss)),
          [c0] "i"(offsetof(tqt_q3_row_args, c0)), [c1] "i"(offsetof(tqt_q3_row_args, c1)),
          [d0] "i"(offsetof(tqt_q3_row_args, d0)), [d1] "i"(offsetof(tqt_q3_row_args, d1)),
          [bias] "i"(offsetof(tqt_q3_row_args, bias)),
          [blocks] "i"(offsetof(tqt_q3_row_args, blocks)), [mr] "i"(offsetof(tqt_q3_row_args, mr)),
          [nr] "i"(offsetof(tqt_q3_row_args, nr)),
          [qstride] "i"(offsetof(tqt_q3_row_args, qstride)),
          [sstride] "i"(offsetof(tqt_q3_row_args, sstride)),
          [clamp] "i"(offsetof(tqt_q3_row_args, clamp))
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
          "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
          "v27", "v28", "v29", "v30", "v31", "a0", "a1", "a2", "a6", "a7", "t0", "t1", "t2", "t3",
          "t4", "t5", "t6", "s1", "s3", "s4", "ft0", "ft1", "memory");
}

static inline void tqt_compute_tile_kernel_gemm_qsi8d256_qsi3c16_4x1vl_rvv(
    size_t mr, size_t nr, size_t k, size_t k_idx, float *d, const int8_t *a_packed,
    const uint8_t *b_packed, const float *scale_a, size_t ldsa, const int8_t *qscale_b, size_t ldqb,
    const float *super_scale_b, size_t ldssb, const float *c, const float *bias, size_t ldc,
    size_t ldd, float clamp_min, float clamp_max, bool apply_clamp)
{
    // The public entry guarantees that k and k_idx are multiples of 256.
    const size_t sb = k_idx / 256;
    const size_t group = k_idx / 16;
    for (size_t row = 0; row < mr; row += 2) {
        const bool pair = row + 1 < mr;
        const size_t next = pair ? row + 1 : row;
        const tqt_q3_row_args args = {reinterpret_cast<uintptr_t>(a_packed) + row,
                                      reinterpret_cast<uintptr_t>(b_packed),
                                      reinterpret_cast<uintptr_t>(scale_a + row * ldsa + sb),
                                      reinterpret_cast<uintptr_t>(scale_a + next * ldsa + sb),
                                      reinterpret_cast<uintptr_t>(qscale_b + group),
                                      reinterpret_cast<uintptr_t>(super_scale_b + sb),
                                      reinterpret_cast<uintptr_t>(c ? c + row * ldc : nullptr),
                                      reinterpret_cast<uintptr_t>(c ? c + next * ldc : nullptr),
                                      reinterpret_cast<uintptr_t>(d + row * ldd),
                                      reinterpret_cast<uintptr_t>(d + next * ldd),
                                      reinterpret_cast<uintptr_t>(bias),
                                      k / 256,
                                      mr,
                                      nr,
                                      ldqb,
                                      ldssb * sizeof(float),
                                      static_cast<uintptr_t>(apply_clamp)};
        if (pair) {
            tqt_q3_rows_rvv<2>(args, clamp_min, clamp_max);
        } else {
            tqt_q3_rows_rvv<1>(args, clamp_min, clamp_max);
        }
    }
}
