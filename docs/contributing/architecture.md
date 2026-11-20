<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Internal Architecture

This document explains the internal design of TORQ-Tile micro-kernels for advanced
contributors who need to understand the dispatch mechanism, tiling strategy, and
data layout conventions.

## Overview

Each TORQ-Tile micro-kernel is a self-contained function that computes a tile of the
output matrix. The design optimizes for:

1. **Register utilization** -- maximize the compute-to-load ratio by filling all
   available vector registers with accumulator state.
2. **Partial-tile handling** -- RVV kernels handle edge cases (M < m_step,
   N < n_step) through dispatch; Ztt uses the layout-specific contract below.
3. **Stateless execution** -- all state is passed as arguments, enabling the caller to
   partition work across threads without coordination.

## Dispatch Table Pattern

Many RVV kernel variants use a 2D dispatch table indexed by `(mr, nr)`:

```cpp
static constexpr gemm_kernel_fn_t kernel_table[8][3] = {
    {gemm_kernel<1,1>, gemm_kernel<1,2>, gemm_kernel<1,3>},
    {gemm_kernel<2,1>, gemm_kernel<2,2>, gemm_kernel<2,3>},
    // ... up to MR_MAX x NR_MAX
    {gemm_kernel<8,1>, gemm_kernel<8,2>, gemm_kernel<8,3>},
};
```

At call time, the tiling loop computes the actual number of rows (`actual_m`) and
vector groups (`nr_idx`) for the current tile, then dispatches:

```cpp
size_t mr_idx = actual_m - 1;                       // 0-based index
size_t nr_idx = (actual_n + vlmax - 1) / vlmax - 1; // 0-based vector group index
kernel_table[mr_idx][nr_idx](K, D, A, B, C, bias, ...);
```

This avoids runtime conditionals inside the inner loop. Each instantiation of
`gemm_kernel<MR, NR>` is fully specialized at compile time via `if constexpr`,
eliminating dead code for unused register rows/columns.

### Why Not Runtime Branches?

A branch-based approach (checking `if (MR >= 2) ...` at runtime) would:
- Prevent the compiler from allocating registers statically
- Introduce branch mispredictions in the hot path
- Bloat the instruction cache with unreachable code

The dispatch table moves all branching to a single indirect call before the K-loop,
which the branch predictor handles efficiently (it is called repeatedly with the same
target for interior tiles).

## Template Kernel Structure

Each `gemm_kernel<MR, NR>` follows a fixed phase sequence:

```
┌─────────────────────────────────────────────┐
│ 1. INIT: Load C into accumulators (or zero) │
├─────────────────────────────────────────────┤
│ 2. PROLOGUE: Load first A column + B row    │
├─────────────────────────────────────────────┤
│ 3. K-LOOP: Fused load + fmacc (K-1 iters)  │
├─────────────────────────────────────────────┤
│ 4. EPILOGUE: Final fmacc (no next load)     │
├─────────────────────────────────────────────┤
│ 5. BIAS: Add bias vector to accumulators    │
├─────────────────────────────────────────────┤
│ 6. CLAMP: Conditional min/max saturation   │
├─────────────────────────────────────────────┤
│ 7. STORE: Write accumulators to D           │
└─────────────────────────────────────────────┘
```

Regular kernels implement each phase as a separate template function in the shared inner kernel
header (e.g., `tqt_kernel_gemm_f16_f16_8x3vl_rvv.h`). This separation enables
kernel-base variants to customize the epilogue (bias type, clamp, requantize) while
reusing the same init + K-loop + store logic. Floating GEMM and IGEMM entry points compute
`apply_clamp` from their original public bounds before live vector/matrix state, and
pass it privately to the original clamp site. Exactly `[-INFINITY, INFINITY]` skips
min/max. Raw clamp helpers always clamp when called; entry-point policy gates their use. Fused dequant/epilogue helpers retain scale, bias, configuration,
conversion and store even when their min/max instructions are skipped.

## Shared Inner Kernel Header

The inner kernel header (`tqt_kernel_<op>_<A>_<B>_<tile>_<arch>.h`) is the
performance-critical shared code. It provides template functions:

| Function | Purpose |
|----------|---------|
| `tqt_init_zero_kernel_*<MR,NR>` | Zero all accumulators |
| `tqt_init_c_kernel_*<MR,NR>` | Load C matrix into accumulators |
| `tqt_load_a_kernel_*<MR>` | Load one column of A into scalar/fp registers |
| `tqt_load_b_kernel_*<NR>` | Load one row of B into vector registers |
| `tqt_fused_load_vfmacc_axb_kernel_*<MR,NR>` | Fused: load next A+B while computing current fmacc |
| `tqt_epilogue_vfmacc_kernel_*<MR,NR>` | Final fmacc without next-iteration load |
| `tqt_add_1xnbias_kernel_*<MR,NR>` | Add row-broadcast bias |
| `tqt_add_mx1bias_kernel_*<MR,NR>` | Add column-broadcast bias |
| `tqt_clamp_kernel_*<MR,NR>` | Apply min/max when called; floating GEMM policy gates the call |
| `tqt_store_kernel_*<MR,NR>` | Store accumulators to output D |

**Critical rule**: These headers are shared across all kernel-bases in a family. Never
modify them for a single variant. If you need different behavior, add a new function
with a distinct name (e.g., `tqt_requantize_1xn_kernel_*` or
`tqt_requantize_mx1_kernel_*` for requantization).

### Register Allocation

The inner kernel header documents register usage in its leading comment. For an 8x3vl
FP16 RVV kernel:

```
ft0-ft7:  A[0]-A[7]  (8 scalar FP16 values, one per M-row)
v0-v2:    B[0]-B[2]  (3 vector registers, one per N-group)
v8-v31:   D[0-7][0-2] (24 accumulator vectors = 8 rows x 3 columns)
```

The MR x NR template parameters control which subset of registers is active:
- `gemm_kernel<3, 2>` uses ft0-ft2, v0-v1, v8-v13 (6 accumulators)
- `gemm_kernel<8, 3>` uses all registers (24 accumulators)

## Family Directory Convention

Families group kernels by their input type pair (A, B), because all variants sharing
the same (A, B) reuse the same inner kernel for the hot loop:

```
src/gemm/gemm_f16_f16/
├── tqt_kernel_gemm_f16_f16_8x3vl_rvv.h          # Shared inner kernel
├── tqt_kernel_gemm_f16_f16_8x1vl_f32acc_rvv.h   # FP32 accumulation
├── tqt_kernel_gemm_f16_f16_4x4vl_rvv.h          # Different tile, same family
├── tqt_kernel_gemm_f16_f16_2x8vl_rvv.h          # Different tile, same family
├── gemm_1xnbias_clamp_f16_f16_f16/              # D=f16, 1xN bias
├── gemm_1xnbias_clamp_f16_f16_f16t/             # D=f16, B transposed
├── gemm_1xnbias_clamp_f16_f16_f16p/             # D=f16, B packed
├── gemm_1xnbias_clamp_f16_f16p_f16p/            # D=f16, both packed
├── gemm_mx1bias_clamp_f16_f16_f16/              # D=f16, Mx1 bias
└── gemm_mx1bias_clamp_f16_f16p_f16p/            # D=f16, Mx1 bias, packed
```

Multiple output types can coexist in one family. For example, `gemm_i8_i8/` contains
kernel-bases with D in {i32, qai8, f32, f16} -- the inner kernel always accumulates
in i32, and the epilogue narrows/requantizes/dequantizes to the target D type.

Activation (`src/activation/`) degenerates. Most of its kernels have a single input, so
there is no (A, B) pair to group by, and the family label simply equals the `KERNEL_BASE`
(`src/activation/relu_f32_f32/relu_f32_f32/`). The gated `reglu`, `geglu`, and
`swiglu` kernels take two operands and carry a `dtypeB` slot, but each still gets one family
per kernel-base (`src/activation/geglu_f32_f32_f32/geglu_f32_f32_f32/`) rather than a shared
input-pair parent -- grouping by (A, B) buys nothing when the group has one member.
With one kernel-base per family there is nothing to share, so those kernels carry no
`tqt_kernel_*` inner header and keep their implementation in the `.cpp`. What activation
kernels do share crosses op-group boundaries rather than kernel-bases inside a family, so it
lives in `src/common/` instead: RVV `silu` and `swiglu` call the polynomial `exp`, while RVV
`gelu` and `geglu` call the rational `tanh`, both in `src/common/tqt_math_rvv.h`. Each helper is
instantiated only at the LMUL consumed by a shipped kernel.

Reduction (`src/reduction/`) carries no `tqt_kernel_*` inner header either, for a
different reason: its families group by op (`sum_f32_f32/`, `mean_f32_f32/`) while the
two kernel-bases inside a family differ by reduction direction, and row and column
reduction are separate algorithms. The row path accumulates column blocks and ends in a
horizontal `th.mfredsum.s.c.mf`; the column path accumulates the M row-vectors
elementwise and never reduces. They share only generic load/store/add primitives, so
there is nothing for a family-level header to hold. The reuse that does exist runs along
the other axis -- `sumrow` with `meanrow`, `sumcol` with `meancol` -- which would put a
shared header above both families; each kernel keeps its own copy in its `.cpp` instead,
matching activation and the RVV reduction variants. Each sum / mean pair carries a
`Keep in sync with <the other file>` note: the row pairs' kernel bodies are identical,
the column pairs differ only by the in-register mean scaling step.

### Why Group by (A, B) Not by D?

The inner kernel (K-loop) is the performance bottleneck. It depends only on how A and
B are loaded and multiplied. The output processing (narrow, conditional clamp, requantize) runs
once per tile and is cheap relative to the K-loop. Grouping by (A, B) maximizes code
reuse for the expensive part.

### Public Type and Execution Type

Family names describe the public operand storage types. RVV W16A32 families
`gemm_f32_f16/` and `gemm_f32_bf16/` preserve A as FP32 and widen FP16/BF16 B
values into FP32 FMA. Their packed A panels contain FP32 elements. Use the
matching size and offset getters to allocate and address packed storage.

## Tiling Loop Structure

The public `tqt_run_*` function implements the outer tiling loop:

```cpp
void tqt_run_gemm_...(size_t m, size_t n, size_t k, ...)
{
    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = TQT_MIN(m - m_idx, m_step);
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);

            // Compute pointers for this tile
            const float16_t *a_ptr = A + m_idx * lda;
            const float16_t *b_ptr = B + n_idx;
            float16_t *d_ptr = D + m_idx * ldd + n_idx;

            // Handle N remainder: full groups + tail
            size_t full_n = (actual_n / vlmax) * vlmax;
            if (full_n > 0) {
                gemm_mrxnr_rvv(actual_m, full_n, k, d_ptr, a_ptr, b_ptr, ...);
            }
            size_t remain_n = actual_n - full_n;
            if (remain_n > 0) {
                // Tail: use vl < vlmax for the last partial vector
                gemm_mrxnr_rvv(actual_m, remain_n, k, d_ptr + full_n, ...);
            }
        }
    }
}
```

The M-loop iterates in chunks of `m_step` (e.g., 8 for 8x3vl). The N-loop iterates
in chunks of `n_step` (e.g., 3*vlmax). Within each tile, the N dimension is further
split into full vector-length groups and a remainder handled by setting `vl < vlmax`.

### MR and NR Semantics

- **MR** (1..MR_MAX): number of M-rows processed in this tile. Determines how many
  A scalars are loaded per K iteration and how many accumulator rows are active.
- **NR** (1..NR_MAX): number of N-vector-groups processed. Determines how many B
  vectors are loaded and how many accumulator columns are active.

For the 8x3vl tile: MR_MAX=8, NR_MAX=3. The dispatch selects
`kernel_table[actual_m-1][(actual_n+vlmax-1)/vlmax-1]`.

## Ztt Square Traversal

The `1nlenxpnlen` F32/F16/BF16 variants use family-level inner headers
`src/gemm/gemm_<A>_<B>/tqt_kernel_gemm_<A>_<B>_1nlenxpnlen_ztt.h`.
Each header fixes its input datatype and shares packing and traversal across that
family's ordinary, transposed and packed variants. F16/BF16 native and F32
accumulation use the same traversal with a compile-time accumulation-mode switch.
The headers carry synchronization notes for their common fixed-register algorithm;
GEMM geometry, packing and ownership policy stay out of `tqt_common.h`. Public
wrappers retain per-base function tables. Layout and UDS specialize at compile
time. It computes S-by-P*S output tiles with P independent accumulators and
P*S K steps, where S is runtime `amenlen` and
P is the input datatype's `pack_factor = max(1, UDS / input_bits)`.
A single matrix-multiply instruction still produces one S-by-S accumulator square;
the P-square output width is this kernel's blocking strategy.

The noinline compute function owns its fixed M/ACC registers for the duration
of the call. It uses no compiler-managed matrix values and makes no calls while
matrix values are live; assembly memory barriers expose loads/stores to the
compiler. Do not inline this boundary into code using matrix intrinsics. Packed
B uses block-column-major squares to make a P-square K segment contiguous.
See the [Ztt API contract](../reference/api_reference.md#ztt-gemm).

Hosted builds cache `amenlen` and `ameudsz` on first use per compilation unit;
freestanding builds read them on demand. Resource ownership remains scoped to
each call, with no ownership cache or context object.

## Pack Format

Packed variants (`_f16p`, `_f32p`, etc.) rearrange input data into a layout optimized
for sequential access by the inner kernel:

### Panel-Major Packing (B matrix)

Standard B layout (row-major, k x n):
```
B[0,0] B[0,1] B[0,2] ... B[0,n-1]
B[1,0] B[1,1] B[1,2] ... B[1,n-1]
...
```

Packed B layout (panel-major, n_step-wide panels):
```
Panel 0: B[0,0..n_step-1], B[1,0..n_step-1], ..., B[k-1,0..n_step-1]
Panel 1: B[0,n_step..2*n_step-1], ..., B[k-1,n_step..2*n_step-1]
...
```

Within each panel, elements are stored contiguously along N (vector-length aligned),
then along K. This matches the inner kernel's access pattern: for each K iteration,
it loads a contiguous vector of N elements.

### Panel-Major Packing (A matrix)

Packed A layout groups m_step rows together:
```
Panel 0: A[0,0..k-1], A[1,0..k-1], ..., A[m_step-1,0..k-1]
Panel 1: A[m_step,0..k-1], ..., A[2*m_step-1,0..k-1]
...
```

Pack helpers are provided as `tqt_run_a_pack_*`, `tqt_run_b_pack_*`, and
`tqt_run_bt_pack_*` functions, which transform standard row-major or transposed
inputs into the packed format.

## Offset Getters

The `get_*_offset` functions enable callers to compute byte offsets into buffers
without knowing the internal data layout. This is essential for:

1. **K-tiling**: splitting the K dimension across multiple kernel invocations.
   The caller uses `get_a_offset(m_idx, k_start, lda)` to find where each K-slice
   begins.

2. **M/N-tiling**: partitioning work across threads. Each thread computes its own
   pointer offset using `get_d_offset(m_start, n_start, ldd)`.

3. **Packed buffers**: packed layouts have non-trivial offset calculations. The getter
   hides this complexity behind a uniform interface.

For standard (non-packed) layouts, the offset is simply:
```cpp
size_t tqt_get_a_offset_...(size_t m_idx, size_t k_idx, size_t lda)
{
    return (m_idx * lda + k_idx) * sizeof(element_type);
}
```

For packed layouts, the offset computation accounts for panel boundaries and padding.

The pointer passed to a packed `run` function is the origin selected by the caller's
offset getter. During split-K, the outer tiling loop must derive all internal panel
addresses relative to that origin rather than applying the absolute `k_idx` a second
time. This invariant also keeps tail panels correct when the first panel has fewer
rows or columns than the compile-time step.

## Stateless Design

Each micro-kernel operates on caller-provided buffers and requires no context
object or explicit initialization call. It retains no tensor data between calls.

This means:
- Multiple threads can call the same kernel concurrently on different tiles
- Hosted geometry caches require thread-safe first-use initialization
- Freestanding geometry queries do not require local-static initialization guards
- Testing is trivial: set up inputs, call function, check outputs

The only library state is the hosted cache of hardware-derived geometry: RVV
`vlenb` and Ztt `amenlen`/`ameudsz`. Each is read on first use and
cached per compilation unit; freestanding builds read on demand. Neither mode
reads optional CSRs at library load time. Calls sharing a cache must use the same
geometry, including across threads. See the
[runtime integration contract](../guide/integration.md#runtime-isa-and-width-queries).

## K-Loop Optimization: Fused Load + FMA

The steady-state K-loop fuses the load of the next iteration's A and B data with the
FMA computation of the current iteration:

```
Iteration k:
  1. Compute fmacc(A[k], B[k]) using pre-loaded registers
  2. Load A[k+1] into scalar registers (overlaps with FMA latency)
  3. Load B[k+1] into vector registers (overlaps with FMA latency)
```

This software-pipelining technique hides load latency behind compute. The prologue
loads the first iteration's data, and the epilogue performs the final FMA without
a subsequent load.

## Quantized Kernel Extensions

Quantized kernels extend the basic structure with additional epilogue steps:

### Requantize Path (INT8 output)

```
K-loop (i32 accumulator) → add bias → requantize(scale, zp) → clamp → store i8
```

The requantization applies: `output = clamp(round(acc * scale) + zero_point, -128, 127)`

### Dequantize Path (FP output)

```
K-loop (i32 accumulator) → dequant(scale_a, scale_b) → add bias → conditional clamp → store fp
```

The dequantization applies: `output = clamp(acc * scale_a * scale_b + bias, min, max)`.
This schematic does not move type conversion: for example, RVV i8-to-FP16 first
narrows the dequantized accumulator, then adds FP16 bias, clamps and stores. The
private policy only skips that path's original min/max for exactly unbounded public
arguments; finite and one-sided bounds keep their existing precision and order.

Both paths reuse the same inner kernel K-loop; only the post-accumulation epilogue
differs.

### RVV Quantized-Activation Block GEMM Boundary

These integer-dot variants accept adapter-produced execution tensors and already
quantized activations. Q2_K/Q3_K/Q4_K/Q5_K/Q6_K use the Q8_K-compatible
`qsi8d256` activation contract; Q8_0 uses the Q8_0-compatible `qsi8d32`
contract. Code packers transform only A/B code tensors. FP32 scales, factorized
group/super-block metadata, and required activation sums remain caller-owned
arrays selected through the params structure and the global K index.

Each weight group first accumulates an integer dot product. The kernel then
applies group or super-block FP32 factors before adding C, 1xN bias, and clamp.
Q2_K applies its factorized min correction with exact 16-value A sums. Q4_K and
Q5_K subtract `real_zero_b * sum(A_codes)`. The plane-packed Q6_K path uses the
same sums for its offset-binary `-32` correction; the byte-expanded signed-I8
Q6_K path does not need them. Container decoding and activation quantization
remain outside TORQ-Tile.

#### RVV Execution

RVV provides compact Q2_K/Q3_K, nibble-expanded Q2_K/Q3_K, normalized Q4_K,
byte-expanded Q5_K/Q6_K, plane-packed Q5_K/Q6_K, and Q8_0 variants. Its packers
produce 4x1vl A/B panels with N-interleaved B code bytes. The plane-packed
`gemm_i8_u5` and `gemm_i8_i6` families decode group-local low/high bit planes in
registers and process two A rows per decode.

#### Q2_K/Q3_K Factorized Metadata

The Q2_K/Q3_K kernels consume adapter-owned execution tensors rather than raw
GGUF blocks. `qau2c16p` and `qsi3c16p` preserve compact 2-bit/3-bit code planes
for RVV register-time decode. `qau4c16p` and `qsi4c16p` expand only the codes to
nibbles so RVV can share the metadata contract.

Metadata remains factorized across the original 256-value super-block. Q2_K
carries unsigned `qscale_b`/`qmin_b` per 16 values plus FP32
`super_scale_b`/`super_min_b` per 256 values; Q3_K carries signed `qscale_b` per
16 values plus FP32 `super_scale_b` per 256 values. Kernels complete the
group-scaled integer accumulation, including the Q2_K min correction, before
applying the super-block FP32 factors. K and Split-K offsets must therefore be
256-aligned.

No public API parses `block_q2_K` or `block_q3_K`. Importers own canonical
scale/min decoding and may write the final code and metadata tensors directly.

## Adding a New Architecture Target

To add support for a new RISC-V extension (beyond RVV):

1. Define new architecture guard macros (e.g., `__riscv_xnewext`).
2. Create new inner kernel headers with the appropriate tile dimensions.
3. Add a new CMake source variable (e.g., `TORQ_TILE_NEWEXT_SOURCES`).
4. Implement the inner kernel template functions using the new extension's intrinsics
   or inline assembly.
5. Follow the same dispatch table pattern -- only the register allocation and
   instruction selection differ.

The public API surface (getters + run function) remains identical regardless of the
backend architecture, enabling transparent dispatch at a higher level.
