<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# TORQ-Tile naming convention

Every micro-kernel in TORQ-Tile is identified by a structured name. The same name appears
in the source filename and public function names. Tests and benchmarks group variants
by operation, type and architecture. Examples use calling-scenario names rather than
kernel names; see the [example index](../../examples/README.md). This document defines
the kernel-name grammar.

Read this page first when you encounter a name like:

```
tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv
```

For an exhaustive list of every variant currently shipped, see the [micro-kernel tables](microkernel_tables.md).

On this page:

- [Anatomy of a kernel name](#anatomy-of-a-kernel-name)
- [Field reference](#field-reference)
- [Function naming](#function-naming)
- [File and directory layout](#file-and-directory-layout)
- [Worked examples](#worked-examples)
- [Cross references](#cross-references)

## Anatomy of a kernel name

```
File:     tqt_<KERNEL_BASE>_<tile>_<arch>.cpp / .h
Function: tqt_<action>_<KERNEL_BASE>_<tile>_<arch>(...)

KERNEL_BASE = <op>_[<biastype>_][<fused_ops>_]<dtypeD>[suffix]_<dtypeA>[suffix]_<dtypeB>[suffix]
KERNEL_FULL = <KERNEL_BASE>_<tile>_<arch>
```

Concretely, the example above decomposes as:

| Field        | Value           | Meaning                                             |
|--------------|-----------------|-----------------------------------------------------|
| prefix       | `tqt_`          | Library prefix; appears on every public symbol.     |
| op           | `gemm`          | Primary operator: matrix multiplication.            |
| biastype     | `1xnbias`       | Per-N bias broadcast across rows.                   |
| fused_ops    | `reqi32toi8_clamp` | Fused i32→i8 requantization, then clamp.         |
| dtypeD       | `qai8`          | Destination: quantized asymmetric int8.             |
| dtypeA       | `qai8`          | LHS: quantized asymmetric int8.                     |
| dtypeB       | `qsi8cxt`       | RHS: quantized symmetric int8, channel-wise, transposed. |
| tile         | `4x1vl`         | Output tile: 4 rows × 1×vlmax columns.              |
| arch         | `rvv`           | RISC-V Vector Extension implementation.             |

## Field reference

### `op` — primary operator

| Token        | Operator |
|--------------|----------|
| `gemm`       | General matrix multiplication, `D = A · B + bias`, with optional clamp (omitted for raw i32 output). |
| `sgemm`      | BLAS-style single-precision GEMM: `D = alpha · A · B + beta · C`. No bias/clamp; alpha and beta are scalar arguments to the run function. |
| `dgemm`      | BLAS-style double-precision GEMM: same semantics as `sgemm` but with `double`. |
| `convcf`     | Convolution, **channel-first** (CHW) layout, im2col + GEMM. |
| `convcl`     | Convolution, **channel-last** (HWC) layout, im2col + GEMM. |
| `igemmcf`    | Implicit GEMM convolution, channel-first layout. |
| `igemmcl`    | Implicit GEMM convolution, channel-last layout. |
| `dwconvcf`   | Depthwise convolution, channel-first, generic kernel size. |
| `dwconvcl`   | Depthwise convolution, channel-last, generic kernel size. |
| `dwconvcf3x3s1` / `dwconvcf3x3s2` / `dwconvcf5x5s1` | Depthwise convolution, channel-first, kernel-size + stride specialised. |
| `dwconvcl3x3s1` / `dwconvcl3x3s2` / `dwconvcl5x5s1` | Depthwise convolution, channel-last, kernel-size + stride specialised. |
| `addvv` / `subvv` / `mulvv` / `divvv` | Elementwise vector-vector add / subtract / multiply / divide: `c[i] = a[i] ∘ b[i]`, `∘ ∈ {+, -, *, /}`. |
| `addvs` / `subvs` / `mulvs` / `divvs` | Elementwise vector-scalar add / subtract / multiply / divide: `c[i] = a[i] ∘ b` (`b` is a scalar). |
| `sumrow`     | Row-wise reduction sum: `c[m] = Σ_n a[m,n]`. |
| `sumcol`     | Column-wise reduction sum: `c[n] = Σ_m a[m,n]`. |
| `meanrow`    | Row-wise reduction mean: `c[m] = (1/N)·Σ_n a[m,n]`. |
| `meancol`    | Column-wise reduction mean: `c[n] = (1/M)·Σ_m a[m,n]`. |
| `relu`       | Rectified linear unit: `c[i,j] = max(a[i,j], 0)`. |
| `gelu`       | Gaussian error linear unit using the standard tanh approximation: `0.5·x·(1 + tanh(sqrt(2/π)·(x + 0.044715·x³)))`. |
| `silu`       | SiLU / swish: `c[i,j] = a[i,j] · sigmoid(a[i,j])`, `sigmoid(x) = 1 / (1 + exp(-x))`. |
| `reglu`      | ReGLU — gated ReLU: `c[i,j] = relu(a[i,j]) · b[i,j]`. |
| `geglu`      | GeGLU — gated standard tanh-approximate GeLU: `c[i,j] = gelu(a[i,j]) · b[i,j]`. |
| `swiglu`     | SwiGLU — gated SiLU: `c[i,j] = silu(a[i,j]) · b[i,j]`. |

For the depthwise specialised variants, the digit pair is the spatial filter size (`3x3`, `5x5`) and `sN` is the stride (`s1`, `s2`).

The elementwise (`{add,sub,mul,div}{vv,vs}`), reduction (`sumrow`, `sumcol`, `meanrow`, `meancol`) and activation (`relu`, `gelu`, `silu`, `reglu`, `geglu`, `swiglu`) operators carry **no biastype, no fused_ops, and no clamp** segment. Their `KERNEL_BASE` collapses to `<op>_<dtypeD>_<dtypeA>_<dtypeB>` for elementwise and gated activation, and `<op>_<dtypeD>_<dtypeA>` for reduction and single-operand activation.

**Gated activation.** `reglu`, `geglu`, and `swiglu` take the full
`<op>_<dtypeD>_<dtypeA>_<dtypeB>` triple — for example,
`reglu_f32_f32_f32`, `geglu_f16_f16_f16`, and `swiglu_f32_f32_f32`.
`A` is the gated operand and `B` the linear one.

Among the elementwise, reduction, and activation op groups, activation alone carries **independent leading dimensions per operand** — `run(m, n, a, lda, c, ldc)` for the single-operand kernels and `run(m, n, a, lda, b, ldb, c, ldc)` for the gated activations, over a row-major `[M, N]` matrix, with `c` allowed to alias any input. Elementwise kernels are 1-D (`n` only, no leading dimension) and reduction kernels take `lda` but write a contiguous 1-D output.

The third leading dimension makes separate split-half gated variants unnecessary: the fused `[M, 2N]` gate-and-up buffer that transformer feed-forward blocks produce is addressed by passing `a = base, lda = 2N` and `b = base + N, ldb = 2N`, with no repacking and no extra kernel.

`igemm` differs from `conv` (= im2col + GEMM) in that the implicit-GEMM formulation never materialises an im2col buffer; it walks the input via an indirection table, see [`src/conv/igemm_*`](../../src/conv/).

### `biastype` — bias broadcasting mode

| Token               | Bias semantics |
|---------------------|----------------|
| `1xnbias`           | Bias is a row vector of length N, broadcast across the M dimension. |
| `1xnbiasf32`        | Per-N FP32 bias broadcast across rows. |
| `mx1bias`           | Bias is a column vector of length M, broadcast across the N dimension. |
| `mx1bias_1xnbias`   | Both per-M and per-N bias simultaneously (dual bias). Used by UINT8 kernels. |
| *(omitted)*         | No bias term. Used by plain INT8 kernels (e.g. `gemm_clamp_i32_i8_i8`), BLAS-style `sgemm`/`dgemm` (which have no bias by design), and all elementwise / reduction kernels (`addvv`, `addvs`, `sumrow`, `sumcol`, `meanrow`, `meancol`). |

### `fused_ops` — fused epilogue operations

Read left to right; each token is a single fused step applied to the accumulator before the result is written to memory.

| Token         | Operation |
|---------------|-----------|
| `clamp`       | Clamp the output to `[clamp_min, clamp_max]`. Present for floating-point and quantized integer output types. Omitted when the output is an unclamped integer type (e.g. i32), or in BLAS-style `sgemm`/`dgemm` (which have no bias/clamp by design). |
| `reqi32toi8`  | Requantize an i32 accumulator to i8 using per-channel scale and zero point, then clamp. Used exclusively by the int8 quantized kernels. |

The two tokens always appear in the order `reqi32toi8_clamp` when both are present.

### `dtypeD`, `dtypeA`, `dtypeB` — data types

The output (`D`), LHS (`A`), and RHS (`B`) data types appear in that order. Each is built from one or more of the following tokens:

#### Float / integer base types

| Token  | Type |
|--------|------|
| `f64`  | IEEE 754 double-precision float. Used by BLAS-style `dgemm`. |
| `f32`  | IEEE 754 single-precision float. |
| `f16`  | IEEE 754 half-precision float (FP16). |
| `bf16` | Brain floating point (BF16). |
| `i32`  | Signed 32-bit integer (used for unquantized i32 accumulator output). |
| `u8`   | Unsigned 8-bit integer. Used raw by UINT8 GEMM and as normalized quantized data in `qau8c32`. |
| `i8`   | Signed 8-bit integer. Used raw or with quantization metadata such as `qsi8c16` and `qsi8c32`. |
| `b1`   | Packed binary 1-bit code. In `qsb1c128`, bit 0 represents -1 and bit 1 represents +1. |
| `u5`   | Unsigned 5-bit integer stored as group-local low4/high1 bit planes. Used by plane-packed `qau5c32`. |
| `i5`   | Signed 5-bit integer stored as group-local low4/sign bit planes. Used by `qsi5c32`. |
| `i3`   | Signed 3-bit integer stored as group-local low2/high1 bit planes. Used by compact `qsi3c16`. |
| `i6`   | Signed 6-bit integer stored as group-local low4/high2 offset-binary bit planes. Used by plane-packed `qsi6c16`. |
| `i4`   | Signed 4-bit integer (no quantization metadata). |
| `u4`   | Unsigned 4-bit integer stored as two nibbles per byte. Used by normalized asymmetric U4 execution inputs such as `qau4c32`. |
| `u2`   | Unsigned 2-bit integer stored as four fields per byte. Used by compact `qau2c16`. |
| `i2`   | Signed 2-bit integer (symmetric INT2 weight, per-channel quantized). Stored as `[n, k/4]` uint8 with four 2-bit slots per byte. |

#### Quantization prefixes and qualifiers

The integer types may carry quantization metadata. The full token is built up from these pieces:

| Token  | Meaning |
|--------|---------|
| `qa…`  | Quantized **asymmetric** (has zero point). |
| `qs…`  | Quantized **symmetric** (zero point fixed at 0). |
| `…cx`  | Per-**channel** scale — one scale per column of B along the N dimension (N scales total). |
| `…dx`  | Per-**dimension** scale — one scale per row of A along the M dimension (M scales total). |
| `…c2`  | Per-channel + K-blocked — B side, one scale per (column, K-block) pair; block length passed at run-time. |
| `…c16` | Per-channel + fixed 16-element K blocks — B side, one scale per (column, K-block). |
| `…c32` | Per-channel + fixed 32-element K blocks — B side, one scale and, for asymmetric formats, one real-domain zero offset per (column, K-block). |
| `…c64` | Per-channel + fixed 64-element K blocks — B side, one fp32 scale per (column, K/64) block. |
| `…c128` | Per-channel + fixed 128-element K blocks — B side, one fp32 scale per (column, K/128) block. |
| `…d2`  | Per-dimension + K-blocked — A side, one scale per (row, K-block) pair (symmetric, no zero point). |
| `…d32` | Per-dimension + fixed 32-element K blocks — A side, one fp32 scale per (row, K/32) block. |
| `…d256` | Per-dimension + fixed 256-element K blocks — A side, one fp32 scale per (row, K/256) block. Selected B execution formats also consume exact int16 activation sums per 16 K values. |

Combining prefix + base type + qualifier gives the full data-type token, e.g.:

| Token       | Decomposition                                | Description |
|-------------|----------------------------------------------|-------------|
| `qai8`      | quant-asym + i8                              | Quantized asymmetric int8 (per-tensor scale + zero point). |
| `qai8dx`    | quant-asym + i8 + per-dimension              | Quantized asymmetric int8, per-row scale. |
| `qsi8cx`    | quant-sym + i8 + per-channel                 | Quantized symmetric int8, per-channel scale. |
| `qsi8dx`    | quant-sym + i8 + per-dimension               | Quantized symmetric int8, per-row scale. |
| `qsi8d2`    | quant-sym + i8 + per-dimension + K-blocked   | Quantized symmetric int8, per-row K-blocked scale. |
| `qsi8d32`   | quant-sym + i8 + fixed 32-element K-block    | Q8_0-compatible symmetric activation: int8 codes plus one fp32 scale per (M, K/32) block. |
| `qsi8d256`  | quant-sym + i8 + fixed 256-element K-block   | Q8_K-compatible symmetric activation: int8 codes plus one fp32 scale per (M, K/256) block; `qau2c16`, `qau4c16`, `qau4c32`, `qau8c32`, `qau5c32`, and `qsi6c16` paths additionally consume exact 16-value code sums. |
| `qsb1c128`  | quant-sym + binary b1 + fixed 128-element K-block | Q1_0 execution weight: bit `0 -> -1`, bit `1 -> +1`, with one fp32 scale per (N, K/128) block. |
| `qau2z1c64` | quant-asym + u2 + fixed zero point 1 + fixed 64-element K-block | Q2_0 execution weight: logical value `code - 1`, with one fp32 scale per (N, K/64) block. |
| `qsi4cx`    | quant-sym + i4 + per-channel                 | Quantized symmetric int4, per-channel scale (W4A8 per-channel path). |
| `qsi4c2`    | quant-sym + i4 + K-blocked                   | Quantized symmetric int4 with K-blocked group quantization (ZP=8 encoding). |
| `qai4c2`    | quant-asym + i4 + K-blocked                  | Quantized asymmetric int4 with K-blocked group quantization (unsigned codes with independent per-block fp32 scale and additive offset). |
| `qau4c32`   | quant-asym + u4 + fixed 32-element K-block   | Normalized asymmetric U4 weight with separate scale and real-zero metadata per (N, K/32) block; it can be produced from Q4_K by an external adapter. |
| `qau2c16`   | quant-asym + u2 + fixed 16-element K-block   | Compact Q2_K-derived U2 codes with integer group scale/min metadata and fp32 scale/min per 256-value super-block. |
| `qau4c16`   | quant-asym + u4 + fixed 16-element K-block   | Nibble-expanded Q2_K-derived U4 codes with the same factorized metadata as `qau2c16`. |
| `qsi3c16`   | quant-sym + i3 + fixed 16-element K-block    | Compact Q3_K-derived signed I3 codes with signed integer group scales and fp32 scale per 256-value super-block. |
| `qsi4c16`   | quant-sym + i4 + fixed 16-element K-block    | Nibble-expanded Q3_K-derived signed I4 codes with the same factorized metadata as `qsi3c16`. |
| `qsi4c32`   | quant-sym + i4 + fixed 32-element K-block    | Signed two's-complement I4 execution weight with one fp32 scale per (N, K/32) block; used by Q1_0/Q2_0 expanded and Q4_0 normalized paths. |
| `qsi5c32`   | quant-sym + i5 + fixed 32-element K-block    | Signed two's-complement I5 execution weight stored as low4 plus sign plane, with one fp32 scale per (N, K/32) block. |
| `qau5c32`   | quant-asym + u5 + fixed 32-element K-block   | Plane-packed asymmetric U5 weight with separate scale and real-zero metadata per (N, K/32) block. |
| `qau8c32`   | quant-asym + u8 + fixed 32-element K-block   | Normalized asymmetric U8 weight with separate scale and real-zero metadata per (N, K/32) block; used by the Q5_K execution path. |
| `qsi6c16`   | quant-sym + i6 + fixed 16-element K-block    | Plane-packed symmetric I6 weight with one scale per (N, K/16) block; codes use unsigned offset-binary storage and register-time `-32` correction. |
| `qsi8c16`   | quant-sym + i8 + fixed 16-element K-block    | Normalized symmetric I8 weight with one scale per (N, K/16) block; used by the Q6_K execution path. |
| `qsi8c32`   | quant-sym + i8 + fixed 32-element K-block    | Normalized symmetric I8 weight with one scale per (N, K/32) block; used by the Q8_0 execution path. |
| `qsi2cx`    | quant-sym + i2 + per-channel                 | Quantized symmetric int2, per-channel scale. |

### Suffixes — layout decorations on data-type slots

A single character may be appended **directly** to a data-type token to describe a layout transformation. Suffixes apply independently to `A` and `B`. The `p` suffix also applies to `D`
for packed-output depthwise kernels and Ztt packed-output GEMM (`f32p_f32p_f32p`, `f16p_f16p_f16p`, and `bf16p_bf16p_bf16p`);
its output requires the matching unpack function for ordinary row-major consumers.

| Suffix | Meaning |
|--------|---------|
| `t`    | **Transposed** — the matrix is laid out in `[N, K]` order rather than the default. For `B`, this means K is the inner dimension when viewed natively. |
| `p`    | **Packed** — the matrix is pre-shuffled into the kernel's preferred internal layout. The kernel does not pack on the fly; the caller must invoke the matching packing routine first. Packed-input kernels are typically the highest-performance variants. |

A single token may not carry both `t` and `p`; pick the path the kernel expects. Examples:

- `f16t` — FP16 matrix in transposed layout.
- `f16p` — FP16 matrix that has been pre-packed.
- `qsi8cxp` — quantized symmetric int8 with per-channel scale, pre-packed.
- `qsi8cxt` — quantized symmetric int8 with per-channel scale, transposed but not packed.
- `qau4c32p` — normalized asymmetric U4 block-32 weights in an architecture-private packed layout.
- `qau2c16p` / `qsi3c16p` — compact Q2_K/Q3_K-derived group-16 code streams in an RVV-private N-interleaved layout.
- `qau4c16p` / `qsi4c16p` — nibble-expanded Q2_K/Q3_K-derived group-16 code streams in RVV-private packed layouts.
- `qsb1c128p` / `qau2z1c64p` — compact Q1_0/Q2_0 execution code streams in RVV-private N-interleaved layouts.
- `qsi4c32p` — signed I4 block-32 execution codes in RVV-private packed layouts.
- `qsi5c32p` — signed I5 low4/sign-plane execution codes in an RVV-private packed layout.
- `qau5c32p` — plane-packed asymmetric U5 block-32 weights in an architecture-private packed layout.
- `qau8c32p` — normalized asymmetric U8 block-32 weights in an architecture-private packed layout.
- `qsi6c16p` — plane-packed symmetric I6 block-16 weights in an architecture-private packed layout.
- `qsi8c16p` / `qsi8c32p` — normalized symmetric I8 block-16/block-32 weights in
  architecture-private packed layouts.
- `qsi8d256p` / `qsi8d32p` — packed RVV Q8_K/Q8_0-compatible activation codes;
  scales and activation sums remain in separate unpacked tensors.

### `tile` — output tile size

The tile encodes how many output elements one invocation of the inner kernel template produces. The format is `<MR>x<NR>[…]<unit>`, where `<unit>` is either a literal element count or an architecture-specific scaling unit.

| Tile           | M direction | N direction | Notes |
|----------------|-------------|-------------|-------|
| `8x3vl`        | 8 rows      | 3 × vlmax   | Most common RVV GEMM/conv tile. |
| `4x4vl`        | 4 rows      | 4 × vlmax   | Alternative RVV tile for FP32/FP16 GEMM (wider N, shorter M). |
| `2x8vl`        | 2 rows      | 8 × vlmax   | Alternative RVV tile for FP32/FP16 GEMM (widest N, shortest M). |
| `8x2vl`        | 8 rows      | 2 × vlmax   | Used by INT4 W4A16 RVV kernels. |
| `8x1vl`        | 8 rows      | 1 × vlmax   | Used by BF16 RVV kernels. |
| `4x1vl`        | 4 rows      | 1 × vlmax   | Used by INT8/INT4/INT2 RVV GEMM/conv kernels. |
| `1x1vl`        | — (dwconv only) | 1 × vlmax   | Used by generic depthwise conv kernels. `get_m_step` is not exposed; spatial dimensions are handled via `tqt_conv_params`. |
| `2x1x1vl` / `4x1x1vl` | spatial × ic-rows | 1 × vlmax | Used by depthwise specialised (`3x3s2` → `2x1x1vl`, `3x3s1`/`5x5s1` → `4x1x1vl`). |
| `3vlx8`        | 3 × vlmax   | 8 cols      | Used by `igemmcf` kernels (channels-along-rows layout). |
| `1vlx4`        | 1 × vlmax   | 4 cols      | Used by `igemmcf` INT8 kernels. |
| `8vl`          | — (elementwise) | 8 × vlmax   | Used by RVV elementwise `{add,sub,mul,div}{vv,vs}` kernels; only `get_n_step` is exposed. |
| `4x4vl` / `4x2vl` | 4 rows | 4 × vlmax (f32) / f16m2 (f16) | Used by `sumrow` / `meanrow` (reduction, RVV); M (rows) is the non-reduced output dim, N is the reduced dim. F32 uses `4x4vl`, F16 uses `4x2vl`. |
| `2x8vl` / `2x4vl` | 2 rows | 8 × vlmax (f32) / f16m2 (f16) | Used by `sumcol` / `meancol` (reduction, RVV); N is the non-reduced output dim, M (unrolled by 2 with dual partial accumulators) is the reduced dim. F32 uses `2x8vl`, F16 uses `2x4vl`. |
| `1x8vl`        | 1 row       | 8 × vlmax   | Used by RVV activation (`relu`, `reglu`); one inner invocation walks a single row, so `m_step = 1`. These paths keep only a few vector groups live, so the widest LMUL is used. |
| `1x4vl` / `1x2vl` | 1 row | 4 × vlmax (f32) / f16m2 (f16) | Used by RVV activation (`gelu`, `geglu`, `silu`, `swiglu`); one inner invocation walks a single row, so `m_step = 1`. The shared polynomial/rational math from [`src/common/tqt_math_rvv.h`](../../src/common/tqt_math_rvv.h) keeps several vector groups live, so these paths cannot use the widest LMUL. F32 loads `f32m4`; F16 loads `f16m2` and widens it into the same `f32m4` group, so both give `n_step = 4 × vlmax(f32)`. |
| `1nlenxpnlen` | 1 × nlen | P × nlen | Ztt F32/F16/BF16 GEMM. `P = pack_factor = max(1, UDS / input_bits)`, specialized at compile time; `nlen = amenlen` at runtime. Nonpacked K also steps by P × nlen. F16/BF16 have native and `f32acc` strategies. |
| `1nlenx1nlen` | 1 × nlen | 1 × nlen | Ztt Q4_K/Q6_K integer-dot GEMM with ordinary F32 output. Both steps equal `amenlen`; code buffers use variant-specific square packing. |

`vlmax` is the runtime vector-length-max of the executing core (in elements of the operand type).

The full mapping from tile to `m_step` / `n_step` / MR / NR template ranges is documented in the kernel headers. See for example [`src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16t/tqt_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv.h`](../../src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16t/tqt_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv.h).

### Ztt pack factor

In `1nlenxpnlen`, `nlen` is the value read from `amenlen`: the side length,
in elements, of one logical square. Lowercase `p` denotes the input datatype's
`pack_factor`, written as P in formulas. The name means an output tile of
`nlen` rows by `P*nlen` columns; `pnlen` is not a separate CSR.

Ztt 0.6 defines `pack_factor` in its **Squares** section and uses `P = pack_factor`
in its memory-instruction notation. With E equal to the input element width in
bits and UDS equal to `AME_UNIT_DATATYPE_SIZE` (also in bits):

```text
P = UDS / E    when E < UDS (packed datatype)
P = 1          when E >= UDS (unit or wide datatype)
```

Supported widths are powers of two, so this is `P = max(1, UDS / E)`.
For a packed datatype, one M register contains P logical squares. For a wide
datatype, P remains 1 even though a square spans multiple M registers; P is not
the register-group size. P is fixed by the compiler profile and input type,
while nlen is read at runtime. F16/BF16 `f32acc` still uses E=16, not E=32.

Every Acc register holds one square. This kernel chooses P independent
accumulators along the output N dimension and a K step of P*nlen. A single
matrix-multiply instruction still produces one nlen-by-nlen square; the wider
output tile is a kernel strategy, not an ISA requirement.

The tile-field `p` denotes register packing. The datatype suffix `p`, as in
`f32p`, instead denotes a packed memory layout managed by the library's packers.
The meanings belong to different name fields. See [Ztt square packing](../guide/packing.md#ztt-square-packing)
for the memory format.

### `arch` — instruction set extension

| Token | Meaning |
|-------|---------|
| `rvv` | RISC-V Vector Extension. The kernel uses the standard RVV intrinsic set. |
| `ztt` | AME (Attached Matrix Extension). |

Architecture-specific kernel `.cpp` sources use a hard architecture guard. Standalone
examples use a guard that produces an empty translation unit on non-matching hosts, while
test and benchmark sources may rely on matching CMake ISA gates.

### optional strategy suffix

When two implementations share the same `KERNEL_BASE`, `tile`, and `arch` but differ only in
their internal execution strategy, a strategy suffix is inserted between `<tile>` and `<arch>`
to disambiguate them, giving `<KERNEL_BASE>_<tile>_<strategy>_<arch>`.

| Token    | Meaning |
|----------|---------|
| `f32acc` | The reduction accumulates in fp32 (widening multiply-accumulate, or widened operands with FP32 FMA), then the result is narrowed back to the public output type before store. Inputs and output are unchanged, so the public contract is identical to the default same-type-accumulator variant; only the internal precision differs. Example: `tqt_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv`. |

## Function naming

All public functions inside a kernel header follow the pattern:

```
tqt_<action>_<KERNEL_FULL>(...)
```

Public actions used across kernel interfaces include:

| Action               | Returns                                                        |
|----------------------|----------------------------------------------------------------|
| `get_m_step`         | The M-direction step the kernel processes per invocation.      |
| `get_n_step`         | The N-direction step the kernel processes per invocation.      |
| `get_oh_step`        | The output-height step for specialized depthwise kernels.       |
| `get_*_offset`       | Byte offset into a kernel buffer for a given tile or split.    |
| `get_*_size`         | Total bytes required for an output / packed buffer of the given shape. |
| `run_a_pack` / `run_b_pack` / `run_bt_pack` | Pre-pack A or B into the kernel's preferred layout. |
| `run_qparams_pack` / `run_scale_*_pack` | Pre-pack quantization parameters or scale tensors. |
| `run_a_im2colpack` / `run_b_im2colpack` | Convert im2col data directly into a packed matrix. |
| `run_pack_input` / `run_pack_weight` / `run_unpack_output` | Convert packed depthwise-convolution buffers. |
| `get_a_packed_size` / `get_b_packed_size` | Buffer size required for the packed A or B matrix. |
| `run` / `run_conv` / `run_dwconv` / `run_igemm` | Invoke the compute entry point. |
| `run_im2col` | Convert a convolution input region into an im2col matrix. |
| `run_c_pack` / `run_bias_pack` / `run_d_unpack` | Pack residual/bias and unpack Ztt packed GEMM output. |

For example, the kernel `tqt_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv` exposes `tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv()`, `tqt_get_n_step_gemm_…()`, `tqt_run_gemm_…()`, and the associated offset / size getters.

The exact set of getters is fixed per operator family by the corresponding `_interface.h` header. Tile and arch variants of the same `KERNEL_BASE` share the same interface and are interchangeable from the caller's point of view.

## File and directory layout

A kernel variant is laid out as follows:

```
src/<family>/<dtype-pair>/<KERNEL_BASE>/
    tqt_<KERNEL_BASE>_<tile>_<arch>.h         # public API for one variant
    tqt_<KERNEL_BASE>_<tile>_<arch>.cpp       # implementation
    tqt_<KERNEL_BASE>_interface.h             # shared by all (tile, arch) variants
```

Where:

- `<family>` is one of `gemm/`, `conv/`, `dwconv/`, `elementwise/`, `reduction/`, `activation/`. (For `conv/`, the subdirectories are split further into `conv_<dtypes>/` and `igemm_<dtypes>/`.)
- `<dtype-pair>` is e.g. `gemm_f16_f16`, `gemm_bf16_bf16`, `gemm_f16_i4`, `gemm_i8_u5`, `gemm_i8_i6`, `gemm_i8_i2`, `gemm_u8_u8`, `conv_i8_i8`, `dwconv_f32_f32`, `add_f32_f32`, `sum_f32_f32`, `relu_f32_f32`. It captures the unquantized base types of the input tensors (A and B, without destination type). For activation the family label equals the `KERNEL_BASE` — there is only one kernel-base per family, whether it has one input (e.g. `src/activation/relu_f32_f32/relu_f32_f32/`) or two (e.g. `src/activation/swiglu_f32_f32_f32/swiglu_f32_f32_f32/`).
- `<KERNEL_BASE>` is the full base name as defined above (without `tqt_`, `<tile>`, or `<arch>`).

The matching test and benchmark artefacts follow operation/dtype naming; examples use scenarios:

```
test/<family>/test_<family>_<dtypes>_<arch>.cpp        # one test file per (family, arch)
benchmark/<family>/bench_<family>_<dtypes>_<arch>.cpp
examples/api/<area>/[<topic>/]<scenario>.cpp  # optional topic, no ISA/tile suffix
examples/integration/<method>/main.cpp     # independent consumer
```

## Worked examples

| Kernel filename | Decomposition |
|---|---|
| `tqt_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv.cpp` | FP16 GEMM, per-N bias, clamp; B is transposed; 8×(3·vlmax) tile; RVV. |
| `tqt_gemm_clamp_i32_i8_i8_4x1vl_rvv.cpp` | Plain int8 GEMM with i32 accumulator output, no bias, only clamp; 4×vlmax tile; RVV. |
| `tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv.cpp` | Int8 quantized GEMM: A is asymmetric int8, B is symmetric int8 with per-channel scale and transposed; output is asymmetric int8 (after i32→i8 requant + clamp); per-N bias; 4×vlmax tile; RVV. |
| `tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv.cpp` | W4A16 GEMM: FP16 activation × symmetric int4 weight with block-of-2 group quantization, transposed; 8×(2·vlmax) tile; RVV. |
| `tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_4x1vl_rvv.cpp` | Packed qsi8d256 activation × asymmetric U4 block-32 weight; I32 group dot and FP32 output; RVV. |
| `tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv.cpp` | Packed qsi8d256 activation × plane-packed asymmetric U5 block-32 weight; register-time unpack and FP32 output; RVV. |
| `tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p_4x1vl_rvv.cpp` | Packed qsi8d256 activation × plane-packed symmetric I6 block-16 weight; register-time unpack and FP32 output; RVV. |
| `tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv.cpp` | Packed qsi8d32 activation × symmetric I8 block-32 weight; I32 group dot and FP32 output; RVV. |
| `tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv.cpp` | Q1_0 execution GEMM: packed qsi8d32 activation × binary 1-bit weight with symmetric `-1/+1` mapping; register-time bit unpack; RVV. |
| `tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv.cpp` | Q2_0 execution GEMM: packed qsi8d32 activation × U2 code with fixed zero point 1; register-time unpack; RVV. |
| `tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv.cpp` | Q5_0 execution GEMM: packed qsi8d32 activation × signed I5 low4/sign-plane weight; register-time unpack; RVV. |
| `tqt_sgemm_f32_f32_f32t_8x3vl_rvv.cpp` | BLAS-style sgemm: `D = alpha·A·B + beta·C`; B transposed; 8×(3·vlmax) tile; RVV. No bias/clamp segment in name. |
| `tqt_dgemm_f64_f64_f64t_8x3vl_rvv.cpp` | BLAS-style dgemm (double-precision); B transposed; 8×(3·vlmax) tile; RVV. |
| `tqt_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv.cpp` | UINT8 GEMM with dual bias (per-M + per-N), dequantized f32 output; B transposed; 4×vlmax tile; RVV. |
| `tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv.cpp` | W4A8 per-channel GEMM: per-row INT8 activation × per-channel symmetric INT4 weight, transposed; f32 output; 4×vlmax tile; RVV. |
| `tqt_gemm_1xnbias_clamp_bf16_qsi8d2p_qai4c2p_4x1vl_rvv.cpp` | W4A8 block-quantized asymmetric: K-blocked symmetric INT8 activation (packed) × K-blocked asymmetric INT4 weight (packed); bf16 output; 4×vlmax tile; RVV. |
| `tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv.cpp` | W2A8 GEMM: per-row INT8 activation × per-channel symmetric INT2 weight, transposed; f32 output; 4×vlmax tile; RVV. |
| `tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi2cxt_4x1vl_rvv.cpp` | W2A8 GEMM: per-tensor asymmetric INT8 activation, per-channel INT2 weight with optional LUT, raw I32 bias and requantized INT8 output. |
| `tqt_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv.cpp` | FP32 channel-first convolution (im2col + GEMM); per-M bias; both im2col-A and weight-B are pre-packed; 8×(3·vlmax) tile; RVV. |
| `tqt_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv.cpp` | FP32 channel-last implicit-GEMM convolution; per-N bias; weight transposed; 8×(3·vlmax) tile; RVV. |
| `tqt_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv.cpp` | FP32 depthwise channel-first convolution, specialised for 3×3 stride-1; all three tensors packed; 4×1×vlmax tile; RVV. |
| `tqt_addvv_f32_f32_f32_8vl_rvv.cpp` | FP32 elementwise vector-vector add (`c[i] = a[i] + b[i]`); 8×vlmax tile; RVV. No biastype/clamp segment in name. |
| `tqt_sumrow_f32_f32_4x4vl_rvv.cpp` | FP32 row-wise reduction sum (`c[m] = Σ_n a[m,n]`); 4×(4·vlmax) tile (4 rows, 4 independent accumulators); RVV. Output type then single input type only (no B). |
| `tqt_sumcol_f32_f32_2x8vl_rvv.cpp` | FP32 column-wise reduction sum (`c[n] = Σ_m a[m,n]`); 2×(8·vlmax) tile (8·vlmax columns, M unrolled by 2 with dual partial accumulators); RVV. |
| `tqt_sumrow_f16_f16_4x2vl_rvv.cpp` | FP16 row-wise reduction sum; FP32 widening accumulator, narrowed back to FP16 on store; 4×(f16m2) tile (4 rows, 4 independent accumulators); RVV. |
| `tqt_meanrow_f32_f32_4x4vl_rvv.cpp` | FP32 row-wise reduction mean (`c[m] = (1/N)·Σ_n a[m,n]`); 4×(4·vlmax) tile; RVV. |
| `tqt_meancol_f16_f16_2x4vl_rvv.cpp` | FP16 column-wise reduction mean; FP32 widening accumulator, narrowed back to FP16 on store; 2×(2·f16m2) tile (M unrolled by 2 with dual partial accumulators); RVV. |
| `tqt_relu_f32_f32_1x8vl_rvv.cpp` | FP32 ReLU (`c[i,j] = max(a[i,j], 0)`) over a `[M, N]` matrix with independent `lda` / `ldc`; 1×(8·vlmax) tile (one row per invocation, widest LMUL); RVV. Output type then single input type only (no B). |
| `tqt_gelu_f32_f32_1x4vl_rvv.cpp` | FP32 standard tanh-approximate GeLU using the shared rational `tanh`; 1×(4·vlmax) tile; RVV. |
| `tqt_reglu_f32_f32_f32_1x8vl_rvv.cpp` | FP32 ReGLU (`c[i,j] = relu(a[i,j]) · b[i,j]`); independent `lda` / `ldb` / `ldc`; 1×(8·vlmax) tile; RVV. |
| `tqt_silu_f32_f32_1x4vl_rvv.cpp` | FP32 SiLU evaluated as `a / (1 + exp(-a))` with the shared FMA-based polynomial `exp`; 1×(4·vlmax) tile (one row per invocation, LMUL=4); RVV. |
| `tqt_silu_f16_f16_1x2vl_rvv.cpp` | FP16 SiLU; one `f16m2` group is widened into `f32m4`, `exp`, the add and the divide all run in FP32, and the result is narrowed once on store; 1×(1·f16m2) tile; RVV. |
| `tqt_swiglu_f32_f32_f32_1x4vl_rvv.cpp` | FP32 SwiGLU (`c[i,j] = silu(a[i,j]) · b[i,j]`); a gated activation with a `dtypeB` slot and a third leading dimension `ldb`. Same `1x4vl` tile and shared polynomial `exp` as the FP32 SiLU, with the linear operand loaded after `exp` has released its temporaries; RVV. |

## Cross references

- Full catalog of variants currently shipped: see [micro-kernel tables](microkernel_tables.md).
- Scenario-based runnable examples: see [`examples/`](../../examples/).
- Operator implementations: see [`src/`](../../src/) and the per-variant `_interface.h` header that ties tile + arch variants together.

W4A16 F32-output `qsi4c2` and `qai4c2` also provide `8x1vl_rvv` variants:
`m_step = MR = 8`, `n_step = VLENB / 2`, one N subgroup, and F32 accumulators.
The `qai4c2` descriptor represents unsigned codes with independent F32 block
scale and additive offset. LUT support is a runtime option of the symmetric
`qsi4c2` contract and does not introduce a new public strategy suffix.
