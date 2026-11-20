<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Quantization Guide

This document covers every quantized kernel path in TORQ-Tile: data types,
granularity schemes, parameter structs, and the naming conventions that tie
them together.

On this page:

- [Overview: Three Output Paths](#overview-three-output-paths)
- [Quantization Granularity](#quantization-granularity)
- [Symmetric vs. Asymmetric](#symmetric-vs-asymmetric)
- [INT8 x INT8 Paths](#int8-x-int8-paths)
- [INT4 Paths](#int4-paths)
- [INT2 Path (W2A8)](#int2-path-w2a8)
- [RVV Quantized-Activation Integer-Dot GEMM](#rvv-quantized-activation-integer-dot-gemm)
- [Ztt Q4_K and Expanded Q6_K Integer Dot](#ztt-q4_k-and-expanded-q6_k-integer-dot)
- [Scale and Zero-Point Memory Layout](#scale-and-zero-point-memory-layout)
- [Choosing a Quantization Path](#choosing-a-quantization-path)
- [Parameter Passing Convention](#parameter-passing-convention)
- [F32 bias for narrow floating outputs](#f32-bias-for-narrow-floating-outputs)
- [Clamp Values](#clamp-values)
- [See Also](#see-also)

## Overview: Three Output Paths

Integer-quantized GEMM paths accumulate integer dot products before their output
epilogue. The main output choices are listed below. W4A16 separately combines
floating activations with dequantized INT4 weights.

| Path | Output type | Naming segment | Description |
|------|-------------|----------------|-------------|
| Raw accumulator | `i32` | `gemm_clamp_i32_i8_i8` | No dequant, no requant. Direct i32 output with clamp. |
| Requantized | `qai8` (int8) | `gemm_*_reqi32toi8_clamp_*` | Requantize i32 to int8 using output scale/zp. |
| Dequantized | `f32` / `f16` | `gemm_*_clamp_f32_qai8dx_*` | Dequantize i32 to float using per-row/per-channel scales. |

### Execution Contract Layers

For llama.cpp-derived block formats, keep four separate concepts when selecting or
integrating a kernel:

| Layer | Meaning | Examples |
|-------|---------|----------|
| Source format | Model/container format decoded by an external adapter | Q2_K, Q5_K, Q8_0 |
| Logical execution encoding | Code and metadata tensors accepted by TORQ-Tile packers | `qau2c16`, `qsi4c16`, `qau8c32` |
| Compute path | Whether A is floating point or quantized and where dequantization occurs | W4A16 dequantization, quantized-activation integer-dot |
| Physical packed layout | Architecture-private buffer consumed by one kernel variant | `...p`, RVV N-interleaved, Ztt square packing |

The source-format name therefore describes a common importer input, not a raw object
consumed by the micro-kernel. A packed (`p`) buffer is an execution artifact, not a
stable serialization format. In this guide, **native** means the quantized-activation
integer-dot path; it never means that the kernel parses a raw GGUF block.

The quantized-activation integer-dot paths are:

| Source | A contract | B execution encoding | Storage | `bsums_a` | Arch |
|--------|------------|----------------------|---------|------------|------|
| Q2_K | `qsi8d256` | `qau2c16` | compact U2 | required | RVV |
| Q2_K | `qsi8d256` | `qau4c16` | nibble-expanded U4 | required | RVV |
| Q3_K | `qsi8d256` | `qsi3c16` | compact signed I3 | not used | RVV |
| Q3_K | `qsi8d256` | `qsi4c16` | nibble-expanded signed I4 | not used | RVV |
| Q4_K | `qsi8d256` | `qau4c32` | normalized asymmetric U4 | required | RVV |
| Q5_K | `qsi8d256` | `qau8c32` | byte-expanded U8 | required | RVV |
| Q5_K | `qsi8d256` | `qau5c32` | plane-packed U5 | required | RVV |
| Q6_K | `qsi8d256` | `qsi8c16` | byte-expanded signed I8 | not used | RVV |
| Q6_K | `qsi8d256` | `qsi6c16` | plane-packed offset-binary I6 | required for the `-32` correction | RVV |
| Q8_0 | `qsi8d32` | `qsi8c32` | signed I8 | not used | RVV |

## Quantization Granularity

The granularity suffix on the A or B type slot tells you how many scale (and
optionally zero_point) values exist:

| Suffix | Name | Layout | Example |
|--------|------|--------|---------|
| (none) | Per-tensor | 1 scalar scale for the whole tensor | `qai8` |
| `dx`   | Per-dimension | `scale_a[m]` -- one scale per row of A | `qai8dx` |
| `cx`   | Per-channel | `scale_b[n]` -- one scale per column of B | `qsi8cx` |
| `d2`   | Per-dim + K-blocked | `scale_a[m, k/bl]` -- one per (row, K-block) | `qsi8d2` |
| `c2`   | Per-channel + K-blocked | `scale_b[n, k/bl]` -- one per (col, K-block) | `qsi4c2` |

## Symmetric vs. Asymmetric

| Prefix | Meaning | Zero point |
|--------|---------|------------|
| `qs`   | Symmetric | Fixed at 0 (not stored) |
| `qa`   | Asymmetric | Stored per scale group |

Examples:
- `qai8`: asymmetric int8, per-tensor (1 scale, 1 zero_point)
- `qsi8cx`: symmetric int8, per-channel (N scales, zero_point = 0)
- `qai8dx`: asymmetric int8, per-dimension (M scales, M zero_points)
- `qsi8d2`: symmetric int8, per-dim + K-blocked (M * K/bl scales, no zp)

---

## INT8 x INT8 Paths

All INT8 kernels live under `src/gemm/gemm_i8_i8/`.

### Path 1: Raw i32 Output

```
gemm_clamp_i32_i8_i8_4x1vl_rvv
gemm_clamp_i32_i8p_i8p_4x1vl_rvv   (packed variant)
```

No bias, no quantization parameters. Simply: `D[m,n] = clamp(sum_k A[m,k]*B[k,n], min, max)`.
Output is `int32_t`.

### Path 2: Requantized i8 Output (`reqi32toi8`)

```
gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv
gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv
```

The i32 accumulator is requantized to int8 using output quantization (scale_d,
zero_point_d). Params struct:

```c
// Defined in src/gemm/gemm_i8_i8/tqt_params_i8_i8.h

struct tqt_qai8_qai8_qsi8cx_params {
    float scale_a;            // per-tensor A scale
    int32_t zero_point_a;     // per-tensor A zero point
    float *scale_b;           // [N] per-channel B scales
    int32_t quant_channel_b;  // number of B channels (N)
    float scale_d;            // output scale
    int32_t zero_point_d;     // output zero point
};
```

Run function signature:

```c
void tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv(
    size_t m, size_t n, size_t k,
    const void *A, size_t lda, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b,
    const void *C, size_t ldc,
    void *D, size_t ldd,
    const void *bias,                              // int32 [1, n]
    int8_t clamp_min, int8_t clamp_max,
    const struct tqt_qai8_qai8_qsi8cx_params *params);
```

Bias is raw `int32_t` in the accumulator domain. The kernel applies input
zero-point correction; callers must not fold that correction into bias. The
`1xnbias` accumulator is `C + bias + sum((A - zero_point_a) * B)`; the
`mx1bias` accumulator is `C + bias + sum(A * (B - zero_point_b))`.
The complete accumulator must fit I32.

Both RVV convert `scale_a * scale_b / scale_d` to a Q31 multiplier
and signed exponent. Requantization rounds the complete I64 product scaled by
`2^(exponent - 31)` once, to nearest with ties to even, then adds
`zero_point_d`, saturates to I8, and applies the requested clamp. Multipliers
below, equal to, and above one are supported. The rounding result does not
depend on the caller's floating-point or vector fixed-point rounding mode.
Scales must be finite and nonnegative, with `scale_d > 0`; zero points must
be representable in I8.

Quantized convolution `run_im2col` and `run_a/b_im2colpack` take an explicit
`int8_t padding_value` after the convolution parameters. Pass the activation
zero point. IGEMM handles these real-zero boundaries internally, including
grouped inputs. Floating-point im2col retains its default zero padding.

Migration from older callers: remove bias zero-point folding and pass the
activation zero point to quantized im2col.

### Path 3: Dequantized Float Output

```
gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv
gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv
gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv   (packed)
```

The i32 accumulator is dequantized to float:
`D[m,n] = clamp((acc[m,n] - zp_a[m] * col_sum_b[n]) * scale_a[m] * scale_b[n] + bias[n], min, max)`

Params struct:

```c
struct tqt_qai8dxp_qsi8cxp_params {
    float *scale_a;           // [M] per-row A scales
    int32_t *zero_point_a;    // [M] per-row A zero points
    float *scale_b;           // [N] per-channel B scales
};
```

> **Note:** Despite the `p` suffix in the struct name, this params struct is shared
> by both packed and non-packed dequant-path variants (e.g.,
> `gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv`).

Run function signature:

```c
void tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv(
    size_t m, size_t n, size_t k,
    const void *A, size_t lda, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b,
    const void *C, size_t ldc,      // must be nullptr (not supported)
    void *D, size_t ldd,
    const void *bias,               // float [1, n], may be nullptr
    float clamp_min, float clamp_max,
    const struct tqt_qai8dxp_qsi8cxp_params *params);
```

---

## INT4 Paths

INT4 kernels are split across two family directories:
- `src/gemm/gemm_f16_i4/` -- W4A16 (fp16 activation x int4 weight)
- `src/gemm/gemm_i8_i4/` -- W4A8 (int8 activation x int4 weight)

### Nibble Encoding

INT4 values are stored as uint8 with two nibbles per byte:
- **Low nibble** (bits [3:0]) = element at position `k+0`
- **High nibble** (bits [7:4]) = element at position `k+1`

Two encoding schemes exist:

| Tag | Name | Dequant formula | Range |
|-----|------|-----------------|-------|
| `qsi4` | Symmetric (ZP=8) | `(nibble - 8) * scale` | signed [-8, 7] stored as [0, 15] |
| `qai4` | Asymmetric (ZP=0) | `nibble * scale + offset_b` | unsigned [0, 15] with independent per-block fp32 offset |

### LUT16 and Independent Affine Offsets

W4A16 symmetric `qsi4c2` run functions take a final nullable `const float16_t *lut`.
The weight is `float(lut[code]) * float(scale_b)` when a table is supplied, or
`(code - 8) * scale_b` otherwise. Existing callers must append `NULL` to preserve
the default decoding. W4A8 variable-block symmetric parameters carry the analogous
`const int8_t *lut`; value-initialize these structs or explicitly set `lut = NULL`.
Both tables have exactly 16 entries, and all nibbles are indices in `[0, 15]`.
The fixed `tqt_qsi8d32_qsi4c32_params` and its two's-complement packing are unchanged.

The W4A16 F32-output `8x1vl_rvv` variants use F32 accumulation, C, bias and clamp.
They widen the FP16 scales and LUT values before arithmetic. The existing
`8x2vl_rvv` variants retain F16 accumulation and F16 intermediate rounding even
when D is F32. Their packed B layouts differ; use the packers/getters belonging
to the selected variant. Both accept even block lengths `bl >= 2`, with K and
K origins block-aligned, including all K values divisible by block size 32.

W4A8 `qsi8d2 × qai4c2` now computes `code * scale_b + offset_b` per block.
Replace `zp_b`/`ldzpb` with `offset_b`/`ldob`, and migrate values with
`offset_b = -old_zp_b * scale_b`. The getter is now `get_offset_b_offset`.
The independent offset supports constant nonzero blocks when `scale_b == 0`.
W4A16 adds the direct and packed `gemm_1xnbias_clamp_f32_f16[p]_qai4c2[t/p]`
`8x1vl_rvv` bases with F32 scale/offset, F32 accumulation and F32 bias. These
functions take the two metadata pointers/strides directly; offset them to the
selected K-block before calling. W4A8 params retain their established convention:
the kernel adds the absolute K-block origin to the params' full-K metadata.

### W4A16: FP16 Activation x K-Blocked INT4 Weight

```
gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv
gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv
gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv   (packed A)
```

- A: plain `f16` (or packed `f16p`)
- B: `qsi4c2t` -- symmetric INT4, K-blocked (`c2`), transposed (`t`)
- D: `f16` or `f32`
- Block length `bl` is passed at run-time.

The run function takes `scale_b` directly as a parameter (no params struct):

```c
void tqt_run_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(
    size_t m, size_t n, size_t k, size_t bl,
    const void *A, size_t lda, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b,
    const void *scale_b, size_t ldsb,   // [n, k/bl] f16 row-major
    const void *C, size_t ldc,
    void *D, size_t ldd,
    const void *bias,
    float16_t clamp_min, float16_t clamp_max,
    const float16_t *lut);  // optional 16-entry decode table
```

For a non-zero K origin, `k_idx_b` must be block aligned (`k_idx_b % bl == 0`).
Select A/B with their data-offset getters, select `scale_b` separately with
`get_scale_b_offset(n_idx, k_idx_b / bl, ldsb)`, retain the full logical
strides, and pass the absolute K origins to `run_gemm`. The direct `scale_b`
argument already denotes the selected scale origin and is not advanced again
inside the kernel.

### W4A8 Per-Channel: INT8 Activation x Per-Channel INT4 Weight

```
gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv
gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv
gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv   (packed)
```

- A: `qai8dx` -- per-row asymmetric INT8
- B: `qsi4cx` -- per-channel symmetric INT4 (no K-block)
- D: `f16` / `f32` / `bf16`

Params struct (from `src/gemm/gemm_i8_i4/tqt_params_i8_i4.h`):

```c
struct tqt_qai8dx_qsi4cx_params {
    const float *scale_a;         // [M]
    const int32_t *zero_point_a;  // [M]
    const float *scale_b;         // [N]
};
```

### W4A8 Block-Quantized Symmetric: INT8 x K-Blocked Symmetric INT4

```
gemm_1xnbias_clamp_f16_qai8dx_qsi4c2t_4x1vl_rvv
gemm_1xnbias_clamp_f16_qsi8d2_qsi4c2t_4x1vl_rvv
gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv   (packed)
```

Two A-side options:

| A-side | Meaning |
|--------|---------|
| `qai8dx` | Per-row asymmetric INT8 (no K-block on A) |
| `qsi8d2` | Per-row symmetric INT8 with K-blocked scale (no zero point) |

Params structs:

```c
struct tqt_qai8dx_qsi4c2_params {
    const float *scale_a;         // [M]
    const int32_t *zero_point_a;  // [M]
    const float *scale_b;         // [N, K/bl] row-major
    size_t ldsb;                  // leading dim of scale_b (typically K/bl)
    const int8_t *lut;      // optional 16-entry signed decode table
};

struct tqt_qsi8d2_qsi4c2_params {
    const float *scale_a;   // [M, K/bl] row-major
    size_t ldsa;            // leading dim of scale_a
    const float *scale_b;   // [N, K/bl] row-major
    size_t ldsb;            // leading dim of scale_b
    const int8_t *lut;      // optional 16-entry signed decode table
};
```

For W4A8 `reqi32toi8` output, `tqt_qai8_qai8dx_qsi4cx_params::bias_i32`
is an optional raw I32 bias vector indexed by N. It is added after activation
zero-point correction and before F32 dequantization. The separate F32 `bias`
argument and F32 residual C remain supported, including simultaneous use with
`bias_i32`. Value-initialize the parameters, or explicitly set `bias_i32 = NULL`
when migrating callers. Requantization and its F32 arithmetic use nearest-even
rounding and restore the incoming `frm`. The final rescale uses direct F32 division by positive finite `scale_d`, including
subnormal values; it does not form a potentially overflowing reciprocal. Float saturation before conversion
prevents I32 overflow when adding the output zero point.

### W4A8 Block-Quantized Asymmetric: INT8 x K-Blocked Asymmetric INT4

```
gemm_1xnbias_clamp_f16_qsi8d2_qai4c2t_4x1vl_rvv
gemm_1xnbias_clamp_bf16_qsi8d2p_qai4c2p_4x1vl_rvv   (packed)
```

- A: `qsi8d2` -- symmetric INT8, K-blocked (no zero point)
- B: `qai4c2` -- **asymmetric** INT4, K-blocked, with independent per-block `offset_b`

Params struct:

```c
struct tqt_qsi8d2_qai4c2_params {
    const float *scale_a;   // [M, K/bl]
    size_t ldsa;
    const float *scale_b;   // [N, K/bl]
    size_t ldsb;
    const float *offset_b;  // [N, K/bl] fp32 additive offset for B
    size_t ldob;            // leading dim of offset_b
};
```

---

## INT2 Path (W2A8)

Kernels live under `src/gemm/gemm_i8_i2/`.

```
gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv
gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv   (packed)
```

- A: `qai8dx` -- per-row asymmetric INT8
- B: `qsi2cx` -- per-channel symmetric INT2, stored as `[n, k/4]` uint8 (4
  crumbs per byte)
- D: `f32`

INT2 decoding uses the LUT: `u2 in {0,1,2,3} -> i8 in {-2,-1,0,1}`
(equivalent to `u2 - 2`).

Params struct (from `src/gemm/gemm_i8_i2/tqt_params_i8_i2.h`):

```c
struct tqt_qai8dx_qsi2cx_f32_params {
    const float *scale_a;         // [M]
    const int32_t *zero_point_a;  // [M]
    const float *scale_b;         // [N]
    const int32_t *sum_b;         // [N] precomputed: sum_k lut[u2_b[n][k]]
};
```

The caller must precompute `sum_b[n]` -- the row-sum of dequantized int2
weights per output channel. This is used for the zero-point correction:
`iacc -= zp_a[m] * sum_b[n]`. A helper is provided in
[`src/gemm/gemm_i8_i2/tqt_utils_i8_i2.h`](../../src/gemm/gemm_i8_i2/tqt_utils_i8_i2.h).

`sum_b` has no K dimension, but its **value** is the sum over the K window the
call consumes, so it is not K-invariant: a caller that splits K must recompute
`sum_b` for each window rather than reuse the full-K vector. The same holds for
`scale_a` / `zero_point_a`, which describe the activation window being passed
in. This is a caller obligation on every K-sliced call, including the ones where
the kernel itself ignores `k_idx_a` / `k_idx_b` because both operands are raw
(see the selected-origin contract in
[`../reference/api_reference.md`](../reference/api_reference.md)).
`test/gemm/test_gemm_i8_i2_rvv.cpp` shows the per-window recomputation.

### Requantized I8 output

The `gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi2cxt_4x1vl_rvv` and
`gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv` variants take
`tqt_qai8_qai8_qsi2cx_params`: scalar `scale_a`, `zero_point_a`, `scale_d`,
`zero_point_d`, and `scale_b[N]`. Their bias argument is nullable raw
`const int32_t *`. The kernel computes its own decoded weight sums for the
current K window and LUT, so no external sum metadata is needed.

The result is `clamp(RNE(float(dot + bias) * scale_a * scale_b / scale_d)
+ zero_point_d)`, with the arithmetic evaluated in F32 and nearest-even mode.
Division by `scale_d` is performed directly in F32 under RNE. Incoming `frm` is
restored on return. An optional four-entry INT8 LUT maps the original two-bit
codes; a null LUT selects `{-2,-1,0,1}`. Both raw and packed contracts require
K and K offsets divisible by four, and `ldb` divisible by four. To adapt a
logical K tail, pad activation elements with `zero_point_a` and allocate the
corresponding weight bytes. Bias and scale offsets follow the selected output
channel origin; output and packed-size getters return bytes.

---

## RVV Quantized-Activation Integer-Dot GEMM

These native integer-dot paths consume adapter-produced weight execution tensors
and replace floating-point A with the activation quantization selected by
llama.cpp's type traits:

| Weight source | A token | A block | B token | B block |
|---------------|---------|---------|---------|---------|
| Q2_K compact | `qsi8d256p` | 256 | `qau2c16p` | 16 |
| Q2_K nibble-expanded | `qsi8d256p` | 256 | `qau4c16p` | 16 |
| Q3_K compact | `qsi8d256p` | 256 | `qsi3c16p` | 16 |
| Q3_K nibble-expanded | `qsi8d256p` | 256 | `qsi4c16p` | 16 |
| Q4_K | `qsi8d256p` | 256 | `qau4c32p` | 32 |
| Q5_K byte-expanded | `qsi8d256p` | 256 | `qau8c32p` | 32 |
| Q5_K plane-packed | `qsi8d256p` | 256 | `qau5c32p` | 32 |
| Q6_K byte-expanded | `qsi8d256p` | 256 | `qsi8c16p` | 16 |
| Q6_K plane-packed | `qsi8d256p` | 256 | `qsi6c16p` | 16 |
| Q8_0 | `qsi8d32p` | 32 | `qsi8c32p` | 32 |

The caller supplies row-major int8 A codes and calls `run_a_pack`; no f32-to-I8
quantizer is part of TORQ-Tile. A metadata stays unpacked:

- `qsi8d256`: `scale_a[M, K/256]`; Q2_K, Q4_K/Q5_K, and plane-packed Q6_K
  additionally require `bsums_a[M, K/16]` as exact int16 sums of the quantized A codes.
- `qsi8d32`: `scale_a[M, K/32]`.
- B scale and real-zero arrays are fp32 row-major tensors. Q4_K/Q5_K use
  `[N, K/32]` scale and real-zero; Q6_K uses `[N, K/16]` scale; Q8_0 uses
  `[N, K/32]` scale.

For each asymmetric 32-value weight group, the kernel computes:

```
scale_a * (scale_b * dot(A_i8, B_u) - real_zero_b * sum(A_i8))
```

Symmetric Q6_K/Q8_0 groups compute
`scale_a * scale_b * dot(A_i8, B_i8)`. Group dot products accumulate in i32;
group scaling, C, 1xN bias, clamp, and D use fp32. Q4_K/Q5_K/Q6_K require K and
Split-K boundaries to be 256-aligned. Q8_0 requires 32 alignment. A and B
`k_idx` values must match; on later splits, pass the previous partial result as C
and apply bias only on the final split.

RVV provides every row in the table, including compact Q2_K/Q3_K and
plane-packed Q5_K/Q6_K variants. Use the matching packers for each execution
encoding; byte-expanded, nibble-expanded and plane-packed buffers are distinct.

### Q2_K/Q3_K Factorized Execution Formats

The Q2_K/Q3_K paths preserve the original integer group metadata instead of
pre-multiplying it into one fp32 scale per group. An external GGUF adapter emits
code rows and the following row-major metadata:

| Source | Group metadata `[N, K/16]` | Super metadata `[N, K/256]` |
|--------|-----------------------------|------------------------------|
| Q2_K | `uint8_t qscale_b`, `uint8_t qmin_b` | `float super_scale_b`, `float super_min_b` |
| Q3_K | `int8_t qscale_b` | `float super_scale_b` |

For one output element and one 256-value super-block, Q2_K computes

```text
scale_a * (super_scale_b * sum_g(qscale_b[g] * dot_g)
           - super_min_b * sum_g(qmin_b[g] * bsum_a[g]))
```

and Q3_K computes

```text
scale_a * super_scale_b * sum_g(qscale_b[g] * dot_g)
```

where each `dot_g` spans 16 K values and `bsum_a[g]` is the exact int16 sum of
those activation codes. The integer sums complete before fp32 super-block
scaling. C, 1xN bias, clamp, and D are fp32.

Compact Q2_K B rows store four 2-bit codes per byte (`K/4` bytes). Compact
Q3_K rows store four low-two-bit bytes followed by two high-bit bytes per
16-value group (`3*K/8` bytes); a cleared high bit selects the negative
`low2-4` range. Expanded forms use one signed/unsigned nibble per value
(`K/2` bytes). RVV decodes all four formats.

All four public variants require K and `k_idx` to be multiples of 256. Packed
buffers contain only code bytes; factorized metadata remains unpacked and may
be shared between compact and expanded variants.

### Plane-Packed Q5_K/Q6_K Code Layout

The plane-packed variants keep only quantized codes in packed B. Scale,
real-zero, and activation-sum metadata remain in the parameter structure.
`run_bt_pack` continues to receive logical K dimensions.

For each Q5_K 32-value group, the row-major source contains 20 bytes:

```text
low4[16]   // adjacent codes: q[2*p] in low nibble, q[2*p+1] in high nibble
high1[4]   // bit j stores bit 4 of q[8*h+j]
```

For each Q6_K 16-value group, let `u = q + 32`; the source contains 12 bytes:

```text
low4[8]    // adjacent offset-binary codes
high2[4]   // 2-bit high fields for four adjacent codes
```

The RVV packer transforms these streams to
`[N tile][K group][plane-byte position][N lane]`. Packed B sizes are
`N*K*5/8` and `N*K*3/4`. Q5 high1 data is unpacked once per eight K values;
Q6 high2 data is unpacked once per four K values. The public `4x1vl` kernel
processes two A rows together so the decoded B vectors are reused.

For Q6, the inner loop accumulates offset-binary U6 values and applies
`dot(A, q) = dot(A, u) - 32*sum(A)` once per 16-value group.

An adapter reading canonical `block_q5_K` or `block_q6_K` may write the final
N-interleaved layout directly while extracting fp32 parameters. This fused
load-time traversal requires neither an expanded `N*K` buffer nor a complete
row-major plane-packed intermediate.

For canonical Q5_K group `g` and in-group index `i`:

```text
low  = (block.qs[(g/2)*32 + i] >> (4*(g&1))) & 0xf
high = (block.qh[i] >> g) & 1
```

For canonical Q6_K group `g`, define `half=g/8`, `local=g%8`, and
`segment=local/2`:

```text
ql_index = half*64 + (segment&1)*32 + (local&1)*16 + i
qh_index = half*32 + (local&1)*16 + i
low      = (block.ql[ql_index] >> (segment >= 2 ? 4 : 0)) & 0xf
high     = (block.qh[qh_index] >> (2*segment)) & 3
```

After composing the group-local plane bytes, lane `l` writes byte position `p`
to `tile[(group*group_bytes+p)*nr+l]`. During the same traversal, the adapter
writes Q5_K `scale_b=d*scale`, `real_zero_b=dmin*min`, or Q6_K
`scale_b=d*scale` into their final row-major parameter arrays.

### Q8_0 Activation with Q1_0/Q2_0/Q4_0/Q5_0 Weights

These kernels consume TORQ execution tensors produced by an external adapter;
they do not parse raw GGML blocks. A is always packed signed-I8 Q8_0 code with
`scale_a[M,K/32]`. Each 32-value group forms an I32 dot, converts to fp32, and
then multiplies the matching activation and weight scales.

| Weight encoding | Decoded value | Bytes per 32 codes | Weight scales | RVV storage |
|---|---|---:|---|---|
| `qsb1c128` | `bit ? 1 : -1` | 4 | `[N,K/128]` | compact |
| `qau2z1c64` | `u2 - 1` | 8 | `[N,K/64]` | compact |
| `qsi4c32` | signed two's-complement I4 | 16 | `[N,K/32]` | native I4 |
| `qsi5c32` | signed two's-complement I5 | 20 | `[N,K/32]` | compact low4/sign plane |
| `qsi8c32` | signed I8 | 32 | `[N,K/32]` | existing baseline |

The canonical stream is row-major `[N,K]` before `run_bt_pack`: B1 uses
least-significant-bit-first K order, U2 stores four adjacent codes per byte,
I4 stores two adjacent values with the smaller K in the low nibble, and I5
stores 16 low-nibble bytes followed by a four-byte sign plane per 32 values.
Raw Q4_0/Q5_0 half-split nibbles therefore require reordering. Raw GGML scale
headers are never part of the TORQ code stream.

Q1_0 and Q2_0 expanded execution duplicates each source weight scale four or
two times respectively to form `[N,K/32]`; compact execution keeps the original
scale granularity. All variants require `k`, `k_idx_a`, and `k_idx_b` to be
32-aligned and require `k_idx_a == k_idx_b`. A compact Q1/Q2 split may start
inside a 128/64-value weight block and reuses that block's scale.

All dimensions passed to pack and compute functions are logical element counts;
packed-size and offset getters return bytes. The compute entry points consume
packed A/B tensors and enforce these row-stride constraints:

| B token and compute path | Packed-A `lda` passed to `run_gemm` | Packed-B `ldb` passed to `run_gemm` |
|--------------------------|--------------------------------------|--------------------------------------|
| `qsb1c128` | multiple of 32 | multiple of 128 |
| `qau2z1c64` | multiple of 32 | multiple of 64 |
| `qsi4c32` | multiple of 32 | multiple of 32 |
| `qsi5c32` | multiple of 32 | multiple of 32 |
| `qsi8c32` RVV | multiple of 32 | no additional multiple |

Packing has a separate source/destination stride contract. Use 32-aligned `k`
and `k_idx` for every Qx_0 execution path. RVV `run_a_pack` adds no divisibility
requirement to its source `lda` or destination `lda_packed`. B packing converts logical
element strides to byte strides as follows:

| B token and pack path | Source `ldb` | Destination `ldb_packed` |
|-----------------------|--------------|--------------------------|
| `qsb1c128` RVV | multiple of 8 | multiple of 8 |
| `qau2z1c64` RVV | multiple of 4 | multiple of 4 |
| `qsi4c32` RVV | multiple of 2 | multiple of 2 |
| `qsi5c32` RVV | multiple of 32 | multiple of 32 |
| `qsi8c32` RVV | no additional multiple | no additional multiple |

The destination strides become `lda`/`ldb` when the packed buffers are passed
to `run_gemm`, so they must then satisfy the stronger compute table above.
The low-bit RVV wrappers state these divisor preconditions with `TQT_ASSUME`
before converting logical element counts to byte strides. Setting all full-K
source and destination strides to a suitably block-aligned `K` satisfies both
contracts.

Packed sizes cover codes only:

```text
qsb1c128p:  N*K/8 bytes       qau2z1c64p: N*K/4 bytes
qsi4c32p:   N*K/2 bytes       qsi5c32p:    N*K*5/8 bytes
qsi8c32p:   N*K bytes
```

The public variant determines the packed layout. An adapter must call the
matching size, offset, and pack functions and must not reuse a packed buffer
across variants or across different runtime VLEN configurations.

---

## Ztt Q4_K and Expanded Q6_K Integer Dot

Ztt reuses the existing `qsi8d256` activation and `qau4c32` / `qsi8c16` weight
definitions, params structures and public interface signatures. Both variants
return ordinary F32 output and accept arbitrary logical M/N. K and both equal
K origins must be multiples of 256. GGUF decoding and activation quantization
remain caller responsibilities.

For each 32-value Q4 group, packing changes the U4 code to `r = q - 8`.
The kernel computes an I8-by-I4 I32 dot and restores the unsigned dot exactly:
`dot_q = dot_r + 8 * sum_a`, where `sum_a` combines the two supplied exact
16-value activation sums. Its contribution is
`scale_a * (scale_b * dot_q - real_zero_b * sum_a)`.
For each 16-value Q6 group, the signed-I8 dot contributes
`scale_a * scale_b * dot`. Expanded Q6 has no `-32` correction and needs no
activation sums. Adjacent groups and 256-value activation blocks retain their
own scales; the native MMA K span never changes the quantization granularity.

Each group converts its exact I32 result to F32 and uses Ztt floating operations
for scaling and accumulation. Floating reassociation can differ from RVV;
this is semantic compatibility, not bit-exact FP32 compatibility. C and column
bias are optional; C may equal D with identical strides. Clamp retains the
existing number semantics, including NaN bounds, rather than Ztt's propagating
NaN min/max behavior. Empty M/N is a no-op; K=0 processes C/bias/clamp only.

The implementation reuses loaded panels and masks A when a native panel spans
multiple groups; smaller panels accumulate directly without masks. All scratch
is on the stack, dispatched for S=1/2/4/8/16/32/64/128. Larger S and mismatched
UDS are rejected. Existing ownership is retained and sticky unsupported status
is checked. See [quantized Ztt packing](packing.md#ztt-quantized-integer-dot-packing)
for the private packed ABI and Split-K rules.

---

## Scale and Zero-Point Memory Layout

### Per-row (A-side) scales

```
scale_a:       float [M]     -- one fp32 scale per row of A
zero_point_a:  int32_t [M]   -- one zero-point per row of A (0 for symmetric)
```

### Per-channel (B-side) scales

```
scale_b:       float [N]     -- one fp32 scale per output channel
```

### K-blocked scales

```
scale_a:  float [M, K/bl]    -- row-major, ldsa = K/bl
scale_b:  float [N, K/bl]    -- row-major, ldsb = K/bl
offset_b: float [N, K/bl]    -- independent additive term for asymmetric INT4 (qai4c2)
```

Block length `bl` must be >= 2 and a multiple of 2. The dimension `K` must be
a multiple of `bl`.

---

## Choosing a Quantization Path

| Use case | Recommended path | Output |
|----------|-----------------|--------|
| Framework provides its own dequant | `gemm_clamp_i32_i8_i8` | raw i32 |
| Full quantized inference (i8 in, i8 out) | `*_reqi32toi8_clamp_*` | int8 |
| Mixed-precision (i8 compute, fp output) | `*_clamp_f32_qai8dx_*` | fp32/fp16 |
| W4A16 LLM inference | `*_f16_f16_qsi4c2t_*` | fp16 |
| W4A8 LLM inference (per-channel weight) | `*_qai8dx_qsi4cxt_*` | fp16/f32 |
| W4A8 LLM inference (block-quantized) | `*_qsi8d2_qsi4c2t_*` | fp16/f32/bf16 |
| Q2_K/Q3_K factorized quantized-activation block GEMM | `*_f32_qsi8d256p_{qau2c16p,qau4c16p,qsi3c16p,qsi4c16p}_*` | f32 |
| Q4_K/Q5_K/Q6_K quantized-activation block GEMM | `*_f32_qsi8d256p_*_4x1vl_rvv` | f32 |
| Q8_0 quantized-activation block GEMM | `*_f32_qsi8d32p_qsi8c32p_4x1vl_rvv` | f32 |
| Q1_0/Q2_0/Q4_0/Q5_0 quantized-activation block GEMM | `*_f32_qsi8d32p_{qsb1c128p,qau2z1c64p,qsi4c32p,qsi5c32p,qsi8c32p}_*` | f32 |
| Ultra-low-bit weight (2-bit) | `*_qai8dx_qsi2cxt_*` | f32 |

## Parameter Passing Convention

All quantization parameters are passed via a `const struct *params` argument
as the **last** parameter of the run function. The struct is defined in the
family's own params header, whose path follows directly from the family name:

```text
src/gemm/<family>/tqt_params_<A>_<B>.h        # <family> = gemm_<A>_<B>
```

For example `gemm_i8_u4` uses
[`src/gemm/gemm_i8_u4/tqt_params_i8_u4.h`](../../src/gemm/gemm_i8_u4/tqt_params_i8_u4.h).
Every quantized family that takes a params struct follows this rule without
exception; the families that instead pass their scales as direct run-function
arguments are listed below.

Exceptions: W4A16 kernels (`gemm_f16_i4`) pass `scale_b` and `ldsb` as
direct arguments on the run function rather than through a params struct,
because no A-side quantization exists.

## F32 bias for narrow floating outputs

The RVV `qai8dx × qsi8cx`, `qai8dx × qsi4cx`, `qai8dx × qsi4c2`, and
`qsi8d2 × {qsi4c2,qai4c2}` families provide explicit `1xnbiasf32` bases for
F16/BF16 outputs. Existing `1xnbias` entries retain their original bias storage
and rounding behavior. Both use the public `const void *bias` signature; the
kernel name and documented contract select its storage type. New biasf32 getters
return `n_idx * sizeof(float)`, while original getters retain the narrow type.
F32-output bases already use F32 bias and keep their original names.

For biasf32 entries, bias and clamp execute in F32 before one final output
conversion. This permits large bias values to cancel large dot products without
premature F16 overflow or BF16 rounding. C and D keep their original types;
`qai8dx × qsi8cx` continues to ignore C. Its original F16-bias path also preserves
the existing F16-domain bias/clamp epilogue.

RVV BF16×BF16→F16 also provides an explicit F32-bias contract. Its C residual
remains F16, and the original F16-bias entries remain available.

## Clamp Values

- For **int8 output** (`reqi32toi8`): `clamp_min` / `clamp_max` are `int8_t`.
  Typically `-128` / `127` or framework-specific values.
- For **float output** (dequant path): use the declared `float`, `float16_t`, or
  `bfloat16_t` bounds. Exactly `-INFINITY` / `INFINITY`, converted to that public
  type, skips clamping. `-FLT_MAX` / `FLT_MAX` is a finite interval and still
  requests the existing clamp; it is not a no-clamp sentinel.
- For **i32 output** (raw): `clamp_min` / `clamp_max` are `int32_t`.
For every floating GEMM, both infinite bounds must be supplied together; single-sided
infinite intervals retain the existing behavior. Finite bounds keep the implementation's
original clamp precision, including any internal narrowing. See the
[floating GEMM contract](../reference/api_reference.md#gemm-with-bias-and-clamp) for
NaN and conversion guarantees. Integer clamp ranges are unchanged.

## See Also

- [Packing Guide](packing.md) -- packed variants of quantized kernels
  (`*p_*p`) follow the same pack workflow
- [Naming Convention](../reference/naming_convention.md) -- full breakdown of
  quantization tags
- [Micro-kernel Tables](../reference/microkernel_tables.md) -- complete list of
  quantized variants
