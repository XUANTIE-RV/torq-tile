<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# TORQ-Tile API Reference

On this page:

- [Overview](#overview)
- [Common API Functions](#common-api-functions)
- [Run Functions](#run-functions)
- [Packing Functions (Packed Variants Only)](#packing-functions-packed-variants-only)
- [Ztt GEMM](#ztt-gemm)
- [Interface Header and Runtime Dispatch](#interface-header-and-runtime-dispatch)
- [Canonical Tiling Loop](#canonical-tiling-loop)

## Overview

TORQ-Tile micro-kernels expose descriptor-specific C functions following a consistent
naming pattern. The API is stateless, allocation-free, and designed for integration into
ML framework tiling loops. Public functions use C linkage; many buffer arguments use `void *`, while scalars
and some operator inputs use their declared concrete types.

Each kernel variant is parameterized by **KERNEL_FULL** =
`<KERNEL_BASE>_<tile>_<arch>` (e.g., `gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv`).

The API and header paths are identical in hosted and freestanding builds; no
initialization call or runtime registration is required. Hosted RVV/Ztt
geometry getters cache their first CSR result per implementation translation unit.
Freestanding getters read on demand and avoid that initialization guard. Neither
policy reads optional CSRs while loading the library. Calls and their packed
buffers must still use compatible hardware ISA and widths.

Freestanding assertions trap without stdio, including Debug checks. The platform
supplies any compiler-emitted memory and arithmetic runtime symbols. The installed
`TORQ_TILE_BUILT_FREESTANDING` macro describes the matching binary, while inline
helpers follow the consumer compiler's own mode. See
[freestanding build options](../getting_started/build_options.md#freestanding-core).

### Type Aliases

TORQ-Tile defines type aliases in `src/common/tqt_common.h` for use with
micro-kernel APIs:

| Type | C Type | Description |
|------|--------|-------------|
| `float16_t` | `_Float16` | IEEE 754 half-precision float (FP16) |
| `bfloat16_t` | `__bf16` | Brain floating point (BF16) |

These aliases define storage or scalar argument types. Typed pointers convert to
`void *` or `const void *` automatically where the function signature requires it.

`float16_t` and `bfloat16_t` are declared only when `TQT_HAS_NATIVE_FP16` and
`TQT_HAS_NATIVE_BF16`, respectively, are 1. Check these automatic results with
`#if`; do not predefine them. They describe detected native types, independently
of arithmetic ISA and the library's `TORQ_TILE_HAS_*` capabilities. Native types
and ABI are preserved; headers requiring an unavailable type fail to compile.

The Clang 18+ BF16 fallback covers RISC-V only: on other targets without
`__BFLT16_MANT_DIG__`, `bfloat16_t` is omitted even if `__bf16` is supported.

---

## Common API Functions

### Tile Step Queries

```c
size_t tqt_get_m_step_<KERNEL_FULL>(void);  // M-direction (row) tile step
size_t tqt_get_n_step_<KERNEL_FULL>(void);  // N-direction (column) tile step
```

The caller's tiling loop advances by these values. For RVV, `m_step` is a compile-time
constant (e.g., 8 for `8x3vl`); `n_step` depends on the runtime vector length
(`vlenb / sizeof(element)` multiplied by the N-factor in the tile name).

Implementations with a width cache read backend CSRs on first use and retain the
value in a private cache for that compilation unit. Later calls, including calls
from other threads, must use the same execution width (and UDS for Ztt). Select a
supported backend before invoking getters, offset helpers, packing, or compute
functions; getters are not ISA detection probes. C++ runtime support for thread-safe initialization
is required. See the [runtime integration contract](../guide/integration.md#runtime-isa-and-width-queries).

**DWConv note:** Generic depthwise convolution kernels do not expose `get_m_step` or
`get_n_step`. Instead, they operate on entire spatial volumes via `tqt_conv_params`.
Specialized depthwise kernels expose `get_oh_step` for output-height tiling.

**Elementwise / reduction / activation note:** Elementwise kernels
(`{add,sub,mul,div}{vv,vs}`) expose only `get_n_step` (the 1-D vector tile step) — they
have no M dimension, offset, bias, or size getters. Reduction kernels (`sumrow`,
`sumcol`, `meanrow`, `meancol`) expose both `get_m_step` and `get_n_step`; the
non-reduced output dimension is `m` for row reductions and `n` for column reductions.
Activation kernels (`relu`, `gelu`, `silu`, `reglu`, `geglu`, `swiglu`) also expose both steps, covering the rows
and columns of one inner invocation. The exact set of getters for each is fixed by the
corresponding `_interface.h` header.

---

### Offset Helpers

These functions compute byte offsets into the respective buffers, enabling the caller
to pass correctly-positioned pointers to the `run` function.

#### Source matrix A (LHS)

```c
// Non-packed A
size_t tqt_get_a_offset_<KERNEL_FULL>(size_t m_idx, size_t k_idx, size_t lda);

// Packed A (when A suffix is 'p')
size_t tqt_get_a_packed_offset_<KERNEL_FULL>(size_t m_idx, size_t k_idx,
                                             size_t lda, size_t actual_m);
```

- `m_idx`: starting row index
- `k_idx`: starting column (K-dimension) index
- `lda`: leading dimension of A
- `actual_m`: actual number of rows in the packed panel (packed variants only)

#### Source matrix B (RHS)

```c
// Non-packed B
size_t tqt_get_b_offset_<KERNEL_FULL>(size_t n_idx, size_t k_idx, size_t ldb);

// Packed B (when B suffix is 'p')
size_t tqt_get_b_packed_offset_<KERNEL_FULL>(size_t n_idx, size_t k_idx,
                                             size_t ldb, size_t actual_n);
```

- `n_idx`: starting column index
- `k_idx`: starting row (K-dimension) index
- `ldb`: leading dimension of B
- `actual_n`: actual number of columns in the packed panel (packed variants only)

#### Residual matrix C and Output matrix D

```c
size_t tqt_get_c_offset_<KERNEL_FULL>(size_t m_idx, size_t n_idx, size_t ldc);
size_t tqt_get_d_offset_<KERNEL_FULL>(size_t m_idx, size_t n_idx, size_t ldd);
```

Byte offsets into the C (residual, may be nullptr) and D (output) matrices respectively.

#### Bias vector

These functions are available for `gemm`, `conv`, and `igemm` kernels. BLAS-style
`sgemm`/`dgemm` kernels do not expose bias offset functions — they use `alpha`/`beta`
scalars instead.

```c
// 1xN bias (row-broadcast): offset indexed by column
size_t tqt_get_bias_offset_<KERNEL_FULL>(size_t n_idx);

// Mx1 bias (column-broadcast): offset indexed by row
size_t tqt_get_bias_offset_<KERNEL_FULL>(size_t m_idx);

// Dual bias (mx1bias_1xnbias): separate offsets for per-M and per-N bias vectors
size_t tqt_get_bias_m_offset_<KERNEL_FULL>(size_t m_idx);
size_t tqt_get_bias_n_offset_<KERNEL_FULL>(size_t n_idx);
size_t tqt_get_dst_bias_offset_<KERNEL_FULL>(size_t n_idx);   // f32-output variants only
```

Bias storage is defined by each kernel's documented contract, independently of D;
the public pointer remains `const void *bias`. Explicit `1xnbiasf32` bases in the
variable-block/per-channel W4A8, per-row INT8 and BF16-input/F16-output families
use `n_idx * sizeof(float)` even when C and D contain F16 or BF16 elements.
They fuse F32 bias before the final output conversion. Original `1xnbias`
entries retain their narrow bias storage, offsets and rounding behavior.

`tqt_get_bias_m_offset_*` and `tqt_get_bias_n_offset_*` are exposed by every kernel with
`mx1bias_1xnbias` in its name (e.g. UINT8 GEMM kernels). `tqt_get_dst_bias_offset_*` is
additionally exposed only by the dequantized `clamp_f32` output variants — the raw `i32`
accumulator variants have no destination-side bias, so they do not declare it.

---

### Size Queries

```c
size_t tqt_get_d_size_<KERNEL_FULL>(size_t m, size_t n);
```

Returns the total number of bytes required to store the output matrix D of dimensions
`m x n`.

```c
// Packed variants only
size_t tqt_get_a_packed_size_<KERNEL_FULL>(size_t m, size_t k);
size_t tqt_get_b_packed_size_<KERNEL_FULL>(size_t n, size_t k);
```

Returns the total number of bytes required for packed A or packed B buffers.

---

## Run Functions

- [GEMM with Bias and Clamp](#gemm-with-bias-and-clamp)
- [W16A32 GEMM Contract](#w16a32-gemm-contract)
- [GEMM with Quantization Parameters (Dequant Path)](#gemm-with-quantization-parameters-dequant-path)
- [RVV Quantized-Activation Block GEMM](#rvv-quantized-activation-block-gemm)
- [BLAS-style sgemm / dgemm](#blas-style-sgemm--dgemm)
- [Convolution (Channel-First / Channel-Last)](#convolution-channel-first--channel-last)
- [Depthwise Convolution (Generic)](#depthwise-convolution-generic)
- [Elementwise Operations](#elementwise-operations)
- [Reduction Sum / Mean](#reduction-sum--mean)
- [Activation (ReLU / GeLU / SiLU / ReGLU / GeGLU / SwiGLU)](#activation-relu--gelu--silu--reglu--geglu--swiglu)

The compute entry point performs the actual micro-kernel operation on a single tile.
Its signature varies by operator type.

### GEMM with Bias and Clamp

Computes: `D = clamp(C + A * B + bias, clamp_min, clamp_max)`

```c
void tqt_run_gemm_<KERNEL_BASE>_<tile>_<arch>(
    size_t m, size_t n, size_t k,
    const void *A, size_t lda, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b,
    const void *C, size_t ldc,
    void *D, size_t ldd,
    const void *bias,
    <dtype> clamp_min, <dtype> clamp_max);
```

Parameters:
- `m`, `n`, `k`: actual tile dimensions (may be smaller than m_step/n_step at edges)
- `A`: pointer selected for this call's logical A origin (row-major or packed)
- `lda`: leading dimension of the full A allocation (column count for non-packed)
- `k_idx_a`: K-offset index for split-K accumulation (0 for single pass)
- `B`: pointer selected for this call's logical B origin
- `ldb`: leading dimension of the full B allocation
- `k_idx_b`: K-offset index for split-K accumulation (0 for single pass)
- `C`: pointer to residual matrix (may be nullptr)
- `ldc`: leading dimension of C
- `D`: pointer to output sub-block
- `ldd`: leading dimension of D
- `bias`: pointer to bias vector (may be nullptr)
- `clamp_min`, `clamp_max`: output clamping bounds (type matches output dtype)

Floating-point GEMM, IGEMM, and DWCONV skip the external min/max only when the original public arguments
are exactly `clamp_min == -INFINITY` and `clamp_max == INFINITY`. This preserves NaNs
that reach the clamp stage. Finite intervals, one-sided infinite intervals, and the
largest finite bounds of FP32/FP16/BF16 retain their existing clamp behavior.
NaN payloads, signaling state, signs, and exception flags are not preservation guarantees.
NaN bounds and inverted intervals gain no new contract.

The decision uses the public argument precision before any internal conversion. The
existing calculation, scale, bias, output conversion and store order is unchanged.
For example, the two RVV `gemm_f16_i4` variants with FP32 output first narrow their
float bounds to FP16; passing `[-FLT_MAX, FLT_MAX]` still executes that FP16 clamp,
even though those converted bounds become infinite. The RVV FP16 i8-dequant path
narrows the result before FP16 bias and clamp. Neither precision sequence is changed.

Floating GEMM-backed convolution, floating RVV implicit GEMM, and the 22 RVV
depthwise convolution variants follow this rule too.
Integer clamp interfaces and GEMM/BLAS interfaces without clamp are unaffected.
IGEMM still applies bias on the first K segment and external clamp only on the last.
For interfaces that already support `k == 0`, skipping clamp still preserves their
C, bias, conversion and store operations; other interfaces gain no zero-K support.

`k_idx_a` and `k_idx_b` are absolute logical element indices and may be
different when A and B originate from independently sliced allocations. The
data pointers must already denote those origins (use the matching offset getter
for packed operands); kernels use the indices only for origin-relative packed
addressing and K-indexed metadata. Do not add the K offset to a raw pointer a
second time inside the kernel.

Raw row-major/transposed operands may start at any element. A packed origin must
start at a complete packing K group and satisfy the variant's getter/packer
preconditions. Quantized variants impose the additional K-block alignment
rules described below.

### W16A32 GEMM Contract

The `gemm_f32_f16` and `gemm_f32_bf16` families expose an FP32 public activation
and output contract:

- A is row-major FP32 before packing.
- B is FP16 or BF16, either transposed (`*t`) or packed (`*p`).
- C, bias, clamp bounds, and D are FP32.
- Accumulation is FP32.

RVV kernels load A as FP32 and widen B during the multiply.

For `k == 0`, A and B may be null. The epilogue still applies C and bias, conditionally
clamps under the floating GEMM rule above, and stores D.
For split-K, advance packed A and B with the matching
`get_*_packed_offset` getter and pass the same non-zero `k_idx` to `run_gemm`.
The pointer passed to `run_gemm` denotes the selected call origin; internal tiling
uses offsets relative to that origin.

### GEMM with Quantization Parameters (Dequant Path)

Computes: `D = clamp(dequant(A * B) + bias, clamp_min, clamp_max)`

```c
void tqt_run_gemm_<KERNEL_BASE>_<tile>_<arch>(
    size_t m, size_t n, size_t k,
    const void *A, size_t lda, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b,
    const void *C, size_t ldc,
    void *D, size_t ldd,
    const void *bias,
    float clamp_min, float clamp_max,
    const struct tqt_qai8dxp_qsi8cxp_params *params);
```

The `params` struct carries quantization scales, zero points, and block lengths
required for the dequantization epilogue.

For the Q8_0-activation execution kernels, A codes are signed I8 and
`scale_a` is always `[M,K/32]`. The B token fixes both code packing and
weight-scale shape:

| B token | Canonical input to `run_bt_pack` | `scale_b` shape |
|---------|-----------------------------------|-----------------|
| `qsb1c128` | B1, LSB-first K order | `[N,K/128]` |
| `qau2z1c64` | four adjacent U2 codes per byte, logical value `code-1` | `[N,K/64]` |
| `qsi4c32` | two adjacent signed I4 values per byte | `[N,K/32]` |
| `qsi5c32` | 16 low4 bytes plus four sign-plane bytes per 32 values | `[N,K/32]` |
| `qsi8c32` | adjacent signed I8 values | `[N,K/32]` |

All `k`, `k_idx`, `lda`, and `ldb` arguments remain logical element counts;
packed size and offset getters return bytes. Split-K calls must use matching
non-zero `k_idx_a == k_idx_b`, advance both packed pointers with the selected
variant's getters, and keep segment lengths 32-aligned. The compute entry point
does not parse raw GGML blocks or quantize floating-point activations.

Compute and pack entry points use different leading dimensions. `run_gemm`
receives packed-tensor strides and enforces:

| B token and compute path | Packed-A `lda` | Packed-B `ldb` |
|--------------------------|----------------|----------------|
| `qsb1c128` | multiple of 32 | multiple of 128 |
| `qau2z1c64` | multiple of 32 | multiple of 64 |
| `qsi4c32` | multiple of 32 | multiple of 32 |
| `qsi5c32` | multiple of 32 | multiple of 32 |
| `qsi8c32` RVV | multiple of 32 | no additional multiple |

For every Qx_0 pack call, keep `k` and `k_idx` 32-aligned. RVV `run_a_pack`
adds no divisibility rule for its source `lda` or destination `lda_packed`. `run_bt_pack`
uses these minimum code-stride divisors:

| B token | Source `ldb` | Destination `ldb_packed` |
|---------|--------------|--------------------------|
| `qsb1c128` | multiple of 8 | multiple of 8 |
| `qau2z1c64` | multiple of 4 | multiple of 4 |
| `qsi4c32` | multiple of 2 | multiple of 2 |
| `qsi5c32` | multiple of 32 | multiple of 32 |
| `qsi8c32` | no additional multiple | no additional multiple |

When a pack destination is passed to `run_gemm`, its `lda_packed` or
`ldb_packed` becomes the compute `lda` or `ldb` and must also satisfy the
stronger compute table. The low-bit RVV wrappers state these divisor
preconditions with `TQT_ASSUME` before converting logical element counts to byte
strides. See
[Quantization](../guide/quantization.md) for the execution-format and
scale-granularity details.

### RVV Quantized-Activation Block GEMM

These RVV variants pack only code tensors. Quantization metadata is
passed as row-major arrays and remains addressable with its public leading
dimensions:

```c
struct tqt_qsi8d256_qau4c32_params {
    const float *scale_a;       // [M, K/256]
    size_t ldsa;
    const int16_t *bsums_a;     // [M, K/16]
    size_t ldbsuma;
    const float *scale_b;       // [N, K/32]
    const float *real_zero_b;   // [N, K/32]
    size_t ldsb;
};

struct tqt_qsi8d256_qsi8c16_params {
    const float *scale_a;       // [M, K/256]
    size_t ldsa;
    const float *scale_b;       // [N, K/16]
    size_t ldsb;
};

struct tqt_qsi8d256_qau5c32_params {
    const float *scale_a;       // [M, K/256]
    size_t ldsa;
    const int16_t *bsums_a;     // [M, K/16]
    size_t ldbsuma;
    const float *scale_b;       // [N, K/32]
    const float *real_zero_b;   // [N, K/32]
    size_t ldsb;
};

struct tqt_qsi8d256_qsi6c16_params {
    const float *scale_a;       // [M, K/256]
    size_t ldsa;
    const int16_t *bsums_a;     // [M, K/16]
    size_t ldbsuma;
    const float *scale_b;       // [N, K/16]
    size_t ldsb;
};

struct tqt_qsi8d32_qsi8c32_params {
    const float *scale_a;       // [M, K/32]
    size_t ldsa;
    const float *scale_b;       // [N, K/32]
    size_t ldsb;
};
```

Q5_K uses the same asymmetric struct shape as Q4_K under the
byte-expanded `tqt_qsi8d256_qau8c32_params` form or the plane-packed
`tqt_qsi8d256_qau5c32_params` form. Each kernel exposes code-buffer
`run_a_pack`/`run_bt_pack`, packed size/offset getters, metadata offset getters,
and the standard f32 GEMM run signature without a runtime block-length argument.
For full-buffer Split-K, advance A/B with their packed offset getters but keep
the parameter structure rooted at the original metadata tensors; the global
`k_idx` selects the corresponding metadata blocks. Q4_K/Q5_K/Q6_K splits are
256-aligned, Q8_0 splits are 32-aligned, and A/B K indices must match.

RVV provides compact and nibble-expanded Q2_K/Q3_K plus the Q4_K/Q5_K/Q6_K/Q8_0
byte-code and plane-packed formats described here.

Ztt adds `_1nlenx1nlen_ztt` variants for the existing Q4_K
`gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p` and expanded Q6_K
`gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p` interfaces. Both getters return
`amenlen`; params and run signatures are unchanged. Code sizes include M/N and
K padding, so they must come from the corresponding variant's getters.
`run_gemm` allows K=0 with null code and params pointers, and empty M/N is a no-op.
C=D requires matching strides. Full-buffer Split-K retains the original
metadata K base and equal, 256-aligned A/B origins. See
[Ztt quantized packing](../guide/packing.md#ztt-quantized-integer-dot-packing).

For RVV plane-packed B, `ldb` and `k_idx` remain logical-weight dimensions.
`qau5c32p` occupies `(K/32)*20` bytes per row and `qsi6c16p` occupies
`(K/16)*12` bytes per row. Their packed tile layout is
`[K group][plane-byte position][N lane]`.

#### Q2_K/Q3_K Factorized Metadata API

Q2_K/Q3_K variants keep group metadata and super-block metadata factorized and
unpacked. Compact and nibble-expanded forms use the same metadata shape, so an
adapter may share those arrays between the two execution encodings.

The Q2_K compact and nibble-expanded params structures are:

```c
struct tqt_qsi8d256_qau2c16_params {
    const float *scale_a;          // [M, K/256]
    size_t ldsa;
    const int16_t *bsums_a;        // [M, K/16]
    size_t ldbsuma;
    const uint8_t *qscale_b;       // [N, K/16]
    const uint8_t *qmin_b;         // [N, K/16]
    size_t ldqb;
    const float *super_scale_b;    // [N, K/256]
    const float *super_min_b;      // [N, K/256]
    size_t ldssb;
};

struct tqt_qsi8d256_qau4c16_params {
    const float *scale_a;          // [M, K/256]
    size_t ldsa;
    const int16_t *bsums_a;        // [M, K/16]
    size_t ldbsuma;
    const uint8_t *qscale_b;       // [N, K/16]
    const uint8_t *qmin_b;         // [N, K/16]
    size_t ldqb;
    const float *super_scale_b;    // [N, K/256]
    const float *super_min_b;      // [N, K/256]
    size_t ldssb;
};
```

The Q3_K compact and nibble-expanded params structures are:

```c
struct tqt_qsi8d256_qsi3c16_params {
    const float *scale_a;          // [M, K/256]
    size_t ldsa;
    const int8_t *qscale_b;        // [N, K/16]
    size_t ldqb;
    const float *super_scale_b;    // [N, K/256]
    size_t ldssb;
};

struct tqt_qsi8d256_qsi4c16_params {
    const float *scale_a;          // [M, K/256]
    size_t ldsa;
    const int8_t *qscale_b;        // [N, K/16]
    size_t ldqb;
    const float *super_scale_b;    // [N, K/256]
    size_t ldssb;
};
```

Compact `qau2c16p`/`qsi3c16p` variants are RVV-only. Nibble-expanded
`qau4c16p`/`qsi4c16p` variants run on RVV. Every variant requires K,
`k_idx_a`, and `k_idx_b` to be multiples of 256, with equal A/B K indices.
For Split-K, advance A/B code pointers with their packed offset getters, but
keep the params structure rooted at the original metadata arrays; the global
K index selects the group and super-block metadata for each split.

### BLAS-style sgemm / dgemm

Computes: `D = alpha * A * B + beta * C`

```c
void tqt_run_sgemm_<KERNEL_BASE>_<tile>_<arch>(
    size_t m, size_t n, size_t k,
    const void *A, size_t lda, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b,
    const void *C, size_t ldc,
    void *D, size_t ldd,
    float alpha, float beta);
```

No bias or clamp parameters. The scalar `alpha` and `beta` replace them. For `dgemm`,
the scalars are `double`.

### Convolution (Channel-First / Channel-Last)

Conv kernels operate on im2col'd data and expose additional im2col-specific functions
(in place of the standard `get_b_offset`):

```c
// Query size of the im2col buffer
size_t tqt_get_b_im2col_size_<KERNEL_FULL>(size_t k, size_t n);

// Compute byte offset into the im2col buffer
size_t tqt_get_b_im2col_offset_<KERNEL_FULL>(size_t n_idx, size_t k_idx, size_t ldb_im2col);

// Run im2col transformation (materializes the im2col buffer)
void tqt_run_im2col_<KERNEL_BASE>_<tile>_<arch>(
    size_t group_idx,
    const void *activation,
    void *im2col_buf,
    const tqt_conv_params *params);
```

The run function follows the same GEMM signature because it operates on the im2col'd
or implicit-GEMM formatted data:

```c
void tqt_run_conv_<KERNEL_BASE>_<tile>_<arch>(
    size_t m, size_t n, size_t k,
    const void *A, size_t lda, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b,
    const void *C, size_t ldc,
    void *D, size_t ldd,
    const void *bias,
    <dtype> clamp_min, <dtype> clamp_max);
```

The spatial parameters are encoded in the `tqt_conv_params` struct (see `src/common/tqt_conv.h`):

```c
struct tqt_conv_params {
    size_t input_shape[3];   // {D, H, W} (set D=1 for 2D)
    size_t output_shape[3];  // {OD, OH, OW} (caller pre-computes)
    size_t kernel_shape[3];  // {KD, KH, KW} (set KD=1 for 2D)
    size_t stride[3];        // {SD, SH, SW}
    size_t dilation[3];      // {DD, DH, DW}
    size_t pad[3];           // {PD, PH, PW}
    size_t input_channel;    // IC
    size_t output_channel;   // OC
    size_t groups;           // Number of groups (1 for non-grouped conv)
};
```

### Depthwise Convolution (Generic)

```c
// Query total output buffer size
size_t tqt_get_output_size_<KERNEL_FULL>(const tqt_conv_params *params);

// Run the depthwise convolution over the full spatial volume
void tqt_run_dwconv_<KERNEL_BASE>_<tile>_<arch>(
    const void *input, const void *weight, const void *bias, void *output,
    const tqt_conv_params *params,
    <dtype> clamp_min, <dtype> clamp_max);
```

Generic depthwise kernels operate on entire spatial volumes via `tqt_conv_params`
which encodes input/output dimensions, kernel size, stride, padding, and dilation.
They do **not** expose `get_m_step` or `get_n_step` — the tiling is handled internally.

Packed specialized depthwise kernels (e.g., `dwconvcf3x3s1` with `f32p_f32p_f32p`) additionally expose:
- `tqt_get_oh_step_<KERNEL_FULL>()` — output-height step for OH-dimension tiling
- `tqt_get_output_size_<KERNEL_FULL>(const tqt_conv_params *params)` — output buffer size
- `tqt_get_packed_input_size_<KERNEL_FULL>(const tqt_conv_params *params)` — packed input buffer size
- `tqt_get_packed_weight_size_<KERNEL_FULL>(const tqt_conv_params *params)` — packed weight buffer size
- `tqt_run_pack_input_<KERNEL_FULL>(...)`, `tqt_run_pack_weight_<KERNEL_FULL>(...)` — pack helpers
- `tqt_run_unpack_output_<KERNEL_FULL>(...)` — unpack output back to standard layout

### Elementwise Operations

Elementwise kernels operate on contiguous 1-D floating-point vectors of length `n`.
`<op>vv` combines two vectors; `<op>vs` combines a vector with a scalar. There is no
bias, clamp, C residual, or leading dimension.

```c
// <op>vv: c[i] = a[i] <op> b[i], where <op> is add/sub/mul/div
void tqt_run_<op>vv_f32_f32_f32_<tile>_<arch>(
    size_t n, const void *a, const void *b, void *c);

// <op>vs: c[i] = a[i] <op> b, where b is a scalar
void tqt_run_<op>vs_f32_f32_f32_<tile>_<arch>(
    size_t n, const void *a, float b, void *c);
```

- `n`: number of elements in each vector
- `a`: pointer to input vector A `[n]`
- `b`: pointer to input vector B `[n]` (`*vv`), or the scalar operand (`*vs`)
- `c`: pointer to output vector C `[n]`

The available operations and scalar operand types are:

| Data type | RVV operations | Scalar operand type |
|-----------|----------------|---------------------|
| `f32` | `{add,sub,mul,div}{vv,vs}` | `float` |
| `f16` | `{add,sub,mul,div}{vv,vs}` | `float16_t` |

### Reduction Sum / Mean

Reduction kernels reduce a row-major `[m, n]` matrix `a` (leading dimension `lda`) along
one axis. `sumrow` / `meanrow` produce one value per row (`c[m]`); `sumcol` / `meancol`
produce one value per column (`c[n]`). The optional `init` vector seeds the sum outputs;
for mean outputs it is added as an output bias after the reduced sum is scaled.
Pass `nullptr` for zero initialization.

```c
// sumrow: c[i] = (init ? init[i] : 0) + Σ_j a[i*lda + j]   -> c has length m
void tqt_run_sumrow_f32_f32_<tile>_<arch>(
    size_t m, size_t n, const void *a, size_t lda, void *c, const void *init);

// sumcol: c[j] = (init ? init[j] : 0) + Σ_i a[i*lda + j]   -> c has length n
void tqt_run_sumcol_f32_f32_<tile>_<arch>(
    size_t m, size_t n, const void *a, size_t lda, void *c, const void *init);

// meanrow: c[i] = (init ? init[i] : 0) + (1/n) * Σ_j a[i*lda + j]
void tqt_run_meanrow_f32_f32_<tile>_<arch>(
    size_t m, size_t n, const void *a, size_t lda, void *c, const void *init);

// meancol: c[j] = (init ? init[j] : 0) + (1/m) * Σ_i a[i*lda + j]
void tqt_run_meancol_f32_f32_<tile>_<arch>(
    size_t m, size_t n, const void *a, size_t lda, void *c, const void *init);
```

- `m`, `n`: matrix dimensions (rows, columns)
- `a`: pointer to the row-major input matrix `[m, n]`
- `lda`: leading dimension of A (elements per row)
- `c`: pointer to the output vector (`[m]` for `*row`, `[n]` for `*col`)
- `init`: optional per-output initial values (same length as `c`), or `nullptr` for zero-init

All four reduction operations are available for `f32` and `f16` on RVV.
Input, output, and optional `init` buffers use the data type encoded in the kernel name.

#### FP32 numerical behavior

The current RVV `f32` sum/mean kernels use FP32 accumulation. The formulas
above describe the mathematical operations, not a fixed floating-point evaluation
order. These kernels prioritize throughput and do not perform overflow checks,
compensated summation, or higher-precision recomputation.

Mean kernels accumulate the sum in FP32 before multiplying by the FP32 reciprocal
of the reduction length. Intermediate sums can overflow even when the exact mean
is finite; scaling an infinite sum cannot recover the finite result. For example,
with round-to-nearest, ties-to-even and zero `init`, summing two `FLT_MAX` values
in FP32 produces positive infinity before the mean scaling, although their exact
mean is `FLT_MAX`. Callers must account for the range of intermediate partial sums,
not only the final mathematical result.

Rounding can also discard small contributions before larger terms cancel. For
example, sequential FP32 evaluation of `(2^24 + 1) - 2^24` with round-to-nearest,
ties-to-even produces zero, whereas the exact sum is one. This illustrates a
rounding limitation, not a prescribed result for every kernel: different grouping
or reduction orders can retain the small term. Results need not match an FP64
reference rounded to FP32, and bitwise reproducibility across VLEN values or
implementations is not guaranteed.

These range and precision limits are accepted tradeoffs of the default
high-performance kernels. Callers requiring stronger guarantees for extreme
magnitudes, strong cancellation, or reproducibility should select an implementation
with the required numerical properties.

### Activation (ReLU / GeLU / SiLU / ReGLU / GeGLU / SwiGLU)

Activation kernels map a row-major `[m, n]` matrix elementwise. Unlike elementwise and reduction,
every operand carries **its own leading dimension**, so inputs and output may live in differently
strided buffers:

```c
// relu: c[i][j] = max(a[i][j], 0)
void tqt_run_relu_f32_f32_<tile>_<arch>(
    size_t m, size_t n, const void *a, size_t lda, void *c, size_t ldc);

// gelu: standard tanh approximation
void tqt_run_gelu_f32_f32_<tile>_<arch>(
    size_t m, size_t n, const void *a, size_t lda, void *c, size_t ldc);

// silu: c[i][j] = a[i][j] * sigmoid(a[i][j])
void tqt_run_silu_f32_f32_<tile>_<arch>(
    size_t m, size_t n, const void *a, size_t lda, void *c, size_t ldc);

// reglu: c[i][j] = relu(a[i][j]) * b[i][j]
void tqt_run_reglu_f32_f32_f32_<tile>_<arch>(
    size_t m, size_t n, const void *a, size_t lda, const void *b, size_t ldb,
    void *c, size_t ldc);

// geglu: c[i][j] = gelu(a[i][j]) * b[i][j], using tanh-approximate gelu
void tqt_run_geglu_f32_f32_f32_<tile>_<arch>(
    size_t m, size_t n, const void *a, size_t lda, const void *b, size_t ldb,
    void *c, size_t ldc);

// swiglu: c[i][j] = silu(a[i][j]) * b[i][j]   -- a is the gate, b the linear operand
void tqt_run_swiglu_f32_f32_f32_<tile>_<arch>(
    size_t m, size_t n, const void *a, size_t lda, const void *b, size_t ldb,
    void *c, size_t ldc);
```

- `m`, `n`: matrix dimensions (rows, columns)
- `a`: pointer to the row-major input matrix `[m, n]` (the gate for gated activations)
- `lda`: leading dimension of A (elements per row, `>= n`)
- `b`: pointer to the row-major linear operand `[m, n]` (gated activations only)
- `ldb`: leading dimension of B (elements per row, `>= n`) (gated activations only)
- `c`: pointer to the row-major output matrix `[m, n]`
- `ldc`: leading dimension of C (elements per row, `>= n`)

`c` may alias any input (`ldc == lda`, or `ldc == ldb` for a gated activation) for an in-place
call — the operation is element-wise with no cross-element dependency, and both
operands of a tile are read before the tile is written. Only the `[m, n]` window is
written; columns `[n, ldc)` of each row are left untouched.

Because every gated activation takes `ldb` independently of `lda`, the fused `[M, 2N]` gate-and-up
buffer a transformer feed-forward block produces needs no repacking — pass
`a = base, lda = 2 * n` and `b = base + n, ldb = 2 * n`.

`relu`, `gelu`, `silu`, `reglu`, `geglu`, and `swiglu` are available for `f32`
and `f16` on RVV. Public `gelu` uses the standard tanh approximation
`0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`. GeLU/GeGLU use the shared
rational `tanh`; SiLU/SwiGLU use the shared FMA-based `exp`. Both helpers live
in `src/common/tqt_math_rvv.h`, with FP16 arithmetic widened to FP32. Every `gelu`, `geglu`, `silu`, and `swiglu` variant
is an approximation with an unbounded relative error in the saturating tails, so
verify its result against an absolute tolerance. The matching tests encode the accepted
tolerance; some per-variant headers additionally state measured bounds and ranges. For
gated approximate activations, absolute error is the gate-function error scaled by `|b|`, so it grows with the
magnitude of the linear operand and not only with `|a|`. The numerical budgets cover finite
inputs in the tested activation range. RVV SiLU/SwiGLU require RNE rounding and propagate NaNs;
the SiLU gate preserves signed zeros, maps `+Inf` to `+Inf`, and maps `-Inf` to NaN.
The `-Inf` result follows native PyTorch/GGML forward evaluation of `-Inf / +Inf`.
SwiGLU then multiplies that gate by the linear operand using ordinary floating-point semantics,
including `0 * Inf = NaN`. Their exp helper preserves subnormal results and overflows to `+Inf`;
sufficiently negative finite gates therefore return zero when the denominator overflows.
Other activation variants retain their existing special-value contracts. See the per-variant
headers for the applicable ranges and assumptions.

The current RVV ReLU kernels use native maximum instructions for throughput:
NaN and -0 map to +0, +Inf is preserved, and -Inf maps to +0. This differs from
PyTorch's NaN-propagating ReLU. ReGLU uses the same ReLU gate behavior before multiplication.

---

## Packing Functions (Packed Variants Only)

When both A and B carry the `p` suffix, the kernel expects pre-packed inputs and
exposes packing routines:

```c
void tqt_run_a_pack_<KERNEL_BASE>_<tile>_<arch>(
    size_t m, size_t k, size_t lda, size_t lda_packed,
    size_t k_idx, const void *A, void *A_packed);

void tqt_run_b_pack_<KERNEL_BASE>_<tile>_<arch>(
    size_t n, size_t k, size_t ldb, size_t ldb_packed,
    size_t k_idx, const void *B, void *B_packed);

// Pack from transposed B into packed format
void tqt_run_bt_pack_<KERNEL_BASE>_<tile>_<arch>(
    size_t n, size_t k, size_t ldb, size_t ldb_packed,
    size_t k_idx, const void *B, void *B_packed);
```

Packed storage is variant-specific. In particular, a packed W16A32 A panel uses
FP32 elements for RVV. Always allocate with the selected variant's
`get_*_packed_size` getters and obtain sub-buffer pointers with its
`get_*_packed_offset` getters. Do not derive storage width from the public FP32 A
type or from the `f32p` name alone.

Packed buffers are tied to the producing kernel's ISA, tile, and runtime vector or
matrix register length. Packing and execution must use the same variant and target
configuration.

---

<a id="ztt-f32-gemm"></a>

## Ztt GEMM

The `1nlenxpnlen_ztt` variants implement `D = clamp(C + A*B + 1xN_bias)` for
F32, F16 and BF16, with ordinary B, transposed B and packed A/B/C/D interfaces.
F16/BF16 also provide `1nlenxpnlen_f32acc_ztt`. Each variant reuses its kernel
base's function table; callers do not select a separate public API for each UDS.
UDS specialization is compiled into the implementation. A binary requires a
matching runtime UDS; it is not a runtime dispatcher between incompatible ISAs.

Let `S = amenlen`, `P = pack_factor = max(1, UDS / input_bits)`.
The [tile naming definition](naming_convention.md#ztt-pack-factor) explains the
register packing factor and its distinction from packed memory layouts. The getters return
`m_step = S`, `n_step = P*S`. Ordinary layouts require M aligned to S and N/K
aligned to P*S, with independent element strides. `k_idx_a/k_idx_b` are unused;
pass pointers advanced by the byte-offset getters. Packed layouts accept arbitrary
logical dimensions and use the matching [packers](../guide/packing.md#ztt-square-packing).

| Input/output T | UDS8 | UDS16 | UDS32 | UDS64 | UDS128 |
|---|---|---|---|---|---|
| F16/BF16 P | 1 | 1 | 2 | 4 | 8 (A8/A16 required) |
| F32 P | 1 | 1 | 1 | 2 | 4 (A4 sufficient) |

The implementation uses M16 or M32 and A4/A8/A16 profiles, subject to the P
requirement above. Both F16/BF16 precision strategies require A8 at UDS128;
there is no A4 batching fallback. F32 uses the single capability macro
`TORQ_TILE_HAS_ZTT_GEMM_F32`, derived from compilation of this implementation.

Native kernels accumulate and add bias in T. `f32acc` widens C to F32,
accumulates and adds bias in F32, then narrows to T using round-to-nearest-even.
Clamping follows that narrowing and uses `fminf(fmaxf(value,min),max)`; only
exact `[-INFINITY,INFINITY]` bypasses it. Thus the two precision strategies
can produce different results even though their buffer types and packing match.
Hosted builds use scalar libm min/max. Freestanding builds use scalar RISC-V
`fmin.s`/`fmax.s`, or an IEEE binary32 bitwise fallback when scalar F is absent;
one NaN yields the numeric operand and signed zeros retain min/max ordering.
The same path serves all three input families, after any output narrowing.
Memory operations use the platform's `TQT_MEMCPY`/`TQT_MEMSET` contract.
Geometry reads use the common CSR getters: hosted builds cache the first result
per translation unit, while freestanding builds read on demand. Packing and
execution must use the same `amenlen` and UDS.

C/bias may be null; exact C=D is allowed with identical layouts and strides.
D must not overlap A/B/bias. Empty M/N is a no-op, and K=0 never accesses A/B.
Size arithmetic is checked and invalid preconditions fail in release builds too.
Ownership is acquired only when needed and only that acquisition is released.
Acquisition failure or an unsupported operation (`amestatus.UN`, including a
pre-existing UN status) fails explicitly without clearing the caller's sticky
status. M/ACC state is call-clobbered; callers must not retain live matrix values across GEMM calls.

The scalar-only noinline compute boundary isolates fixed-register assembly from
compiler-managed M/ACC values. UDS specialization fixes the register groups and
unrolls the P output squares at compile time. The kernels allocate no scratch
matrices and make no calls while matrix values are live. This isolation is needed
because the current compiler does not accept M/ACC inline-assembly clobber names.

Size and packing helpers require capability CSR access but do not acquire AME
ownership. No ownership state is cached. The platform must enable AME state and
provide the operation tuples and register profile used by the binary.

### Packed F32 interface example

The packed-output base `gemm_1xnbias_clamp_f32p_f32p_f32p` has its own function
table. In the signature templates below, `FULL` means that base followed by
`_1nlenxpnlen_ztt`. Sizes are bytes and pack/unpack strides count elements.

```c
size_t tqt_get_a_packed_size_FULL(size_t m, size_t k);
size_t tqt_get_b_packed_size_FULL(size_t n, size_t k);
size_t tqt_get_d_packed_size_FULL(size_t m, size_t n);  // Also the C allocation size.
size_t tqt_get_bias_packed_size_FULL(size_t n);
void tqt_run_a_pack_FULL(size_t m, size_t k, size_t lda, const void *A, void *Ap);
void tqt_run_b_pack_FULL(size_t n, size_t k, size_t ldb, const void *B, void *Bp);
void tqt_run_bt_pack_FULL(size_t n, size_t k, size_t ldb, const void *Bt, void *Bp);
void tqt_run_c_pack_FULL(size_t m, size_t n, size_t ldc, const void *C, void *Cp);
void tqt_run_bias_pack_FULL(size_t n, const void *bias, void *bias_padded);
void tqt_run_d_unpack_FULL(size_t m, size_t n, size_t ldd, const void *Dp, void *D);
void tqt_run_FULL(size_t m, size_t n, size_t k, const void *Ap, const void *Bp,
                  const void *Cp, void *Dp, const void *bias_padded,
                  float clamp_min, float clamp_max);
```

The matching F16/BF16 interfaces follow their own public types and accumulation
strategy. Use the selected header and the [packing guide](../guide/packing.md#ztt-square-packing)
for buffer sizing and padding.

---

## Interface Header and Runtime Dispatch

Each kernel base defines a `_interface.h` header containing:
1. `typedef` for every function pointer type
2. A `struct tqt_<KERNEL_BASE>_ukernel` aggregating all function pointers

This enables runtime dispatch across tile/arch variants:

```c
// Example: tqt_gemm_1xnbias_clamp_f16_f16_f16t_interface.h

struct tqt_gemm_1xnbias_clamp_f16_f16_f16t_ukernel {
    tqt_gemm_1xnbias_clamp_f16_f16_f16t_get_m_step_func_t   get_m_step;
    tqt_gemm_1xnbias_clamp_f16_f16_f16t_get_n_step_func_t   get_n_step;
    tqt_gemm_1xnbias_clamp_f16_f16_f16t_get_a_offset_func_t get_a_offset;
    tqt_gemm_1xnbias_clamp_f16_f16_f16t_get_b_offset_func_t get_b_offset;
    tqt_gemm_1xnbias_clamp_f16_f16_f16t_get_c_offset_func_t get_c_offset;
    tqt_gemm_1xnbias_clamp_f16_f16_f16t_get_bias_offset_func_t get_bias_offset;
    tqt_gemm_1xnbias_clamp_f16_f16_f16t_get_d_offset_func_t get_d_offset;
    tqt_gemm_1xnbias_clamp_f16_f16_f16t_get_d_size_func_t   get_d_size;
    tqt_gemm_1xnbias_clamp_f16_f16_f16t_run_gemm_func_t     run_gemm;
};
```

To select a variant at runtime, populate the struct with function pointers from
the desired tile/arch variant header (e.g., `tqt_get_m_step_..._8x3vl_rvv`).

---

## Canonical Tiling Loop

The caller is responsible for the outer tiling loop. Here is the canonical pattern:

```c
const size_t m_step = ukernel.get_m_step();
const size_t n_step = ukernel.get_n_step();

for (size_t m_idx = 0; m_idx < M; m_idx += m_step) {
    for (size_t n_idx = 0; n_idx < N; n_idx += n_step) {
        size_t actual_m = min(M - m_idx, m_step);
        size_t actual_n = min(N - n_idx, n_step);

        const uint8_t *a_ptr   = (const uint8_t *)A + ukernel.get_a_offset(m_idx, 0, lda);
        const uint8_t *b_ptr   = (const uint8_t *)B + ukernel.get_b_offset(n_idx, 0, ldb);
        const uint8_t *c_ptr   = (const uint8_t *)C + ukernel.get_c_offset(m_idx, n_idx, ldc);
        const uint8_t *bias_ptr = (const uint8_t *)bias + ukernel.get_bias_offset(n_idx);
        uint8_t *d_ptr         = (uint8_t *)D + ukernel.get_d_offset(m_idx, n_idx, ldd);

        ukernel.run_gemm(actual_m, actual_n, K,
                         a_ptr, lda, 0, b_ptr, ldb, 0,
                         c_ptr, ldc, d_ptr, ldd,
                         bias_ptr, clamp_min, clamp_max);
    }
}
```

Key observations:
- The kernel handles edge tiles when `actual_m < m_step` or `actual_n < n_step`.
- `k_idx_a` / `k_idx_b` are for split-K accumulation; pass 0 for single-pass.
- Offset helpers return **byte offsets** -- use `uint8_t *` for pointer arithmetic.
- Leading dimensions (`lda`, `ldb`, etc.) are in **elements**, not bytes.
- Packed variants require calling `run_a_pack` / `run_b_pack` before `run_gemm`.
- All functions are thread-safe on disjoint tile regions; no allocation, no global state.
- Always refer to the specific `.h` header for the authoritative function signature.
