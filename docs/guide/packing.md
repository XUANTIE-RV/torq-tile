<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Packing Guide

This document explains the packing system in TORQ-Tile: what it is, why it
matters, and how to use it end-to-end.

On this page:

- [What Is Packing?](#what-is-packing)
- [Three Input Modes](#three-input-modes)
- [Pack Helper Locations](#pack-helper-locations)
- [Public Pack API Per Kernel](#public-pack-api-per-kernel)
- [Typical Workflow](#typical-workflow)
- [Complete code example](#complete-code-example)
- [Mixed Input Combinations](#mixed-input-combinations)
- [W16A32 Packed Activations](#w16a32-packed-activations)
- [Pack Layout Details](#pack-layout-details)
- [Packing for Convolution](#packing-for-convolution)
- [Ztt Square Packing](#ztt-square-packing)
- [Performance Considerations](#performance-considerations)
- [See Also](#see-also)

## What Is Packing?

GEMM micro-kernels process data in small tiles (e.g. 8 rows x 3*vl columns).
**Packing** rearranges a source matrix into the layout consumed by a selected
kernel. It can improve data reuse or simplify loads, but adds preparation cost
and storage. Ordinary and transposed layouts may already match the workload well.

Common RVV A packers interleave rows into `[k, MR]` strips. Other operands,
datatypes and backends use different layouts; always pair a public packer with
its matching compute variant.

## Three Input Modes

Every TORQ-Tile GEMM kernel variant encodes the expected input format of A and
B via a suffix on the data-type slot:

| Suffix | Name       | Storage layout expected by the kernel               |
|--------|------------|-----------------------------------------------------|
| (none) | Plain      | Standard row-major `[rows, cols]`                   |
| `t`    | Transposed | Row-major of the transposed matrix `[cols, rows]`   |
| `p`    | Packed     | Tile-interleaved layout produced by the pack helper |

### When to use each mode

- **Plain:** use the original row-major data without preprocessing.
- **Transposed:** use existing transposed weights without converting them.
- **Packed:** compare packing plus execution against ordinary layouts. Fixed
  weights can reuse packed contents; changing activations must be repacked.

A packed format may permit aligned subwindows or require complete-matrix calls.
Do not assume arbitrary M/N tile reuse from the `p` suffix alone.

## Pack Helper Locations

| File | Scope |
|------|-------|
| [`src/common/tqt_gemm_pack_rvv.h`](../../src/common/tqt_gemm_pack_rvv.h) | GEMM pack functions for RVV (f32, f16, i8, i4, u8) |
| [`src/common/tqt_dwconv_pack_rvv.h`](../../src/common/tqt_dwconv_pack_rvv.h) | DWConv channel-first/channel-last pack/unpack |

These are internal helpers. Each packed kernel variant exposes its own
public pack API functions (see next section).

## Public Pack API Per Kernel

The following signature templates describe common floating RVV APIs with
both operands packed. A one-sided packed API exposes only its required packers.
Quantized and Ztt APIs have different metadata or parameters; consult their
headers and the Ztt sections below.

```c
// Query required buffer sizes (in bytes)
size_t tqt_get_a_packed_size_<KERNEL_FULL>(size_t m, size_t k);
size_t tqt_get_b_packed_size_<KERNEL_FULL>(size_t n, size_t k);

// Pack A from row-major [m, k] into the tile-interleaved layout
void tqt_run_a_pack_<KERNEL_FULL>(size_t m, size_t k, size_t lda,
                                  size_t lda_packed, size_t k_idx,
                                  const void *A, void *A_packed);

// Pack B from row-major [k, n] (non-transposed B) into the packed layout
void tqt_run_b_pack_<KERNEL_FULL>(size_t n, size_t k, size_t ldb,
                                  size_t ldb_packed, size_t k_idx,
                                  const void *B, void *B_packed);

// Pack B from [n, k] (transposed B) into the packed layout
void tqt_run_bt_pack_<KERNEL_FULL>(size_t n, size_t k, size_t ldb,
                                   size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
```

The `lda_packed` / `ldb_packed` parameters are typically set to `k` (the
reduction dimension), matching the full-K packing case. The `k_idx` parameter
is `0` for full-K packing and non-zero only when K-tiling is used (advanced).

## Typical Workflow

```
1. Query sizes      -->  tqt_get_a_packed_size / tqt_get_b_packed_size
2. Allocate         -->  malloc(a_packed_size), malloc(b_packed_size)
3. Pack A           -->  tqt_run_a_pack_*(M, K, lda=K, lda_packed=K, k_idx=0, A, A_packed)
4. Pack B           -->  tqt_run_bt_pack_*(N, K, ldb=K, ldb_packed=K, k_idx=0, B, B_packed)
5. Run kernel       -->  tqt_run_gemm_*(...)   (uses A_packed, B_packed)
6. (Optional) Reuse -->  Change M/N tile, call run again with same packed buffers
7. Free             -->  free(A_packed), free(B_packed)
```

Packed weights (B) are typically packed once during model initialization and
reused for every inference call. Packed activations (A) must be repacked whenever their contents change.

## Complete code example

The [packed GEMM scenario](../../examples/api/gemm/packed_gemm/README.md) provides
one complete program for B packing, BT packing, fixed-weight reuse and Ztt packed
output. It binds the selected variant's function table, queries byte sizes,
allocates buffers, packs, computes and checks against an ordinary-layout reference.

In `packed_gemm.cpp`, follow the `run_b_pack` and `run_bt_pack` calls: B[K,N] and
Bt[N,K] produce the same packed representation for that variant. The two-iteration
activation loop reuses B while repacking the changed A. Ztt's packed-output flow
uses a separate interface and shows C/bias packing and D unpacking. The example
uses FP32; FP16 and quantized paths require their own declared storage and metadata
types.

## Mixed Input Combinations

Not all combinations require both A and B to be packed. The library provides
variants for common mixed scenarios:

| Kernel name pattern | A input | B input | Notes |
|---------------------|---------|---------|-------|
| `*_f16_f16t_*`      | Plain   | Transposed | Zero-copy B, no pack needed |
| `*_f16_f16p_*`      | Plain   | Packed     | Only B is packed (common for static weights) |
| `*_f16p_f16p_*`     | Packed  | Packed     | Both operands use the selected packed layout |

The `run_b_pack` function packs from a non-transposed B `[k, n]`, while
`run_bt_pack` packs from a transposed B `[n, k]`. Choose whichever matches
your source layout.

## W16A32 Packed Activations

W16A32 kernels have public FP32 activations and outputs with FP16 or BF16
weights. RVV packs A as FP32 and widens the weights during FP32 accumulation.
Allocate packed buffers with the selected variant's size getters and obtain
split-K and tile origins from its offset getters. Packed buffers must match
the chosen variant, tile and VLEN; the `p` suffix alone does not establish
layout compatibility.

For split-K, pack a slice with its logical `k_idx`, pass a pointer obtained with
the corresponding packed-offset getter, and pass the same `k_idx` to `run_gemm`.
For `k == 0`, the compute API permits null A and B pointers because no packed data
is consumed.

## Pack Layout Details

### A packing: `[m, k]` -> `[m/MR, k, MR]`

The A matrix is split into horizontal strips of `MR` rows. Within each strip,
data is interleaved so that the `MR` elements at the same K position are
contiguous. For an 8x3vl tile (MR=8), the packed format uses `vsseg8e`
instructions to store 8 consecutive rows in one shot.

For tail rows where `actual_m < MR`, a smaller interleave factor is used
(e.g., `vsseg4e` for 4 remaining rows), so no zero-padding is required.

### B packing: `[k, n]` or `[n, k]` -> `[n/(NR*vl), k, NR*vl]`

The B matrix is split into vertical strips of `NR * vlmax` columns. Within
each strip, data is arranged so that one row of `NR * vlmax` elements is
contiguous. For transposed B `[n, k]`, gather operations (`vlse`) collect
strided elements into the contiguous layout.

## Packing for Convolution

### Channel-First DWConv: `NCDHW` -> `NC1DHWC0`

Depthwise convolution kernels using the packed layout re-tile the input from
standard channel-first `[C, D, H, W]` into a blocked format
`[C1, D, H, W, C0]` where `C0 = vlmax` for the given dtype. This ensures each
vector load covers a full `C0` slice of channels, eliminating strided access.

Helper functions in [`src/common/tqt_dwconv_pack_rvv.h`](../../src/common/tqt_dwconv_pack_rvv.h):

```c
// Query the packed tensor size (bytes, single batch)
size_t tqt_dwconv_packed_tensor_bytes_f32_rvv(size_t channel, size_t spatial);

// Query the packed weight size (bytes)
size_t tqt_dwconv_packed_weight_bytes_f32_rvv(size_t channel, size_t kernel_size);

// Pack channel-first input: [IC, D, H, W] -> [C1, D, H, W, C0]
void tqt_dwconv_pack_input_cf_f32_rvv(const float *input, float *packed,
                                      const tqt_conv_params *params);
```

The tail block (`c1 = C1-1`) is compact -- it holds only `tail_vl` elements
with no zero-padding, so total element count equals `IC * spatial`.

### Channel-Last Conv/DWConv

Channel-last kernels (`convcl`, `dwconvcl`) typically work with mixed-packed
variants where the weight (B) is packed but the activation (A) remains in
`[H, W, C]` layout. The im2col transform (if needed) is handled externally;
TORQ-Tile provides the weight-packing and compute kernel.

## Ztt Square Packing

The `1nlenxpnlen_ztt` and `1nlenxpnlen_f32acc_ztt` variants use one format per
public scalar type T. Let `S = get_m_step()` and `P*S = get_n_step()`, where
`P = pack_factor = max(1, UDS / (8*sizeof(T)))`. This is the input
datatype's Ztt register packing factor, not the `p` suffix that marks a packed
memory layout; see the [definition](../reference/naming_convention.md#ztt-pack-factor).
Pad M to S and N/K to P*S. Bias is a
one-dimensional array padded to P*S. Every padding element is zeroed by the
matching packer. Size getters return bytes and pack/unpack strides count elements.

Each square contains S-by-S elements in row-major order. A/C/D squares are
block-row-major, while B squares are block-column-major. For padded dimensions
R/C and logical element (i,j), the indices in units of T are:

```text
A/C/D: ((i/S)*(C/S) + j/S)*S*S + (i%S)*S + j%S
B:     ((j/S)*(R/S) + i/S)*S*S + (i%S)*S + j%S
```

`run_b_pack(N,K,ldb,B,Bp)` reads B[K,N]; `run_bt_pack(N,K,ldb,Bt,Bp)` reads
Bt[N,K]. Both produce the same Bp. C uses the D allocation size. Native and
`f32acc` variants of the same T share all packed buffers, including bias and C.
Call GEMM with logical M/N/K, then unpack D's valid M-by-N region. Output padding
is unspecified after bias/clamp. Packing buffers must not overlap their source.

These buffers require the same T, UDS and S at packing and execution. The common
interface struct does not imply a common packed representation across backends.
The packed API processes complete matrices; it has no packed subwindow or split-K
parameters. See the [Ztt API](../reference/api_reference.md#ztt-gemm) and the
[F32 packed example](../../examples/api/gemm/packed_gemm/README.md).

### Ztt Quantized Integer-Dot Packing

The `qsi8d256p_qau4c32p_1nlenx1nlen_ztt` and
`qsi8d256p_qsi8c16p_1nlenx1nlen_ztt` variants pack codes only. Metadata remains
in ordinary row-major tensors with independent leading dimensions. Let
`S = amenlen`, `W = 4` for Q4 and `W = 8` for expanded Q6, and `H = UDS/W*S`.
Both A and transposed B use `[row tile][K square][row lane][K lane]`, with
row-major `S*S` squares. Rows are padded to S and K to H. Padding codes are zero
in the private encoding. The public source of Q4 B stores adjacent U4 codes,
low K in the low nibble; the packer recenters each code to signed I4 (`q - 8`).
The source B stride must be even. Q6 B contains signed I8 codes and is copied
without a zero-point adjustment.

| Buffer | Required bytes |
|--------|----------------|
| A | `round_up(M,S) * round_up(K,H)` |
| Q4 B | `round_up(N,S) * round_up(K,H) / 2` |
| Q6 B | `round_up(N,S) * round_up(K,H)` |

Pack and run leading dimensions remain logical element counts; destination
leading dimensions and K origins are multiples of 256. For a row-tile origin R
and K origin Q, the packed offset is `R*round_up(ld,H) + Q*S` bytes for A and
Q6 B, and half that for Q4 B. R must be S-aligned; Q must be 256-aligned.
`actual_m`/`actual_n` do not shrink padded tiles. Always use the public getters.

For segmented packing, advance source and destination pointers to the segment
origin and pass the full packed leading dimension. Each call writes only its
K interval plus row padding; the segment ending at that leading dimension also
writes final K padding. Segment calls may share an MMA panel and do not overwrite
one another. Split-K compute uses the original complete packed allocation:
advance code pointers by the getters and retain the metadata's original K base.
The implementation may read the remainder of that allocation's native panel
and masks values outside the selected group.

Packed A/B buffers are tied to the exact variant, UDS and S, including A whose
K padding differs between Q4 and Q6. They are not interchangeable with RVV,
with floating Ztt GEMM, or across the two quantized variants.

## Performance Considerations

- **Pack once, run many**: For inference, pack weights at model load time.
  Measure the reuse count needed to amortize packing for the workload.
- **Activation repacking**: Reuse the allocation when sizes permit, but repack
  changing activation values on every call.
- **Memory overhead**: Packed buffers can be larger than the source matrix
  because tail tiles are rounded up. Use `tqt_get_*_packed_size` to get the
  exact size.
- **Packed vs. transposed**: If your weight is already transposed and you
  cannot afford the packing latency (e.g., single-use weight), prefer the `t`
  variant. Compare total preparation and execution costs for repeated use.

## See Also

- [Naming Convention](../reference/naming_convention.md) -- full breakdown of
  suffix meanings
- [Micro-kernel Tables](../reference/microkernel_tables.md) -- list of all
  available kernel variants
- [BLAS-Style GEMM](blas_style_gemm.md) -- sgemm/dgemm have no packed variants
  (caller manages layout directly)
