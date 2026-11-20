<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Choosing a kernel

Select a compatible operation and numerical contract first, then compare layouts
and tiles on the target hardware. The [kernel catalog](../reference/microkernel_tables.md)
lists actual variants; a family or backend name does not promise every combination.

## 1. Choose the operation

| Need | Kernel names | Next reference |
|------|--------------|----------------|
| Matrix multiplication, optionally with bias/clamp | `gemm_*` | [GEMM API](../reference/api_reference.md#gemm-with-bias-and-clamp) |
| `D = alpha*A*B + beta*C` | `sgemm_*`, `dgemm_*` | [BLAS-style GEMM](../guide/blas_style_gemm.md) |
| Convolution with explicit im2col preparation | `convcf_*`, `convcl_*` | [im2col example](../../examples/api/convolution/conv_im2col/README.md) |
| Convolution directly from input and geometry | `igemmcf_*`, `igemmcl_*` | [Implicit GEMM](../guide/implicit_gemm.md) |
| Depthwise convolution | `dwconvcf*`, `dwconvcl*` | [Depthwise convolution](../guide/depthwise_conv.md) |
| Vector arithmetic | `{add,sub,mul,div}{vv,vs}_*` | [Elementwise API](../reference/api_reference.md#elementwise-operations) |
| Row/column sum or mean | `{sum,mean}{row,col}_*` | [Reduction API](../reference/api_reference.md#reduction-sum--mean) |
| Unary or gated activation | `relu`, `gelu`, `silu`, `reglu`, `geglu`, `swiglu` | [Activation API](../reference/api_reference.md#activation-relu--gelu--silu--reglu--geglu--swiglu) |

`cf` and `cl` mean channel-first and channel-last. For vector operations, `vv`
takes two vectors and `vs` takes a vector and scalar. For reduction, `row` reduces
N and produces M values; `col` reduces M and produces N values.

## 2. Match types and numerical requirements

Matrix names normally list output D, input A and input B. The family directory
groups base A/B types, so different output types can share one family. For other
operators, follow their [naming grammar](../reference/naming_convention.md).

Check more than the stored types:

- **Floating GEMM:** distinguish native accumulation, `f32acc`, and reduced-precision
  multiplication. An F32 output does not guarantee full FP32 multipliers.
- **W16A32:** RVV uses FP32 A and widens the FP16/BF16 weights during FP32 accumulation.
- **Quantized GEMM:** match codes, scale/zero-point granularity, metadata layout,
  output type and rounding. Model formats such as Q4_K are not directly
  interchangeable with the library's packed execution formats.
- **Bias and clamp:** verify broadcast direction and bias storage type.
  `1xnbiasf32` explicitly uses FP32 bias even when the output is narrower.
- **Activations:** GeLU uses the tanh approximation, and GeLU/SiLU paths have
  backend-specific approximations and special-value behavior. ReLU's native maximum
  returns zero for a NaN input. Read the contract before relying on aliases,
  rounding or NaN/Inf behavior.

Use the [quantization guide](../guide/quantization.md) to select INT8/INT4/INT2,
and block integer-dot paths. The
[API reference](../reference/api_reference.md) describes accumulation, strides,
aliasing and special values; each public header supplies the variant's exact rules.

## 3. Check ISA and runtime support

| Backend | Requirement | Selection notes |
|---------|-------------|-----------------|
| `rvv` | RVV 1.0 and any datatype-specific extensions | FP16 and BF16 require additional instructions/types |
| `ztt` | A matching Ztt profile and runtime state | Match UDS, register counts and runtime geometry; only selected GEMM paths exist |

CMake's `TORQ_TILE_HAS_*` macros report the built library's source groups. They do
not detect the runtime CPU. Verify support before calling getters, packers or run
functions; even a getter may read an ISA-specific CSR. See
[build configuration](build_options.md) and
[runtime dispatch requirements](../guide/integration.md#runtime-isa-and-width-queries).

Where several backends satisfy the same numerical contract, benchmark them for
your workload. The backend suffix alone does not establish which is faster.

## 4. Match layout and epilogue

For a logical GEMM `A[M,K] * B[K,N]`:

| Suffix | Storage | Preparation |
|--------|---------|-------------|
| None | Ordinary row-major A or B | Use the original buffer and its element stride |
| `t` | Transposed storage; for B, row-major `[N,K]` | No conversion if weights already have this layout |
| `p` | The selected variant's packed layout | Query sizes and run its public packers |

A and B may carry suffixes independently. A packed D suffix also changes the
output contract: for example, Ztt floating packed-output GEMM requires D unpacking.
Ordinary Ztt layouts require M aligned to `nlen` and N/K to `pack_factor*nlen`;
use its packed-output path for arbitrary logical dimensions.

Packing is a candidate when weights can be reused. Include A packing, B packing,
metadata conversion and output unpacking in the cost comparison as appropriate.
A shared function-table type does not guarantee a shared packed representation.
See [Packing](../guide/packing.md) and the
[packed GEMM example](../../examples/api/gemm/packed_gemm/README.md).

| Epilogue | Meaning |
|----------|---------|
| `1xnbias` | One value per output column, broadcast across rows |
| `mx1bias` | One value per output row, broadcast across columns |
| `mx1bias_1xnbias` | Both broadcasts, where provided |
| `clamp` | Apply the bounds in the public API's declared type |
| `sgemm` / `dgemm` | Alpha/beta semantics without fused bias/clamp |

Null C/bias and split-K support are variant-specific. For floating GEMM, only exact
negative/positive infinity bounds together request the unbounded clamp behavior.

## 5. Compare available tiles

Start with a small candidate set rather than assuming one universal default:

| Workload shape | Candidate RVV tiles, where available |
|----------------|-------------------------------------|
| General floating GEMM | `8x3vl`, `4x4vl` |
| Small M, wide N | `2x8vl` |
| Narrow N | `8x1vl`, `8x2vl` |
| Integer-dot GEMM | The family's available tiles, commonly `4x1vl` |

Query the variant's getters instead of hard-coding VLEN or matrix dimensions.
`m_step`/`n_step` describe tile geometry and valid block origins; they are not
application cache-block sizes. The [naming reference](../reference/naming_convention.md)
contains the exact tile-to-step mapping. Compare both kernel time and complete
operation cost using the [performance guide](../guide/performance_tuning.md).

## Find the implementation

Use the catalog to identify a kernel base, then inspect its header and available
source variants. For example, from the repository root:

```bash
find src/gemm/gemm_f32_f32 -name 'tqt_gemm_*_rvv.cpp'
find src/gemm/gemm_i8_i8 -name '*reqi32toi8*_rvv.cpp'
```

A first ordinary FP32 call is shown in [Quick start](quick_start.md). The
[integration guide](../guide/integration.md) explains source/package integration,
buffer management and dispatch, and the [example index](../../examples/README.md)
provides complete calling scenarios.
