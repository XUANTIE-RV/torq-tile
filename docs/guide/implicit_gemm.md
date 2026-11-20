<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Implicit GEMM convolution guide

This guide explains implicit GEMM (igemm) convolution in TORQ-Tile: how it
differs from explicit im2col + GEMM, when to prefer it, and how to use it.

## What is implicit GEMM

Standard convolution via GEMM works in two steps:

1. **im2col**: Rearrange input patches into a matrix (materializes all input
   windows into a contiguous buffer).
2. **GEMM**: Multiply the weight matrix by the im2col matrix.

Implicit GEMM **fuses** these two steps. Instead of materializing the im2col
buffer, the kernel computes input element addresses on-the-fly during the matrix
multiplication. It reads directly from the original input tensor using an address
computation that maps `(output_position, kernel_element)` to the correct input
location.

```
Explicit:   im2col_buf = im2col(input)    -->  output = weight * im2col_buf
Implicit:   output = weight * input        (address mapping done inside kernel)
```

## conv (im2col + GEMM) vs igemm (implicit GEMM)

TORQ-Tile provides both approaches:

| Aspect | conv (im2col + GEMM) | igemm (implicit GEMM) |
|--------|---------------------|----------------------|
| Memory | Needs im2col buffer: O(M * K) extra | No extra buffer |
| Compute | Separate im2col pass + GEMM | Single fused pass |
| Setup | Call `tqt_im2col_cf/cl()` then GEMM kernel | Pass `tqt_conv_params` to igemm kernel |
| Kernel flexibility | Any GEMM tile works | Dedicated igemm tiles |
| Best for | Large/dilated kernels, buffer reuse | Memory-constrained, standard kernels |

### Memory comparison

For a Conv2D with IC=256, KH=KW=3, OH=OW=56:
- im2col buffer: 256 * 9 * 56 * 56 * sizeof(float) = ~28 MB
- igemm: 0 extra bytes

This difference matters on embedded/edge devices with limited SRAM/DRAM.

## Available igemm variants

### Channel-first (igemmcf)

Input layout: `[C, D, H, W]` (NCDHW for 3D, NCHW for 2D)

| Kernel | Tile | Data type | Notes |
|--------|------|-----------|-------|
| `igemmcf_mx1bias_clamp_f32_f32_f32` | 3vlx8 | FP32 | Non-packed activation |
| `igemmcf_mx1bias_clamp_f32_f32p_f32` | 3vlx8 | FP32 | Packed activation |
| `igemmcf_mx1bias_clamp_f16_f16_f16` | 3vlx8 | FP16 | Non-packed activation |
| `igemmcf_mx1bias_clamp_f16_f16p_f16` | 3vlx8 | FP16 | Packed activation |
| `igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8` | 1vlx4 | INT8 | Requantized output |
| `igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8` | 1vlx4 | INT8 | Packed activation, requantized |

The `3vlx8` tile has a **swapped layout** compared to standard GEMM tiles:
- M direction (3*vl) maps to output channels (OC) -- vectorized
- N direction (8) maps to spatial positions (OD*OH*OW) -- scalar

This matches the channel-first weight layout where OC is the leading dimension.

### Channel-last (igemmcl)

Input layout: `[D, H, W, C]` (NDHWC for 3D, NHWC for 2D)

| Kernel | Tile | Data type | Notes |
|--------|------|-----------|-------|
| `igemmcl_1xnbias_clamp_f32_f32_f32t` | 8x3vl | FP32 | Transposed weight |
| `igemmcl_1xnbias_clamp_f32_f32_f32p` | 8x3vl | FP32 | Packed weight |
| `igemmcl_1xnbias_clamp_f16_f16_f16t` | 8x3vl | FP16 | Transposed weight |
| `igemmcl_1xnbias_clamp_f16_f16_f16p` | 8x3vl | FP16 | Packed weight |
| `igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt` | 4x1vl | INT8 | Requantized output |
| `igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp` | 4x1vl | INT8 | Packed weight, requantized |

The `8x3vl` tile uses the standard GEMM layout:
- M direction (8) maps to spatial positions -- scalar
- N direction (3*vl) maps to output channels (OC) -- vectorized

## How it works internally

### Address computation

For channel-last igemm, the kernel maps each `(m_idx, k_idx)` pair to an input
element address:

```
m_idx -> (od, oh, ow)     spatial output position
k_idx -> (kd, kh, kw, ic) kernel element + input channel

input_addr = input_base + (id * IH * IW + ih * IW + iw) * C + ic
where:
    id = od * stride_d + kd * dilation_d - pad_d
    ih = oh * stride_h + kh * dilation_h - pad_h
    iw = ow * stride_w + kw * dilation_w - pad_w
```

If the computed input coordinate falls outside bounds (padding region), the kernel
returns zero.

### The tqt_conv_params struct

Both conv and igemm kernels use the same parameter structure:

```c
struct tqt_conv_params {
    size_t input_shape[3];   // {D, H, W}
    size_t output_shape[3];  // {OD, OH, OW} -- caller pre-computes
    size_t kernel_shape[3];  // {KD, KH, KW}
    size_t stride[3];        // {SD, SH, SW}
    size_t dilation[3];      // {DD, DH, DW}
    size_t pad[3];           // {PD, PH, PW}
    size_t input_channel;
    size_t output_channel;
    size_t groups;           // For grouped convolution
};
```

For 2D convolution, set D-dimensions to 1: `input_shape = {1, IH, IW}`,
`kernel_shape = {1, KH, KW}`, etc.

## Using igemm kernels

### Channel-last example (igemmcl)

```c
#include "tqt_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv.h"
#include "src/common/tqt_conv.h"

void my_conv2d_cl(const float *input,   // [IH, IW, IC]
                  const float *weight,  // [OC, KH * KW * IC] (transposed)
                  const float *bias,    // [OC] or nullptr
                  float *output,        // [OH, OW, OC]
                  size_t IC, size_t OC,
                  size_t IH, size_t IW, size_t KH, size_t KW,
                  size_t stride, size_t pad)
{
    // Set up params
    tqt_conv_params params = {};
    params.input_shape[0] = 1;  params.input_shape[1] = IH;  params.input_shape[2] = IW;
    params.kernel_shape[0] = 1; params.kernel_shape[1] = KH;  params.kernel_shape[2] = KW;
    params.stride[0] = 1;      params.stride[1] = stride;    params.stride[2] = stride;
    params.dilation[0] = 1;    params.dilation[1] = 1;       params.dilation[2] = 1;
    params.pad[0] = 0;         params.pad[1] = pad;          params.pad[2] = pad;
    params.input_channel = IC;
    params.output_channel = OC;
    params.groups = 1;

    size_t OH = (IH + 2 * pad - KH) / stride + 1;
    size_t OW = (IW + 2 * pad - KW) / stride + 1;
    params.output_shape[0] = 1;  params.output_shape[1] = OH;  params.output_shape[2] = OW;

    // GEMM dimensions for channel-last igemm:
    //   M = OH * OW (spatial output positions)
    //   N = OC (output channels)
    //   K = KH * KW * IC (kernel elements * input channels)
    size_t M = OH * OW;
    size_t N = OC;
    size_t K = KH * KW * IC;

    // Run igemm -- no im2col buffer needed
    tqt_run_igemm_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(
        M, N, K,
        input,          // original input tensor (NOT im2col)
        0,              // group_idx
        0,              // m_idx (start of spatial positions)
        0,              // k_idx_a
        weight, K, 0,   // B, ldb, k_idx_b
        nullptr, N,        // C (accumulation), ldc
        output, N,      // D, ldd
        bias,
        -FLT_MAX, FLT_MAX,  // finite clamp range
        &params);
}
```

### Channel-first example (igemmcf)

```c
#include "tqt_igemmcf_mx1bias_clamp_f32_f32_f32_3vlx8_rvv.h"
#include "src/common/tqt_conv.h"

void my_conv2d_cf(const float *input,   // [IC, IH, IW]
                  const float *weight,  // [OC, IC * KH * KW] (row-major)
                  const float *bias,    // [OC]
                  float *output,        // [OC, OH, OW]
                  size_t IC, size_t OC,
                  size_t IH, size_t IW, size_t KH, size_t KW,
                  size_t stride, size_t pad)
{
    tqt_conv_params params = {};
    params.input_shape[0] = 1;  params.input_shape[1] = IH;  params.input_shape[2] = IW;
    params.kernel_shape[0] = 1; params.kernel_shape[1] = KH;  params.kernel_shape[2] = KW;
    params.stride[0] = 1;      params.stride[1] = stride;    params.stride[2] = stride;
    params.dilation[0] = 1;    params.dilation[1] = 1;       params.dilation[2] = 1;
    params.pad[0] = 0;         params.pad[1] = pad;          params.pad[2] = pad;
    params.input_channel = IC;
    params.output_channel = OC;
    params.groups = 1;

    size_t OH = (IH + 2 * pad - KH) / stride + 1;
    size_t OW = (IW + 2 * pad - KW) / stride + 1;
    params.output_shape[0] = 1;  params.output_shape[1] = OH;  params.output_shape[2] = OW;

    // GEMM dimensions for channel-first igemm:
    //   M = OC (output channels)
    //   N = OH * OW (spatial output positions)
    //   K = IC * KH * KW
    size_t M = OC;
    size_t N = OH * OW;
    size_t K = IC * KH * KW;

    tqt_run_igemm_igemmcf_mx1bias_clamp_f32_f32_f32_3vlx8_rvv(
        M, N, K,
        input,          // original input tensor
        0,              // group_idx
        0,              // m_idx
        0,              // k_idx_a
        weight, K, 0,   // B (weight as row-major [OC, K]), ldb=K, k_idx_b
        nullptr, N,        // C, ldc
        output, N,      // D, ldd
        bias,
        -FLT_MAX, FLT_MAX,
        &params);
}
```

## K-dimension tiling with igemm

Like GEMM kernels, igemm supports K-dimension tiling via `k_idx_a` and `k_idx_b`.
This is useful when K is large and you want to improve cache reuse:

```c
size_t K_BLOCK = KH * KW * IC_per_group;  // CL: one complete KD plane

for (size_t k = 0; k < K; k += K_BLOCK) {
    size_t k_len = MIN(K - k, K_BLOCK);
    bool is_first = (k == 0);
    bool is_last = (k + K_BLOCK >= K);

    tqt_run_igemm_igemmcl_...(
        M, N, k_len,
        input, group_idx, 0, k,       // k_idx_a = k
        weight, K, k,                  // k_idx_b = k
        is_first ? nullptr : output, N,   // accumulate from D on subsequent chunks
        output, N,
        is_first ? bias : nullptr,        // bias initializes the first chunk
        is_last ? clamp_min : -INFINITY,
        is_last ? clamp_max : INFINITY,
        &params);
}
```

**Important:** CL split-K slices cover complete `KH * KW * IC_per_group` KD planes;
CF slices cover complete `KD * KH * KW` input-channel blocks. Both K origins must
match the selected slice. Bias initializes the first segment, C carries intermediate
results, and clamp is applied only on the last segment. Floating IGEMM skips
external max/min on that final segment only for exact `[-INFINITY, INFINITY]` public
bounds. Finite and one-sided intervals retain their original behavior.

## When to choose igemm over conv

### Prefer igemm when:

- **Memory is constrained**: No im2col buffer allocation (saves O(M * K) memory).
  For large feature maps this can be tens of megabytes.
- **Small batch size**: The one-time im2col cost is proportionally higher for
  small batches.
- **Standard kernels (3x3, 5x5, 7x7)**: The address computation overhead is
  minimal relative to the compute.
- **Streaming inference**: Avoids a separate im2col pass, reducing latency.

### Prefer conv (im2col + GEMM) when:

- **Dilated convolution with large dilation**: The igemm address computation
  becomes complex and the access pattern scattered.
- **Very large kernels** (11x11+): The number of kernel elements increases the
  per-output-element address computation cost.
- **Reusing im2col across layers**: If multiple layers share the same input
  spatial dimensions, you can compute im2col once and reuse it.
- **You need maximum GEMM throughput**: Explicit GEMM on a contiguous im2col
  buffer can achieve slightly higher sustained throughput due to perfectly
  sequential memory access.

### Performance comparison (typical)

| Kernel size | igemm overhead vs conv | Recommendation |
|-------------|----------------------|----------------|
| 1x1 | ~0% (no real im2col) | Either (igemm saves the buffer) |
| 3x3 | ~2-5% slower compute | igemm (memory savings dominate) |
| 5x5 | ~5-8% slower compute | igemm for memory-constrained |
| 7x7 | ~8-12% slower compute | Depends on available memory |
| 11x11+ | ~15%+ slower compute | conv usually better |

## Grouped convolution

Both conv and igemm support grouped convolution via the `groups` parameter and
`group_idx` argument:

```c
params.groups = num_groups;
size_t IC_per_group = IC / num_groups;
size_t OC_per_group = OC / num_groups;

for (size_t g = 0; g < num_groups; g++) {
    // Adjust weight and output pointers for this group
    const float *weight_g = weight + g * OC_per_group * K_per_group;
    float *output_g = output + g * OC_per_group * spatial_size;

    tqt_run_igemm_igemmcf_...(
        OC_per_group, spatial_size, K_per_group,
        input,         // full input tensor (kernel uses group_idx internally)
        g,             // group_idx
        0, 0,
        weight_g, K_per_group, 0,
        nullptr, spatial_size,
        output_g, spatial_size,
        bias_g,
        clamp_min, clamp_max,
        &params);
}
```

## Comparison with conv kernels

TORQ-Tile also provides explicit conv kernels (`convcf`, `convcl`) that use the
im2col approach internally or expect the caller to provide im2col input:

```c
// convcl approach: caller does im2col, then calls a GEMM-like kernel
tqt_im2col_cl<float>(group_idx, input, im2col_buf, &params);
// Then use any standard GEMM kernel on the im2col buffer

// igemmcl approach: kernel handles everything
tqt_run_igemm_igemmcl_...(M, N, K, input, group_idx, ...);
```

The conv kernels in `src/conv/conv_*` provide the im2col utilities
(`tqt_im2col_cf`, `tqt_im2col_cl`) as inline template functions in
`src/common/tqt_conv.h`.

## See also

- [Depthwise conv guide](depthwise_conv.md) -- for depthwise convolution
- [Integration guide](integration.md) -- general framework integration
- [Performance tuning](performance_tuning.md) -- tile selection and optimization
- [Naming convention](../reference/naming_convention.md) -- decode igemm kernel names
- Source: `../../src/conv/igemm_f32_f32/`, `../../src/conv/igemm_f16_f16/`, `../../src/conv/igemm_i8_i8/`
- im2col helpers: `../../src/common/tqt_conv.h`
