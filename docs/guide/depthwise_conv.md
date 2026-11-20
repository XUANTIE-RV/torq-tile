<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Depthwise convolution guide

This guide covers the depthwise convolution micro-kernels in TORQ-Tile: when to
use them, how to choose between generic and specialized variants, and how to
integrate them into your framework.

On this page:

- [What is depthwise convolution](#what-is-depthwise-convolution)
- [Available variants](#available-variants)
- [Layout choice](#layout-choice)
- [Generic vs specialized: when to use which](#generic-vs-specialized-when-to-use-which)
- [API differences from GEMM](#api-differences-from-gemm)
- [Packing for depthwise](#packing-for-depthwise)
- [Threading for depthwise conv](#threading-for-depthwise-conv)
- [Complete example: MobileNet depthwise layer](#complete-example-mobilenet-depthwise-layer)
- [Multiplier regression checks](#multiplier-regression-checks)
- [See also](#see-also)

## What is depthwise convolution

Depthwise convolution applies a separate filter to each input channel
independently. Unlike standard convolution, there is no cross-channel mixing:
each output channel depends only on the corresponding input channel.

```
Standard conv:  output[oc] = sum_ic( filter[oc, ic] * input[ic] )
Depthwise conv: output[c]  = filter[c] * input[c]     (one filter per channel)
```

This makes depthwise convolution much cheaper (O(C * K^2 * H * W) vs
O(C^2 * K^2 * H * W)) and is widely used in efficient architectures like
MobileNet, EfficientNet, and ConvNeXt.

## Available variants

TORQ-Tile provides generic and specialized depthwise convolution kernels.
The public variants use RVV with FP32 or FP16 operands.

### Generic kernels (any kernel size and stride)

| Kernel | Layout | Tile | Bias type | Description |
|--------|--------|------|-----------|-------------|
| `dwconvcf_mx1bias_clamp_f32_f32_f32` | CF | 1x1vl | Mx1 | Generic, non-packed |
| `dwconvcf_mx1bias_clamp_f32p_f32p_f32p` | CF | 1x1vl | Mx1 | Generic, packed |
| `dwconvcl_1xnbias_clamp_f32_f32_f32t` | CL | 1x1vl | 1xN | Generic, non-packed |
| `dwconvcl_1xnbias_clamp_f32p_f32p_f32p` | CL | 1x1vl | 1xN | Generic, packed |
| `dwconvcf_mx1bias_clamp_f16_f16_f16` | CF | 1x1vl | Mx1 | Generic FP16, non-packed |
| `dwconvcf_mx1bias_clamp_f16p_f16p_f16p` | CF | 1x1vl | Mx1 | Generic FP16, packed |
| `dwconvcl_1xnbias_clamp_f16_f16_f16t` | CL | 1x1vl | 1xN | Generic FP16, non-packed |
| `dwconvcl_1xnbias_clamp_f16p_f16p_f16p` | CL | 1x1vl | 1xN | Generic FP16, packed |

### Specialized kernels (fixed kernel size and stride)

| Kernel | Layout | Tile | Filter | Stride |
|--------|--------|------|--------|--------|
| `dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p` | CF | 4x1x1vl | 3x3 | 1 |
| `dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p` | CF | 2x1x1vl | 3x3 | 2 |
| `dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p` | CF | 4x1x1vl | 5x5 | 1 |
| `dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p` | CL | 4x1x1vl | 3x3 | 1 |
| `dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p` | CL | 4x1x1vl | 3x3 | 1 |
| `dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p` | CL | 2x1x1vl | 3x3 | 2 |
| `dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p` | CL | 4x1x1vl | 5x5 | 1 |
| `dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p` | CF | 4x1x1vl | 3x3 | 1 |
| `dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p` | CF | 2x1x1vl | 3x3 | 2 |
| `dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p` | CF | 4x1x1vl | 5x5 | 1 |
| `dwconvcl3x3s1_1xnbias_clamp_f16_f16_f16p` | CL | 4x1x1vl | 3x3 | 1 |
| `dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p` | CL | 4x1x1vl | 3x3 | 1 |
| `dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p` | CL | 2x1x1vl | 3x3 | 2 |
| `dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p` | CL | 4x1x1vl | 5x5 | 1 |

Specialized kernels fully unroll the filter loops, eliminating inner-loop overhead
and enabling register scheduling across taps. Speedups depend on the target and shape.

## Layout choice

### Channel-first (CF): input = [C, H, W]

- Kernels use `dwconvcf` prefix and `mx1bias` (bias is per-channel, broadcast
  across spatial dimensions).
- Common in classic frameworks (PyTorch default, Caffe).
- Weight layout: `[C, 1, KH, KW]` (PyTorch depthwise convention).

### Channel-last (CL): input = [H, W, C]

- Kernels use `dwconvcl` prefix and `1xnbias` (bias is per-channel, applied along
  the channel/N dimension).
- Common in mobile/edge frameworks (TFLite, ONNX Runtime, XNNPACK).
- Weight layout: varies by variant (transposed `t` or packed `p`).

**Choosing a layout:** Match the layout your framework uses natively. Converting
between CF and CL just for the depthwise kernel is rarely worthwhile -- the
transpose cost typically exceeds any kernel performance difference.

## Generic vs specialized: when to use which

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| 3x3 filter, stride 1 | `dwconv*3x3s1` | Fixed filter/stride permits an unrolled implementation |
| 3x3 filter, stride 2 | `dwconv*3x3s2` | Fixed filter/stride permits an unrolled implementation |
| 5x5 filter, stride 1 | `dwconv*5x5s1` | Fixed filter/stride permits an unrolled implementation |
| 5x5 filter, stride 2 | Generic | No specialized variant yet |
| 7x7 or larger filter | Generic | No specialized variant |
| Non-standard stride | Generic | Specialized only for s1/s2 |
| Dynamic kernel size | Generic | Kernel size resolved at runtime |

**Rule of thumb:** If your filter is 3x3 or 5x5 with stride 1 or 2, always use
the specialized variant.

## API differences from GEMM

Unlike GEMM kernels (which take M, N, K and matrix pointers with leading
dimensions), depthwise convolution kernels take spatial parameters via the
`tqt_conv_params` struct:

```c
struct tqt_conv_params {
    size_t input_shape[3];   // {D, H, W} (set D=1 for 2D)
    size_t output_shape[3];  // {OD, OH, OW} (caller computes)
    size_t kernel_shape[3];  // {KD, KH, KW} (set KD=1 for 2D)
    size_t stride[3];        // {SD, SH, SW}
    size_t dilation[3];      // {DD, DH, DW}
    size_t pad[3];           // {PD, PH, PW}
    size_t input_channel;    // IC (= groups for depthwise)
    size_t output_channel;   // OC (multiple of IC for depthwise)
    size_t groups;           // Must equal input_channel for depthwise
};
```

For depthwise convolution, `groups == input_channel > 0` and output channels are
a multiple of input channels. With `M = OC / IC`, output channels are ordered as
`oc = ic * M + m`, where `0 <= m < M`; output channel `oc` uses input channel
`oc / M`. Raw weights, bias, and unpacked output all use this standard order.

Packed weights and output retain multiplier-major blocks: block
`oc1 = m * IC1 + ic1`, where `IC1 = ceil(IC / C0)`, starts at channel offset
`m * IC + ic1 * C0`. Each block has contiguous lanes for input channels and a
compact final block. Pack/unpack converts between this internal order and the
standard channel order; the buffer sizes and API signatures are unchanged.

**Migration for M > 1:** earlier versions incorrectly used `oc = m * IC + ic`
for ordinary tensors. Regenerate cached packed weights from standard raw weights,
and remove any integration-side conversion that compensated for the old order.
M=1 channel semantics are unchanged.

### Generic kernel API

```c
// Non-packed generic (channel-first)
void tqt_run_dwconv_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv(
    const void *input,     // [IC, IH, IW]
    const void *weight,    // [OC, 1, KH, KW]
    const void *bias,      // [OC] or nullptr
    void *output,          // [OC, OH, OW]
    const tqt_conv_params *params,
    float clamp_min,
    float clamp_max);
```

### Specialized kernel API

The packed specialized kernels use the following calling pattern:

```c
// Query buffer sizes
size_t out_size = tqt_get_output_size_dwconvcf3x3s1_..._4x1x1vl_rvv(&params);
size_t inp_size = tqt_get_packed_input_size_dwconvcf3x3s1_..._4x1x1vl_rvv(&params);
size_t wgt_size = tqt_get_packed_weight_size_dwconvcf3x3s1_..._4x1x1vl_rvv(&params);
size_t oh_step  = tqt_get_oh_step_dwconvcf3x3s1_..._4x1x1vl_rvv();

// Allocate packed buffers
void *packed_input  = malloc(inp_size);
void *packed_weight = malloc(wgt_size);
void *packed_output = malloc(out_size);

// Pack input and weight
tqt_run_pack_input_dwconvcf3x3s1_..._4x1x1vl_rvv(input, packed_input, &params);
tqt_run_pack_weight_dwconvcf3x3s1_..._4x1x1vl_rvv(weight, packed_weight, &params);

// Run kernel (processes oh_step output rows per call)
tqt_run_dwconv_dwconvcf3x3s1_..._4x1x1vl_rvv(
    0, OH,              // oh_start, oh_end (full height)
    packed_input,
    packed_weight,
    bias,
    packed_output,
    &params,
    clamp_min, clamp_max);

// Unpack output back to standard layout
tqt_run_unpack_output_dwconvcf3x3s1_..._4x1x1vl_rvv(packed_output, output, &params);
```

## Packing for depthwise

### Pack helpers

Pack utilities are provided in `src/common/tqt_dwconv_pack_rvv.h`. Each
specialized kernel also provides its own `pack_input`, `pack_weight`, and
`unpack_output` functions as part of its public API.

### What packing does

- **Input packing:** Rearranges the input tensor so that each output-height strip
  accesses contiguous memory, with channels grouped for vector loads.
- **Weight packing:** Interleaves filter coefficients in the order the kernel's
  inner loop accesses them, enabling sequential loads without gather operations.
- **Output packing:** The kernel writes in packed format for optimal store
  throughput; unpacking restores standard layout.

### Integration pattern for repeated inference

```c
// Model load time: pack weight once
tqt_run_pack_weight_...(weight, packed_weight, &params);

// Each inference call: pack input, run, unpack output
tqt_run_pack_input_...(input, packed_input, &params);
tqt_run_dwconv_...(0, OH, packed_input, packed_weight, bias,
                   packed_output, &params, clamp_min, clamp_max);
tqt_run_unpack_output_...(packed_output, output, &params);
```

For repeated inference, weight packing cost is amortized. Input packing must be
done per-call since activations change each batch.

## Threading for depthwise conv

### Generic kernel

The generic kernel processes the entire spatial volume in one call. For
multi-threaded execution, partition the channel dimension:

```c
#pragma omp parallel for
for (size_t c = 0; c < C; c += chunk) {
    // Each thread runs the kernel on a subset of channels
    // Adjust input/output/weight/bias pointers to channel offset
}
```

### Specialized kernel (OH partitioning)

Specialized kernels accept `oh_start` and `oh_end` parameters, enabling
partitioning along the output height dimension:

```c
size_t oh_step = tqt_get_oh_step_..._4x1x1vl_rvv();  // e.g., 4

#pragma omp parallel for schedule(static)
for (size_t oh = 0; oh < OH; oh += oh_step * ROWS_PER_THREAD) {
    size_t oh_end = MIN(oh + oh_step * ROWS_PER_THREAD, OH);
    tqt_run_dwconv_...(oh, oh_end, packed_input, packed_weight,
                       bias, packed_output, &params, clamp_min, clamp_max);
}
```

## Complete example: MobileNet depthwise layer

```c
#include "tqt_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv.h"
#include "src/common/tqt_conv.h"

void mobilenet_dw_layer(const float *input, const float *weight,
                        const float *bias, float *output,
                        size_t C, size_t IH, size_t IW)
{
    // Set up conv params for 2D depthwise, 3x3, stride 1, pad 1
    tqt_conv_params params = {};
    params.input_shape[0] = 1;   params.input_shape[1] = IH;  params.input_shape[2] = IW;
    params.kernel_shape[0] = 1;  params.kernel_shape[1] = 3;   params.kernel_shape[2] = 3;
    params.stride[0] = 1;        params.stride[1] = 1;         params.stride[2] = 1;
    params.dilation[0] = 1;      params.dilation[1] = 1;       params.dilation[2] = 1;
    params.pad[0] = 0;           params.pad[1] = 1;            params.pad[2] = 1;
    params.input_channel = C;
    params.output_channel = C;
    params.groups = C;  // depthwise: groups == channels

    // Compute output shape
    size_t OH = (IH + 2 * params.pad[1] - params.kernel_shape[1]) / params.stride[1] + 1;
    size_t OW = (IW + 2 * params.pad[2] - params.kernel_shape[2]) / params.stride[2] + 1;
    params.output_shape[0] = 1;  params.output_shape[1] = OH;  params.output_shape[2] = OW;

    // Query packed buffer sizes
    size_t inp_size = tqt_get_packed_input_size_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv(&params);
    size_t wgt_size = tqt_get_packed_weight_size_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv(&params);
    size_t out_size = tqt_get_output_size_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv(&params);

    void *packed_input  = malloc(inp_size);
    void *packed_weight = malloc(wgt_size);
    void *packed_output = malloc(out_size);

    // Pack
    tqt_run_pack_input_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv(
        input, packed_input, &params);
    tqt_run_pack_weight_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv(
        weight, packed_weight, &params);

    // Run (ReLU6 clamp)
    tqt_run_dwconv_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv(
        0, OH, packed_input, packed_weight, bias, packed_output,
        &params, 0.0f, 6.0f);

    // Unpack
    tqt_run_unpack_output_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv(
        packed_output, output, &params);

    free(packed_input);
    free(packed_weight);
    free(packed_output);
}
```

## Multiplier regression checks

Each of the two public DWConv test executables includes `*MultiplierOrdering` checks
for M=1/2/3, channel and spatial tails, optional bias, clamp, and output canaries.
Packed variants additionally check the packed weight order and unpack mapping
independently. The FP32 RVV executable's `*GroupedConvAndIgemm` test compares all
groups against the ordinary conv and igemm paths using exact integer inputs.

Run the full DWConv suites with both Release and Debug builds; vary QEMU `vlen`
over 128/256/512 for RVV to exercise
different channel-block tails.

## See also

- [Implicit GEMM guide](implicit_gemm.md) -- for standard (non-depthwise) convolution
- [Integration guide](integration.md) -- general framework integration patterns
- [Naming convention](../reference/naming_convention.md) -- decode kernel names
- Source: `../../src/dwconv/` -- depthwise kernel implementations
- Pack helpers: `../../src/common/tqt_dwconv_pack_rvv.h`
