<!--
SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
SPDX-License-Identifier: Apache-2.0
-->

# Changelog

## v1.0.0

Initial public versioned release.

- 250 micro-kernel implementations across 52 families: 233 RVV and 17 Ztt.
- GEMM covers FP64, FP32, FP16, BF16, W16A32 mixed precision, integer
  quantization, bias/clamp, BLAS-style SGEMM/DGEMM and block-quantized integer dot.
  FP16 GEMM and convolution include explicit FP32-accumulation variants.
- Convolution supports im2col plus GEMM and implicit GEMM, with channel-first
  and channel-last layouts. RVV depthwise convolution provides FP32/FP16 generic
  and fixed-filter variants, including channel-multiplier packing.
- RVV elementwise add/subtract/multiply/divide, row/column sum and mean, and
  ReLU/GeLU/SiLU with gated ReGLU/GeGLU/SwiGLU support FP32 and FP16.
  SiLU/SwiGLU use the shared FMA-based exp implementation with documented RNE,
  subnormal, overflow, NaN, infinity and signed-zero behavior.
- Integer paths include INT8/UINT8, INT4 W4A16/W4A8, INT2, and adapter-produced
  Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_0-derived execution encodings. Selected Q8_0
  activation paths accept Q1_0/Q2_0/Q4_0/Q5_0-derived weight encodings.
- Ztt provides FP32/FP16/BF16 floating GEMM, native and FP32 accumulation where
  applicable, packed-output tail handling, and Q4_K/expanded Q6_K integer dot.
- CMake static/shared builds, capability probes, selected-header installation,
  direct-source integration and freestanding core support are included.
- Scenario-based examples cover packing, serial K blocks, MN/NK threading,
  quantization, convolution, reductions, activations and independent consumers.
- Public guides document APIs, naming, quantization, packing, numerical behavior,
  building, integration and contribution workflows. Functional tests and
  benchmarks accompany the supported implementations.

Coverage is backend-specific; consult the [kernel catalog](docs/reference/microkernel_tables.md).
Packed buffers require the matching variant and runtime geometry. Model-format
conversion and activation quantization remain the caller's responsibility.
