<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Quantized GEMM

These are three different representation workflows, not a complete datatype catalog.
Each prints a correctness result against the **same quantized contract**, then
reports quantization loss against the original floating-point computation separately.
Loss is diagnostic and finite-checked; no application accuracy threshold is implied.

| Program | Contract | Important preparation |
|---|---|---|
| `int8_requant.cpp` | I8 A/B -> I8 D | Per-tensor asymmetric A, per-channel symmetric B, I32 bias, output scale/zp |
| `int4_weight_only.cpp` | FP16 A x grouped INT4 B -> FP32 D | Offset-8 nibble encoding, FP16 scales per K group, A/B packing |
| `w4a8_dynamic_quant.cpp` | Dynamic per-row I8 A x INT4 B -> FP32 D | Row activation scales/zps, per-channel weight scales, two codes per byte |

INT8 uses exact power-of-two effective multipliers below, equal to and above one,
with an independent integer ties-to-even reference and saturation/clamp cases.
The zero activation row and zero weight channel exercise zero-range handling.
The grouped INT4 program uses the existing `8x1vl` FP32 accumulation variant;
its `8x2vl` sibling has a different FP16 intermediate contract and remains covered
by tests. The INT4 scale is stored before encoding, so the encoder and reference
use the same represented scale.

W4A8 includes zero in each row's min/max range. A zero row uses scale 1 and a valid
zero point; a zero weight channel uses scale 1. The whole-matrix call keeps scale
pointers at their full logical origins. This nibble layout is not an opaque packed
buffer shared with the W4A16 example.

See the [quantization guide](../../../../docs/guide/quantization.md) for INT2,
GGML-derived blocks, custom LUTs, affine offsets and numerical
contracts. New dtype/ISA variants alone do not require another example.
