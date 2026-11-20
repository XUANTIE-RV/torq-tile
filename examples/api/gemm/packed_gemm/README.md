<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Packed GEMM

`packed_gemm.cpp` queries allocation sizes, packs both ordinary B `[K,N]` and
transposed B `[N,K]`, checks their packed buffers agree, and reuses the weights for
two activation matrices. Packing sizes and offsets are bytes; source strides are
elements. Buffers are owned by the caller and remain alive until the call completes.

RVV variants use packed A/B and ordinary C/D. Ztt uses a different existing
interface with packed C, bias and D; the example shows their pack/unpack steps in
a separate namespace. It deliberately does not cast between interface structs or
reuse one variant's packed data in another variant. Arbitrary logical M/N tails are
handled by the respective packers. Scalar references use the original matrices.

Follow the [CMake recipe](../../../README.md#build-one-directory) with
`-S examples/api/gemm/packed_gemm`, a separate build directory and target
`tqt_example_packed_gemm`. Select a compatible RVV or Ztt configuration.
See [packing contracts](../../../../docs/guide/packing.md) and the
[public API](../../../../docs/reference/api_reference.md) for layout details.
