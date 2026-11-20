<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# GEMM examples

Read [basic_gemm](basic_gemm/basic_gemm.cpp) first: the public dimensions and strides
count elements while offset/size getters count bytes. The same ordinary-output
interface is populated with available RVV tiles and Ztt implementations. The
fixture uses exactly representable dyadic inputs and checks each result, including
C, bias, clamp and untouched output padding. Ztt ordinary layouts use aligned M/N/K.

Continue with [packing](packed_gemm/README.md), then [MN/MNK blocking](blocked_gemm/README.md).
The [thread examples](multithreaded_gemm/README.md) add caller-owned parallelism;
[quantized GEMM](quantized_gemm/README.md) explains representation changes separately.
Each example rebuilds packed buffers for its current variant: equal interfaces do
not establish packed-layout compatibility.

All executables run without arguments. Blocking and threading programs additionally
accept positive shape/block parameters, subject to the current variant's alignment.
See the topic pages for the accepted options. Follow the
[example build recipe](../../README.md#build-one-directory), selecting the desired
leaf directory and `tqt_example_<scene>` target.

For other types, transpose modes, BLAS alpha/beta and extended quantization formats,
use the [API reference](../../../docs/reference/api_reference.md) and
[quantization guide](../../../docs/guide/quantization.md). Tests remain the exhaustive
implementation coverage; these examples teach a compact set of calling patterns.
