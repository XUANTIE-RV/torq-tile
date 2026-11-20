<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Implicit Gemm

The program passes the original input tensor and convolution geometry to IGEMM.
The kernel calculates spatial addresses internally; the caller does not allocate
a complete im2col matrix or pointer-indirection table. This library's IGEMM means
**implicit GEMM**. It must not be confused with a different library's caller-built
indirect-convolution table. CF and CL have separate preparation and interface arrays.

The deterministic fixture uses a 5x7 image, 3x3 kernel and padding 1. Ordinary
convolution uses two groups; depthwise uses one group per input channel. A direct
scalar convolution over canonical logical tensors checks both output layouts,
including bias and clamp. It does not derive its reference from im2col output.

Follow the [CMake recipe](../../../README.md#build-one-directory) with
`-S examples/api/convolution/implicit_gemm`, a separate build directory and target
`tqt_example_implicit_gemm`. The initial
arrays contain the available FP32 RVV representative; other backends can join the
same layout-specific array when their contract fits. ISA availability does not
change the source filename. See [convolution contracts](../../../../docs/reference/api_reference.md).
