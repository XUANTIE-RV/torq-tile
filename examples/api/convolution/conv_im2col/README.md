<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Conv Im2Col

The program explicitly creates im2col data for each group, then calls the
convolution GEMM API. CF uses weights `[OC,K]` and lowered input `[K,P]`; CL uses
lowered input `[P,K]` and transposed-storage weights. The CF and CL interfaces,
weight transformations and calls remain separate, with their own compatible arrays.
Temporary im2col buffers belong to the caller. They can be released after the call.

The deterministic fixture uses a 5x7 image, 3x3 kernel and padding 1. Ordinary
convolution uses two groups; depthwise uses one group per input channel. A direct
scalar convolution over canonical logical tensors checks both output layouts,
including bias and clamp. It does not derive its reference from im2col output.

Follow the [CMake recipe](../../../README.md#build-one-directory) with
`-S examples/api/convolution/conv_im2col`, a separate build directory and target
`tqt_example_conv_im2col`. The initial
arrays contain the available FP32 RVV representative; other backends can join the
same layout-specific array when their contract fits. ISA availability does not
change the source filename. See [convolution contracts](../../../../docs/reference/api_reference.md).
