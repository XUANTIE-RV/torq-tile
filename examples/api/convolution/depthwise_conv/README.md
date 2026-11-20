<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Depthwise Conv

The program demonstrates both CF and CL depthwise layouts and a channel multiplier
of two. CF inputs/outputs are channel-major and weights are `[OC,KH,KW]`; CL inputs/
outputs are channel-last and weights are `[KH,KW,OC]`. The two calls use different
interface types even when signatures look similar. Specialized 1D/3x3/5x5 kernels,
packing and low-precision variants remain in the API documentation and tests.

The deterministic fixture uses a 5x7 image, 3x3 kernel and padding 1. Ordinary
convolution uses two groups; depthwise uses one group per input channel. A direct
scalar convolution over canonical logical tensors checks both output layouts,
including bias and clamp. It does not derive its reference from im2col output.

Follow the [CMake recipe](../../../README.md#build-one-directory) with
`-S examples/api/convolution/depthwise_conv`, a separate build directory and target
`tqt_example_depthwise_conv`. The initial
arrays contain the available FP32 RVV representative; other backends can join the
same layout-specific array when their contract fits. ISA availability does not
change the source filename. See [convolution contracts](../../../../docs/reference/api_reference.md).
