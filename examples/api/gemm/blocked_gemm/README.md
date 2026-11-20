<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Blocking GEMM

`blocked_mn.cpp` partitions the output into MC/NC blocks and uses public byte-offset
getters to select inputs and output. Each call consumes the full K. It supports the
ordinary FP32 interface's available RVV/Ztt variants; Ztt shapes must be aligned.

`blocked_mnk.cpp` also limits K to KC. Its visible N -> K -> M loop packs a B panel
once per N/K pair and reuses it across M blocks. A panels are packed for each M/K
pair. Each output block starts from zero, then uses C=D for subsequent K blocks.
Original C, bias and clamp are applied exactly once after all K blocks. The fixture
crosses several blocks in all three dimensions and includes tail blocks.

This MNK example uses the RVV packed-input, ordinary-output FP32 interface. The
packing and accumulation contract, rather than a common filename suffix, determines
which variants can participate. It does not promise that all backends use this layout.

Accepted options: `--m`, `--n`, `--k`, `--mc`, `--nc`; MNK also accepts `--kc`.
MC/NC must be multiples of each selected variant's `m_step/n_step`. Defaults adapt
to each variant. Values are positive and at most 1048576; allocation failures are
reported. Follow the [CMake recipe](../../../README.md#build-one-directory) with
`-S examples/api/gemm/blocked_gemm` and a separate build directory. Build targets
`tqt_example_blocked_mn` and `tqt_example_blocked_mnk`, then run CTest there.
For a manual RVV run with VLEN=128, the MNK binary also accepts, for example,
`--mc 16 --nc 96 --kc 32 --m 65 --n 195 --k 133`; invoke it through a matching
QEMU runner. Other vector lengths may require different block alignment.

The micro-tile steps describe legal call boundaries; MC/NC/KC describe caller blocks.
These small defaults teach mechanics. They are not cache-size recommendations or
measured performance claims. See [API contracts](../../../../docs/reference/api_reference.md).
