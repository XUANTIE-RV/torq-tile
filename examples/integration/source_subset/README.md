<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Build selected source files

This is an independent CMake consumer. It compiles one selected FP32 GEMM source
and includes that source's public, interface and shared implementation headers from
the checkout. It does not configure or link the full library. Read the explicit
`target_sources` branches in `CMakeLists.txt` to see the dependency boundary.

```bash
cmake -S examples/integration/source_subset -B build/consumer-source \
  -DTORQ_TILE_SOURCE_DIR="$PWD" -DTQT_BACKEND=RVV \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
  -DCMAKE_C_COMPILER=riscv64-unknown-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=riscv64-unknown-linux-gnu-g++ \
  -DCMAKE_C_FLAGS='-march=rv64gcv -mabi=lp64d' \
  -DCMAKE_CXX_FLAGS='-march=rv64gcv -mabi=lp64d'
cmake --build build/consumer-source
qemu-riscv64 -cpu rv64,v=true,vlen=128,elen=64 -L /path/to/sysroot build/consumer-source/consumer
```

`TQT_BACKEND` accepts RVV or ZTT. Select matching compiler flags and runtime
for the backend; selection is explicit, not runtime dispatch. The kernel source's
architecture guard rejects incompatible flags. The fixture queries steps and uses
aligned dimensions valid for the selected ordinary interface.

A successful run prints `Status: PASSED` and exits zero. No threading dependency
or test framework is required. Run these commands from the repository root and
replace `/path/to/sysroot` with the target runtime supplied by your compiler.
