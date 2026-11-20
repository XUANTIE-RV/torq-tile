<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Consume an installed package

This independent project uses `find_package(torq_tile CONFIG REQUIRED)` and links
`torq_tile::torq_tile`. Includes use the installed `torq_tile/...` layout. It neither
adds the checkout to its include path nor compiles kernel sources itself. The
installed capability header rejects a backend absent from the built library.

Run from the repository root after building an RVV library in `build` using
the [library build recipe](../../../docs/getting_started/build_options.md#configure-and-build):

```bash
cmake --install build --prefix "$PWD/build/tqt-install"
cmake -S examples/integration/installed_package -B build/consumer-installed \
  -DCMAKE_PREFIX_PATH="$PWD/build/tqt-install" -DTQT_BACKEND=RVV \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
  -DCMAKE_C_COMPILER=riscv64-unknown-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=riscv64-unknown-linux-gnu-g++ \
  -DCMAKE_C_FLAGS='-march=rv64gcv -mabi=lp64d' \
  -DCMAKE_CXX_FLAGS='-march=rv64gcv -mabi=lp64d'
cmake --build build/consumer-installed
qemu-riscv64 -cpu rv64,v=true,vlen=128,elen=64 -L /path/to/sysroot build/consumer-installed/consumer
```

Use RVV or ZTT with matching compiler/runtime settings. The consumer is small
because it demonstrates dependency discovery and linking, not another GEMM tutorial.
A successful run prints `Status: PASSED` and exits zero. Replace `/path/to/sysroot`
with the matching target runtime. To check relocation, move the installation to a
new prefix and configure a fresh consumer build with that prefix; the consumer must
not need the original checkout or installation path.
