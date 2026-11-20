<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Examples

These programs demonstrate public API calls and application integration, organized
by calling scenario. Start with the [GEMM reading order](api/gemm/README.md), or
choose a topic below.

## API scenarios

| Area | Source | What it teaches |
|---|---|---|
| GEMM | [basic_gemm](api/gemm/basic_gemm/basic_gemm.cpp) | Strides, residual C, bias, clamp, interface variants |
| GEMM | [packed_gemm](api/gemm/packed_gemm/README.md) | B/BT packing, packed sizes, weight reuse, packed output |
| GEMM | [blocked_gemm](api/gemm/blocked_gemm/README.md) | MN and MNK blocks, offsets, K accumulation |
| GEMM | [multithreaded_gemm](api/gemm/multithreaded_gemm/README.md) | Real MN and NK worker tasks and deterministic reduction |
| GEMM | [quantized_gemm](api/gemm/quantized_gemm/README.md) | INT8, grouped INT4 and dynamic W4A8 |
| Convolution | [conv_im2col](api/convolution/conv_im2col/README.md) | Explicit im2col and CF/CL layout preparation |
| Convolution | [implicit_gemm](api/convolution/implicit_gemm/README.md) | Original input plus geometry, without caller im2col |
| Convolution | [depthwise_conv](api/convolution/depthwise_conv/README.md) | CF/CL layouts, padding and channel multiplier |
| Elementwise | [same_shape.cpp](api/elementwise/same_shape.cpp) | Add two equal-length inputs, tail and exact alias |
| Elementwise | [scalar_broadcast.cpp](api/elementwise/scalar_broadcast.cpp) | Add a scalar to every element |
| Reduction | [rowwise_reduce.cpp](api/reduction/rowwise_reduce.cpp) | `[M,N] -> [M]`, optional initial values |
| Reduction | [columnwise_reduce.cpp](api/reduction/columnwise_reduce.cpp) | `[M,N] -> [N]`, padded input stride |
| Activation | [unary_activation.cpp](api/activation/unary_activation.cpp) | SiLU with input/output strides and aliasing |
| Activation | [gated_activation.cpp](api/activation/gated_activation.cpp) | SwiGLU gate/up slices in a fused `[M,2N]` buffer |

## Integration

- [source_subset](integration/source_subset/README.md): compile a selected kernel
  source and its headers directly into a consumer.
- [installed_package](integration/installed_package/README.md): consume an installed,
  relocatable `torq_tile` package.

## Build one directory

Each API leaf directory is an independent CMake project. Use a separate build
directory for each example project, such as `build/example-basic`, so it can
coexist with the library build in `build`. Run the commands from the repository
root with a matching cross compiler and QEMU:

```bash
cmake -S examples/api/gemm/basic_gemm -B build/example-basic \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
  -DCMAKE_C_COMPILER=riscv64-unknown-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=riscv64-unknown-linux-gnu-g++ \
  -DCMAKE_C_FLAGS='-march=rv64gcv -mabi=lp64d' \
  -DCMAKE_CXX_FLAGS='-march=rv64gcv -mabi=lp64d' \
  -DCMAKE_BUILD_TYPE=Release \
  -DTORQ_TILE_EXAMPLE_RUNNER='qemu-riscv64 -cpu rv64,v=true,vlen=128,elen=64 -L /path/to/sysroot'
cmake --build build/example-basic --target tqt_example_basic_gemm
(cd build/example-basic && ctest --output-on-failure)
```

The default uses FetchContent with the local checkout to build the library with
tests and benchmarks disabled. To reuse an installed library, add
`-Dtorq_tile_DIR=/path/to/prefix/lib/cmake/torq_tile`. The library and consumer must
use compatible ABI and ISA settings. In topics with several programs, select one
with `--target tqt_example_<scene>`; executables are placed in `bin/`.

API examples select variants using the matching library's `TORQ_TILE_HAS_*` macros
in `torq_tile/tqt_compile_config.h`. Use a CPU supporting those implementations.
CTest runs the program directly for native builds or with a cross runner; it does
not require Python. A nonzero exit, timeout or explicit failure marker fails the
test. CTest does not require a `Status: PASSED` line.
A cross build without a runner only builds. Integration projects have their own
build commands in the READMEs linked above.

## Select another scenario

Use the desired leaf directory as `-S` and a separate build directory as `-B`.
Build the targets in that directory, then run CTest there. For example, after
configuring `examples/api/gemm/blocked_gemm` with the same toolchain and runner:

```bash
cmake --build build/example-blocked --target tqt_example_blocked_mn tqt_example_blocked_mnk
(cd build/example-blocked && ctest --output-on-failure)
```

The [Quick start](../docs/getting_started/quick_start.md) explains the RVV recipe;
[build options](../docs/getting_started/build_options.md) covers other targets and
library reuse. Match both compiler flags and runtime capabilities before selecting
a Ztt scenario.

## Further reading

The [API reference](../docs/reference/api_reference.md) and
[quantization guide](../docs/guide/quantization.md) describe the supported contracts.
Thread-example timings include thread creation and joining; QEMU timings do not
measure hardware performance. See [maintenance rules](../docs/contributing/examples.md)
when changing examples.
