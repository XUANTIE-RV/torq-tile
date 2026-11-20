<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
    SPDX-License-Identifier: Apache-2.0
-->

# TORQ-Tile

[![XuanTie](https://img.shields.io/badge/XuanTie-Homepage-8145E2.svg)](https://www.xrvm.cn/)
![RISC-V](https://img.shields.io/badge/arch-RISC--V-F2AC10.svg?logo=riscv)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](CHANGELOG.md)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)

TORQ-Tile provides optimized **micro-kernels** for AI/ML workloads on RISC-V CPUs,
with particular support for XuanTie processors. It is designed for C and C++
frameworks that manage their own memory, scheduling, and threads.

## Key features

- No external library dependencies or dynamic memory allocation in the micro-kernels.
- Stateless APIs with caller-owned buffers and explicit tensor strides.
- Variants for different instruction sets, data types, tile sizes, and fusion patterns.
- Integration through selected source files or a CMake library package.

## What is a micro-kernel?

A micro-kernel computes an operation such as matrix multiplication on a caller-selected
region of a tensor. The framework controls packing, workload partitioning, and dispatch.
Variants of a kernel implement the same operation with different tile sizes or ISA
requirements, subject to each variant's numerical contract.

## Supported instructions and extensions

- RISC-V Vector Extension (RVV 1.0)
- Attached Matrix Extension (Ztt 0.6)

Availability depends on the operation, data type, and required ISA features. CMake
selects source groups using compiler capability checks; the caller selects the
appropriate kernel for the runtime target.

## Supported micro-kernels

| Family | Operations and data types |
|--------|---------------------------|
| [GEMM](src/gemm/) | FP64, FP32, FP16, BF16, mixed precision and integer quantization; plain, bias/clamp and BLAS-style variants |
| [Convolution](src/conv/) | im2col + GEMM and implicit GEMM; FP32, FP16, BF16, and INT8, depending on the path |
| [Depthwise convolution](src/dwconv/) | FP32 and FP16; generic and specialized 2D variants |
| [Elementwise](src/elementwise/) | Add, subtract, multiply, and divide; FP32 and FP16 |
| [Reduction](src/reduction/) | Row/column sum and mean; FP32 and FP16 |
| [Activation](src/activation/) | ReLU, GeLU, SiLU, and gated ReGLU, GeGLU, SwiGLU; FP32 and FP16 |

The table summarizes each family's coverage; not every type, layout, or operation
is available on every backend. FP64 GEMM is RVV-only; Ztt covers selected GEMM
paths. Elementwise, reduction, activation and depthwise convolution provide FP32
and FP16 RVV implementations. See the [kernel catalog](docs/reference/microkernel_tables.md)
for exact type, layout and backend coverage.

The library exposes execution formats and packing contracts. Framework/model quantization formats may require conversion to
the selected kernel's execution layout.

## How to build

To build and run your first FP32 RVV example, follow the
[Quick start](docs/getting_started/quick_start.md). The commands below build the library.

### Prerequisites

- CMake 3.16 or newer and a build tool such as Make or Ninja.
- A RISC-V C11/C++17 toolchain supporting the instructions and intrinsics used by
  the selected kernels. Ztt requires a toolchain with the corresponding support.

The commands below build a static library for RVV using `rv64gcv`. Choose ISA flags
for your target; FP16, BF16 and Ztt kernels require additional extensions.
See [build options](docs/getting_started/build_options.md) for target configuration,
shared libraries, and freestanding builds.

### Compile natively on RISC-V Linux

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=rv64gcv -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="-march=rv64gcv -mabi=lp64d"
cmake --build build
```

### Cross-compile to RISC-V Linux

Put `riscv64-unknown-linux-gnu-gcc` and `riscv64-unknown-linux-gnu-g++` on `PATH`,
then use the supplied toolchain file:

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-unknown-linux-gnu-gcc.toolchain.cmake \
    -DCMAKE_C_FLAGS="-march=rv64gcv -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="-march=rv64gcv -mabi=lp64d"
cmake --build build
```

Use separate build directories when changing toolchains or target configurations.

### Build and run tests

After either configuration above, enable tests in the same build directory, retaining
its compiler and ISA settings:

```shell
cmake -S . -B build -DTORQ_TILE_BUILD_TEST=ON
```

For cross-compiled tests, configure a compatible QEMU runner before running CTest.
Replace the sysroot path with the one supplied by your cross toolchain:

```shell
cmake -S . -B build \
    -DTORQ_TILE_TEST_RUNNER="qemu-riscv64 -cpu rv64,v=true,vlen=128,elen=64 -L /path/to/sysroot"
```

Then build and run:

```shell
cmake --build build
(cd build/test && ctest --output-on-failure)
```

Tests and benchmarks are disabled by default. Tests use GoogleTest; adding
`-DTORQ_TILE_BUILD_BENCHMARK=ON` enables Google Benchmark. See
[build options](docs/getting_started/build_options.md#tests-and-benchmarks) for
dependencies and offline source paths.

## How to use

Choose the integration method that fits your framework:

- [Build selected source files](examples/integration/source_subset/README.md): compile
  the selected `.cpp` files with their public, interface, and implementation headers.
- [Consume an installed package](examples/integration/installed_package/README.md):
  install the library, use `find_package(torq_tile CONFIG REQUIRED)`, and link
  `torq_tile::torq_tile`.

The [example index](examples/README.md) covers packing, quantized GEMM, convolution,
and multithreaded execution. See the [integration guide](docs/guide/integration.md)
for buffer management and dispatch, or the [documentation index](docs/README.md)
for the full reference.

## Filename convention

Sources are organized by operation group, family, and kernel base. For example:

```text
src/gemm/
└── gemm_f16_f16/
    └── gemm_1xnbias_clamp_f16_f16_f16t/
        ├── tqt_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv.cpp
        └── tqt_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv.h
```

Here `gemm_f16_f16` groups GEMM kernels by their input types. In the variant name,
`1xnbias_clamp` denotes row-broadcast bias and clamping; `f16_f16_f16t` describes
FP16 output, FP16 A, and transposed FP16 B; `8x3vl` is the tile descriptor; and `rvv`
is the backend. The header exposes the variant's getters and `tqt_run_gemm_...` entry point.

See the [naming convention](docs/reference/naming_convention.md) for other families,
quantization tags, layouts, and tile dimensions.

## Release

TORQ-Tile follows Semantic Versioning. The badge shows the source version declared
in [CMakeLists.txt](CMakeLists.txt); release tags identify released revisions.
See [CHANGELOG.md](CHANGELOG.md) for release notes and upcoming changes.

## Support

Please report bugs or request support through this repository's issue tracker.
Include the source version or commit, compiler and target ISA, and a reproducer.

## Frequently Asked Questions (FAQ)

### Is x86 or Arm supported?

The optimized kernels target RISC-V. An x86 or Arm host can cross-compile them,
but executing them requires a compatible RISC-V target or emulator.

### What CPUs does TORQ-Tile support?

A CPU must support all instructions required by the selected variant. An `rvv`,
or `ztt` suffix identifies the backend; data types and operations can require
additional extensions. Check the kernel header and
[build configuration](docs/getting_started/build_options.md) for those requirements.

### Can the micro-kernels be multi-threaded?

The caller partitions the work and dispatches kernel calls through its own threads.
Kernels do not create threads internally. See the
[multithreaded GEMM example](examples/api/gemm/multithreaded_gemm/README.md),
[thread safety guidance](docs/guide/integration.md#thread-safety), and
[depthwise convolution threading guide](docs/guide/depthwise_conv.md#threading-for-depthwise-conv).

## License

TORQ-Tile is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
