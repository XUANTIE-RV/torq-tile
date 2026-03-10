<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
    SPDX-License-Identifier: Apache-2.0
-->

# TORQ-Tile

[![XuanTie](https://img.shields.io/badge/XuanTie-Homepage-8145E2.svg)](https://www.xrvm.cn/)
![RISC-V](https://img.shields.io/badge/arch-RISC--V-F2AC10.svg?logo=riscv)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](./version)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)

TORQ-Tile is an open-source library that provides optimized performance-critical routines, also known as **micro-kernels** and **tile kernels**, for artificial intelligence (AI) workloads tailored for RISC-V CPUs, with special optimizations for XuanTie processors.

These routines are tuned to exploit the capabilities of RISC-V architecture extensions including RVV (RISC-V Vector Extension) and other extensions, aiming to maximize performance on XuanTie processors.

The TORQ-Tile library has been designed for ease of adoption into C or C++ machine learning (ML) and AI frameworks. Specifically, developers looking to incorporate specific micro-kernels into their projects can only include the corresponding **.c** and **.h** files associated with those micro-kernels and a common header file.

## Who is this library for?

TORQ-Tile is a library for AI/ML framework developers interested in accelerating the computation on RISC-V CPUs, particularly XuanTie processors.

## What is a micro-kernel?

A micro-kernel, or **ukernel**, can be defined as a near-minimum amount of software to accelerate a given ML operator with high performance.

Following are examples of a micro-kernel:

- Function to perform matrix multiplication

*However, why are the preceding operations not called kernels or functions instead?*

**This is because the micro-kernels are designed to give the flexibility to process also a portion of the output tensor.**

> ℹ️ The API of the micro-kernel is intended to provide the flexibility to dispatch the operation among different working threads or process only a section of the output tensor. Therefore, the caller can control what to process and how.

A micro-kernel exists for different RISC-V architecture extensions and computational parameters (for example, different output tile sizes). These implementations are called **micro-kernel variants**. All micro-kernel variants of the same micro-kernel type perform the same operation and return the same output result.

## Key features

Some of the key features of TORQ-Tile are the following:

- No dependencies on external libraries

- No dynamic memory allocation

- No memory management​

- No scheduling

- Stateless, stable, and consistent API​

- Performance-critical compute-bound and memory-bound micro-kernels

- Specialized micro-kernels utilizing different RISC-V CPU architectural features (for example, **RVV**)

- Specialized micro-kernels for different fusion patterns

- Micro-kernel as a standalone library, consisting of only a **.c**, **.cpp** and **.h** files

> ℹ️ The micro-kernel API is designed to be as generic as possible for integration into third-party runtimes.

## Supported instructions and extensions

- RISC-V Vector Extension (RVV)

## Filename convention

The `src/gemm` directory is the home for all GEMM (General Matrix Multiply) micro-kernels. The micro-kernels are grouped in separate directories based on the data types. For example, all the FP16 matrix-multiplication micro-kernels are held in the `gemm_f16_f16_f16/` directory.

Inside the operator directory, you can find:

- *The common utilities*, which are helper functions necessary for the correct functioning of the micro-kernels. For example, packing utilities are available in `src/common/tqt_pack_rvv.h`.
- *The micro-kernels* files, which are held in separate sub-directories.

The name of the micro-kernel folder provides the description of the operation performed and the data type of the destination and source tensors. The general syntax for the micro-kernel folder is as follows:

`<op>_<dst-data-type>_<src0-data-type>_<src1-data-type>_...`

All **.c/.cpp** and **.h** pair files in that folder are micro-kernel variants. The variants are differentiated by specifying the computational parameters (for example, the tile size like `8x3vl`), the RISC-V extension (for example, RVV). The general syntax for the micro-kernel variant is as follows:

`tqt_<op>_<data-types>_<compute-params>_<extension>.c/.h`

For example:
- `tqt_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv.h` - FP16 GEMM with bias and clamp, using 8x3vl tile size, optimized for RVV
- `tqt_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv.h` - FP32 packed GEMM with bias and clamp, using RVV

All functions defined in the **.h** header file of the micro-kernel variant follow the naming pattern:

`tqt_<op>_<micro-kernel-variant-filename>.c/.cpp/.h`

## Supported micro-kernels

For a list of supported micro-kernels refer to the [source](/src/) directory. The micro-kernels are grouped in separate directories based on the data types.

Currently supported GEMM variants include:
- **FP32**: `gemm_f32_f32_f32/` - Single precision floating point
- **FP16**: `gemm_f16_f16_f16/` - Half precision floating point

Each variant includes optimized implementations for:
- RVV (RISC-V Vector Extension) - e.g., `8x3vl_rvv`

Common operations supported:
- `gemm_1xnbias_clamp` - GEMM with bias addition and clamp activation
- Packed and transposed variants (indicated by `p` and `t` suffixes)

## Project Structure

```bash
torq-tile/
├── cmake/            # CMake configuration files
├── examples/         # Usage examples
├── scripts/          # Build and utility scripts
├── src/              # Source code
│   ├── gemm/         # GEMM micro-kernel implementations
│   └── common/       # Common utilities and headers
└── test/             # Test cases
```

## How to build

### Prerequisites

TORQ-Tile requires the following dependencies, obtainable via your preferred package manager, to be installed and available on your system to be able to build the project.

- `build-essential`
- `cmake >= 3.18`

In addition, you may choose to use the following toolchains:

- (Optional) `RISC-V GNU toolchain` for cross-compilation to RISC-V targets
- (Optional) `XuanTie toolchain` available from XuanTie official channels for optimal performance on XuanTie processors

### Build Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `CMAKE_C_FLAGS` / `CMAKE_CXX_FLAGS` | String | *(required)* | Compiler flags specifying the target platform and architecture. Must be set according to your target CPU, e.g. `-march=rv64gcv -mabi=lp64d` |
| `TORQ_TILE_BUILD_SHARED` | Bool | `OFF` | Build as shared library (`.so`) instead of static library (`.a`) |
| `TORQ_TILE_BUILD_TEST` | Bool | `ON` | Build test programs |
| `TORQ_TILE_BUILD_BENCHMARK` | Bool | `OFF` | Build benchmark programs |
| `TORQ_TILE_TEST_RUNNER` | String | *(empty)* | Command prefix for running test executables. Useful for cross-compiled binaries that require an emulator, e.g. `"qemu-riscv64 -cpu any -L /path/to/sysroot"` |

### Compile natively on a RISC-V system

You can quickly compile TORQ-Tile on your system with a RISC-V processor by using the following commands:

```shell
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=rv64gcv_zvfh -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="-march=rv64gcv_zvfh -mabi=lp64d" \
    -S . -B build/
cmake --build ./build
```

### Cross-compile to RISC-V Linux®

The RISC-V GNU toolchain can be used to cross-compile to a Linux system with a RISC-V processor from an x86_64 Linux host machine. Ensure the toolchain is available on your PATH and provide to CMake the RISC-V toolchain CMakefile found in `cmake` directory with the `-DCMAKE_TOOLCHAIN_FILE` option.

```shell
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-unknown-linux-gnu.toolchain.cmake \
    -DCMAKE_C_FLAGS="-march=rv64gcv_zvfh -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="-march=rv64gcv_zvfh -mabi=lp64d" \
    -S . -B build/
cmake --build ./build
```

### Build and run tests

To build with tests enabled and run them via QEMU:

```shell
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-unknown-linux-gnu.toolchain.cmake \
    -DCMAKE_C_FLAGS="-march=rv64gcv_zvfh -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="-march=rv64gcv_zvfh -mabi=lp64d" \
    -DTORQ_TILE_BUILD_TEST=ON \
    -DTORQ_TILE_TEST_RUNNER="qemu-riscv64 -cpu any -L /path/to/sysroot" \
    -S . -B build/
cmake --build ./build
ctest --test-dir build/test --output-on-failure
```

## Release

### Cadence

Releases will be done regularly. All releases can be found in the release section.

### Version

The release version conforms to Semantic Versioning.

> ⚠️ Please note that API modifications, including function name changes, and feature enhancements may occur without advance notice.

## Support

Please raise a GitLab/GitHub Issue for technical support.

## Frequently Asked Questions (FAQ)

### Is x86 or Arm supported?

No, TORQ-Tile is a library of micro-kernels optimized for the RISC-V architecture, specifically XuanTie processors. The micro-kernels use specific RISC-V instruction set architecture features which prevent compilation for other architectures like x86 or Arm. Note that this applies to all micro-kernels including the packing and quantization micro-kernels.

### What ML operators are supported?

TORQ-Tile does not provide traditional ML operators; it implements a set of optimized micro-kernels which can be used as building blocks to implement ML operators. Please refer to the sections [What is a micro-kernel](#what-is-a-micro-kernel), [Supported micro-kernels](#supported-micro-kernels) for more information.

### What CPUs does TORQ-Tile support?

Any CPU that implements the RISC-V architecture extensions listed in section [Supported instructions and extensions](#supported-instructions-and-extensions) is supported by TORQ-Tile. The library is specifically optimized for XuanTie processors but can run on any RISC-V CPU with the required extensions. However, every micro-kernel does not exist in versions for all architecture extensions. Each micro-kernel advertises the required architecture extensions through the micro-kernel name.

Most operating systems provide a mechanism to list the architecture extensions supported by the runtime environment, e.g. `cat /proc/cpuinfo` on Linux.

### What is the difference between TORQ-Tile and other RISC-V AI libraries?

TORQ-Tile focuses on providing low-level, highly optimized micro-kernels that can be easily integrated into existing AI/ML frameworks. Unlike full-featured libraries that provide complete operator implementations and runtime management, TORQ-Tile offers building blocks that give framework developers fine-grained control over computation, memory management, and threading.

**TORQ-Tile is designed for frameworks where the runtime, memory manager, thread management, and fusion mechanisms are already available**.

### Can the micro-kernels be multi-threaded?

**Yes, they can**. The micro-kernel can be dispatched among multiple threads using the thread management available in the target AI/ML framework.

*The micro-kernel does not use any internal threading mechanism*. However, the micro-kernel's API is designed to allow the computation to be carried out only on specific areas of the output tensor. Therefore, this mechanism is sufficient to split the workload on parallel threads.

Please refer to the examples directory for examples on how to use micro-kernels in a multithreaded environment.

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.
