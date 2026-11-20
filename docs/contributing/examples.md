<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Maintaining examples

Examples teach distinct public calling scenarios. Before adding one, check the
[scenario index](../../examples/README.md) for a compatible interface and preparation
flow. New datatypes, tiles and ISAs normally reuse an existing scenario; add a new
program when it needs a distinct calling workflow. Tests and benchmarks retain
coverage of individual implementations.

## Source and registration

Use scenario names without ISA or tile suffixes. Each program keeps its data,
reference computation and verification locally, with allocation, packing, offset
arithmetic and threading visible. Keep different interface contracts in separate
flows and avoid a shared generic example runtime. Code and comments are English.

Each API leaf directory is an independent CMake project. Keep its
`torq_tile_add_example(name source REQUIRES ...)` registration on one line so the
batch runner and auditor can discover it. Keep common build setup consistent
across topics; only thread scenarios depend on Threads.

API sources include public `torq_tile/...` headers and use `#if TORQ_TILE_HAS_*`
from `torq_tile/tqt_compile_config.h` to select variants. Align these guards with
the CMake capability requirements. These macros describe the matching library;
use compatible consumer ABI/types and a CPU supporting its instructions. Keep
target-specific options private. Source-subset integration remains independent
of the generated library configuration.

## Documentation and checks

Update the scenario index when adding or removing a program. Keep detailed READMEs
for complex calling workflows and integration projects; simple examples can explain
their contracts in source comments. Link to existing API and quantization guides
for shared contracts.

Each program performs finite elementwise reference checks, returns nonzero on
failure, and prints `Status: PASSED` with exit code zero on success.
Configure the affected leaf project with a matching compiler and
runner using the [example build recipe](../../examples/README.md#build-one-directory).
For a configured basic GEMM project:

```bash
cmake --build build/example-basic --target tqt_example_basic_gemm
(cd build/example-basic && ctest --output-on-failure)
```

Check that each new program has a CMake registration, that its required capability
macros match the source guards, and that the index reaches it. CTest runs the
program directly, optionally through the configured runner. A nonzero exit,
timeout or explicit failure marker fails the test. It does not require a success
marker or Python; do not use `PASS_REGULAR_EXPRESSION`, which can override a
nonzero exit status.

When changing CMake registration, exercise successful execution, a nonzero exit
after printing PASSED and an explicit failure marker. A zero exit without a
success marker is accepted by CTest. Cross builds without a runner must configure
and build without registering executable tests.

Quantization loss and timing are separate from correctness; QEMU timing does not
establish hardware performance.
