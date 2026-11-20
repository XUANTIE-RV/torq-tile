<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Performance tuning guide

Compare variants that satisfy the same numerical and layout requirements, then
measure on the intended hardware. Tile geometry and instruction choice suggest
candidates; they do not establish a universal speedup.

## Tile selection strategy

A micro-kernel tile controls register use and operand reuse. Query its
`m_step`/`n_step` and respect valid block origins. These are distinct from the
application's cache blocks and thread partitions.

### RVV candidates

| Tile | Candidate workload | Tradeoff to measure |
|------|--------------------|---------------------|
| `8x3vl` | General floating GEMM | Row reuse versus N-tail work |
| `4x4vl` | Moderate M, wide N | Wider N block with fewer rows |
| `2x8vl` | Small M, wide N | Wide N block and reduced M reuse |
| `8x2vl`, `8x1vl` | Narrow N or register-intensive paths | Smaller N block versus more iterations |

For these ordinary floating GEMM tiles, `vl = VLENB / sizeof(element)`. At
VLEN=256, FP32 has `vl=8`, so `8x3vl` spans 8 by 24 elements. Other operators
and datatypes have their own tile mapping; use the
[naming reference](../reference/naming_convention.md).

### Matrix backends

Ztt geometry depends on `amenlen`, UDS and datatype. Check each candidate's
precision and tail/alignment contract before comparing it with RVV. Ordinary
Ztt layouts require aligned shapes.
See [Choosing a kernel](../getting_started/choosing_a_kernel.md).

## Packed vs transposed vs plain

| Layout | Preparation cost | Useful comparison |
|--------|------------------|-------------------|
| Ordinary B[K,N] | None | Baseline when the input already has this form |
| Transposed Bt[N,K] | None if already transposed; otherwise transpose cost | Direct use of existing transposed weights |
| Packed | Allocate and run the matching packer | Kernel savings versus preparation and storage cost |

For fixed weights, pack B once and reuse it while its format and target geometry
remain compatible. A generally changes for each inference and must be repacked.
Include metadata conversion and output unpacking where required. Reusing an
allocation does not make packing free.

For R uses of fixed weights, compare complete costs such as:

```text
packed total = B preparation + R * (A preparation + compute + output conversion)
ordinary total = R * ordinary-layout operation
```

Do not assume the packed path wins at every shape or reuse count. The
[packing guide](packing.md) and [packed example](../../examples/api/gemm/packed_gemm/README.md)
describe the required preparation.

## Benchmark methodology

### Configure and select cases

Use the [library build recipe](../getting_started/build_options.md#configure-and-build)
with the target's ISA flags, then enable benchmarks in that build directory:

```bash
cmake -S . -B build -DTORQ_TILE_BUILD_BENCHMARK=ON
cmake --build build --target torq_tile_benchmark -j8
```

On a compatible RISC-V machine, list registrations before choosing a filter:

```bash
./build/benchmark/torq_tile_benchmark --benchmark_list_tests=true
./build/benchmark/torq_tile_benchmark -m 128 -n 256 -k 512 \
    --benchmark_filter='tqt_gemm_1xnbias_clamp_f32_f32_f32_(8x3vl|4x4vl|2x8vl)_rvv' \
    --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
```

For cross-compilation, copy the binary and its required runtime libraries to the
board before running. Use `-m/-n/-k` for GEMM dimensions; convolution benchmarks
have separate `--conv-*` options. Filters must match registered names; an empty
selection is not a measurement. QEMU can check execution but its timings do not
measure target-hardware performance.

### Define the timing scope

The common packed GEMM benchmarks prepare buffers before the timed compute loop.
They measure repeated kernel execution, not complete first-use or per-request cost.
Check the selected benchmark's source because other scenarios can include packing,
unpacking, scheduling or thread creation.

Record separately when relevant:

- Kernel compute time with prepared operands.
- A/B packing, metadata conversion and output unpacking.
- Full operation time, including application tiling and scheduling.
- Fixed-weight setup cost and the number of uses over which it is amortized.

For dense GEMM, `2*M*N*K / time_ns` gives the conventional GFLOP/s estimate. This
operation count excludes epilogue and preparation work. For quantized paths, label
an analogous operation rate as GOP/s and state the counting convention.

### Make comparisons reproducible

Keep source revision, compiler flags, CPU/ISA, VLEN or matrix geometry, shapes,
strides, precision, thread count, affinity and timing scope with the results.
Control frequency and background load where possible; use warm-up and repeated
measurements and report variability. A compute-only gain may disappear after
packing or scheduling is included.

## Blocking and threading

Application cache blocks (`MC`, `NC`, `KC`) group multiple micro-kernel tiles.
Choose them against the combined A, B and output working set and measure the
result. Cache capacity alone does not determine the best block sizes.

Serial K tiling reuses a residual/output chain and may change rounding because
partial results are stored between segments. Parallel split-K needs private
partial outputs and a reduction. Both must preserve supported K alignment,
metadata origins, and final bias/clamp semantics. See the
[MN/MNK examples](../../examples/api/gemm/blocked_gemm/README.md).

Partitioning M or N lets threads write disjoint outputs. Account for false sharing,
thread overhead and input reuse, and keep runtime geometry compatible with hosted
caches. The [thread examples](../../examples/api/gemm/multithreaded_gemm/README.md)
show explicit tasks and deterministic reduction.

## Shape-specific starting points

| Workload | Candidates to evaluate |
|----------|------------------------|
| Large-M prefill | Wider row reuse, packed weights, cache blocking and M/N worker partitioning |
| M=1 decode | Small-M RVV tiles and available matrix variants; weight bandwidth and N partitioning |
| Convolution | Explicit im2col cost versus implicit address generation, using the same tensor layout |
| Depthwise convolution | Applicable fixed-filter variants versus generic paths, including pack/unpack cost |

These are experiment choices, not expected winners. Use supported hardware counters
and the platform's profiling tools to distinguish compute pressure, cache misses
and memory stalls; counter names and availability are platform-specific.

## Common pitfalls

- Comparing different precision or clamp contracts as if they were interchangeable.
- Reusing packed data across incompatible variants or hardware widths.
- Applying ordinary floating tail assumptions to aligned-only or quantized APIs.
- Passing a tight dimension instead of the actual row stride for padded tensors.
- Timing compute alone while reporting full-operation latency.
- Counting a missing benchmark or an emulator timing as a hardware result.

## See also

- [Integration guide](integration.md)
- [Micro-kernel catalog](../reference/microkernel_tables.md)
- [Depthwise convolution](depthwise_conv.md)
- [Implicit GEMM](implicit_gemm.md)
