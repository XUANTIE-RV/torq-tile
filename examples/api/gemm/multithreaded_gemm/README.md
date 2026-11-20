<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Multithreaded GEMM

Both programs create real `std::thread` workers, call micro-kernels concurrently,
join every worker and compare the complete output with an independent scalar
reference. Only these targets link `Threads::Threads`; the library remains stateless.

`parallel_mn.cpp` assigns disjoint M/N output blocks to workers. Every task reads
shared A/B and owns its final output region. There is no cross-thread K reduction.

`parallel_nk.cpp` keeps M intact and splits both N and K. Each task writes a private
FP32 partial buffer with no C, bias or clamp. After join, the main thread reduces K
partitions in a fixed order for each N block, then adds original C and bias once and
clamps once. The default has small M, two N blocks and four K partitions: eight
tasks for eight workers. MN alone would expose only two tasks. Ztt K partitions
respect the chosen matrix alignment.

This is a teaching case for too few output tasks and long K. Large M/N normally
provide enough independent tasks without split-K. An arbitrary small M alone does
not establish a need for split-K; a large N may already supply enough tasks.

Both accept `--m --n --k --nc --threads`; MN also accepts `--mc`, NK accepts `--kc`.
Thread count is 1..256. Shape/block values are positive and at most 1048576. Use the
defaults to run all available tiles without guessing their alignment. Inputs are
read-only; worker failure is propagated after all started threads are joined.

The output reports task counts, worker count and elapsed time; NK also reports
partial-buffer bytes. Timing includes thread creation, kernel calls, join and NK
reduction. It is not a persistent-thread benchmark; QEMU timing gives no hardware
speedup evidence. Functional validation should compare one worker and several
workers, including repeated runs. CPU tuning belongs in dedicated benchmarks.
