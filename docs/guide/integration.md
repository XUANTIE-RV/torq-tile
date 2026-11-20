<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Framework integration guide

TORQ-Tile supplies computation and packing routines. Your application owns memory,
chooses kernels for the runtime target, and schedules work across threads.

- [Choose an integration method](#integration-model)
- [Minimal integration](#minimal-integration)
- [Runtime ISA and width queries](#runtime-isa-and-width-queries)
- [Dispatch and tiling](#framework-integration-patterns)
- [Buffers](#buffer-management), [packing](#packing-integration), [K tiling](#k-dimension-tiling-advanced)
- [Thread safety](#thread-safety)

## Integration model

| Method | What you build | When to use it |
|--------|----------------|----------------|
| Installed CMake package | A static or shared library and its selected public headers | Reuse one configured library across applications |
| Local CMake subproject | The library as part of your build | Keep dependency configuration in your project |
| Selected source files | Specific `.cpp` files with their header dependencies | Control the exact kernel subset and ISA settings |

The [installed-package consumer](../../examples/integration/installed_package/README.md)
and [source-subset consumer](../../examples/integration/source_subset/README.md)
are complete, independent projects for the first and third methods.

### Installed package

Build and install with the desired toolchain and ISA as described in
[Build options](../getting_started/build_options.md#installed-package). Then:

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_application LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(torq_tile CONFIG REQUIRED)
add_executable(my_application main.cpp)
target_link_libraries(my_application PRIVATE torq_tile::torq_tile)
```

Configure with `-DCMAKE_PREFIX_PATH=/path/to/prefix` and compatible consumer
compiler/ABI settings. Use the installed `<torq_tile/...>` header paths. The target
supplies include and link dependencies, but does not add backend ISA flags to your
application. Native FP16/BF16 declarations require consumer compiler type support.

### CMake subproject or selected sources

For a local checkout, add it with `add_subdirectory(third_party/torq-tile)` and
link the build-tree target `torq_tile`. A local FetchContent `SOURCE_DIR` works too;
see the [API example projects](../../examples/README.md#build-one-directory).
Build-tree consumers can include headers through `src/...` using the target's
repository-root include directory; the installed package uses `torq_tile/...`.

For selected sources, compile the exact `.cpp` files and keep all referenced
public, interface, shared-kernel and common headers available. Add the repository
root to the include path. This path does not require `tqt_compile_config.h` or a
whole-library build. The [source-subset CMake file](../../examples/integration/source_subset/CMakeLists.txt)
shows the required files and compiler settings for a concrete FP32 GEMM.

## Minimal integration

The following complete `main.cpp` consumes an installed RVV library. Run it only
on a target supporting the library's RVV instructions. It uses ordinary row-major
A and B, the same layout as [Quick start](../getting_started/quick_start.md).

```cpp
#include <cmath>
#include <cstdio>
#include <limits>
#include <torq_tile/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv.h>

int main()
{
    const float a[] = {1, 2, 3, 4, 5, 6};       // [2, 3]
    const float b[] = {7, 8, 9, 10, 11, 12};    // [3, 2]
    const float bias[] = {1, -1};               // [1, 2]
    const float expected[] = {59, 63, 140, 153};
    float d[4] = {};
    const float inf = std::numeric_limits<float>::infinity();

    tqt_run_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv(
        2, 2, 3, a, 3, 0, b, 2, 0, nullptr, 2, d, 2, bias, -inf, inf);

    for (size_t i = 0; i < 4; ++i) {
        if (!std::isfinite(d[i]) || std::fabs(d[i] - expected[i]) > 1e-5f) {
            std::puts("Status: FAILED");
            return 1;
        }
    }
    std::puts("Status: PASSED");
    return 0;
}
```

For this call, `lda=3`, `ldb=2`, `ldc=2` and `ldd=2` are element strides. C is
null, so no residual is added. Both K origins are zero. Exact `[-infinity, infinity]`
skips clamping; finite maximum values are not a no-clamp sentinel. The RVV run
function handles this region's internal tiles and tails, so no outer tile loop is
needed. Other variants, particularly ordinary-layout Ztt GEMM, impose different
shape constraints.

## Runtime ISA and width queries

Register backend function addresses first, then check the execution environment's
ISA capabilities before calling any backend getter, offset helper, packing
function, or compute function. Hosted implementations read the backend geometry
CSR on the first call that needs it, then reuse that value within the
same compilation unit. Getters are not CPU feature detection functions. Taking
a function address does not execute it.

Ztt follows the same policy for its square size `amenlen` and unit datatype size
`ameudsz`: hosted builds cache both; freestanding builds read on demand. Pack and
compute must still use the same nlen and a compatible profile. Its ownership,
register-clobber, aligned ordinary-layout and packed-output contracts are described in
[Ztt GEMM](../reference/api_reference.md#ztt-gemm).

All calls using a compilation unit's cached width must run with the same vector
or matrix width (and UDS for Ztt) as its first use. The cache is shared by threads
using that compilation unit; it is not a per-thread cache. Do not use these
implementations across execution environments with different geometry in the same cache lifetime.

The default hosted C++17 build uses thread-safe initialization. Keep that support
enabled for concurrent first use. Link through a C++ linker driver, or explicitly
provide the corresponding C++ ABI runtime when linking from C; GCC and Clang builds require
`__cxa_guard_acquire` and `__cxa_guard_release` from the C++ runtime. Compiling with
`-fno-threadsafe-statics` is not supported for concurrent first use.

Keep dispatch and library-loading code compiled for the baseline ISA. Backend
objects may use higher ISA settings, but those settings must not propagate to
the baseline code through global flags or cross-target inlining/LTO. TORQ-Tile
does not perform runtime ISA selection for the caller.

Query dimensions, pack buffers, and execute kernels with widths compatible with
both the cached width and the packed layout. The cache does not detect a width
change or adapt buffers packed for a different width.

## Framework integration patterns

A fixed-target application can call a selected variant directly, as above. To
choose between compatible variants, populate the kernel base's `_interface.h`
function table and select a table after checking runtime ISA support.
[basic_gemm.cpp](../../examples/api/gemm/basic_gemm/basic_gemm.cpp) shows complete
tables for multiple RVV tiles and available matrix backends.

The same function-table type establishes an API shape, not numerical or packed-layout
equivalence. Check accumulation precision and rebuild packed buffers when changing
to a variant with a different packing contract.

An application may pass a whole supported region to `run`, or partition it into
blocks for cache reuse or scheduling. For subwindows of an existing buffer,
derive block origins from `get_m_step()` and `get_n_step()` and use the selected
variant's byte-offset getters; leading dimensions still describe that buffer.
The [MN example](../../examples/api/gemm/blocked_gemm/README.md) demonstrates this
with full-K calls. The MNK example instead packs independent panels, as described
under [K-dimension tiling](#k-dimension-tiling-advanced).

## Buffer management

The caller owns every buffer and keeps it valid for the duration of each call.

| Buffer | Contract to check |
|--------|-------------------|
| A and B | Ordinary, transposed or packed representation required by the selected API |
| C | Residual/accumulator input; null only where the interface permits it |
| Bias | Broadcast direction and storage type, which may differ from D |
| D | Writable output of the documented size, stride and alignment |
| Packed data and metadata | Sizes and offsets from the matching variant's public helpers |

Offset and size helpers return bytes; tensor strides generally count elements.
Quantized and packed paths have additional units and alignment rules: use the
specific header's contract. A tight `get_d_size(m,n)` allocation does not cover
extra row padding introduced by a larger `ldd`. Do not infer arbitrary aliasing
support from a `void *` signature; for example, exact C=D is supported by some
GEMM paths, while overlapping D with A or B is not generally valid.

## Packing integration

For repeated weights, query the selected variant's packed size, allocate the
buffer, and call its public B or BT packer. Reuse the packed contents only while
weights, logical dimensions and target geometry remain compatible. Activations
that change must be repacked; reusing the allocation does not reuse its contents.

Use the packed kernel base and its `get_*_packed_offset` functions to consume
packed data. An ordinary-B interface and `get_b_offset` cannot interpret a packed
weight buffer. Some paths pack both operands; Ztt floating packed-output paths
also pack C/bias and require D unpacking.

The [packing guide](packing.md) specifies layouts and sizing. The
[packed GEMM scenario](../../examples/api/gemm/packed_gemm/README.md) shows ordinary
B and transposed B preparation, weight reuse and packed output without mixing
those interfaces.

## K-dimension tiling (advanced)

K tiling is valid only when the selected API supports partial K and residual
accumulation. Check block alignment, K-origin semantics and metadata indexing
before splitting. The [MNK example](../../examples/api/gemm/blocked_gemm/README.md)
repackages each A/B panel for the current K segment. Packing reads the original
buffers with their full strides; packed strides are the segment's `depth`, and
both K origins passed to packing and compute are zero. Each output block starts
from zero and accumulates subsequent segments through C=D. After all segments,
the example adds the original C and bias and applies clamp once.

When applying the epilogue through a supported GEMM API, start the first segment
from the original C (or null), accumulate subsequent segments from D, and supply
bias and final clamp bounds only on the last segment. Pass exact
`[-infinity, infinity]` bounds for the intermediate segments. K-origin arguments
do not automatically defer the epilogue.

Repeated stores to narrow output can change rounding relative to one full-K call.
Packed offsets and quantization metadata may use different origin conventions;
do not apply a generic pointer-advance rule to every API. Ztt floating packed-output
GEMM has no public packed-subwindow or split-K interface.

## Thread safety

Concurrent calls require disjoint output regions, valid input lifetimes and a
compatible target geometry. Hosted implementations share private geometry caches
within each compilation unit and require thread-safe first-use initialization;
see [Runtime ISA and width queries](#runtime-isa-and-width-queries).

Partition M or N into aligned output regions and share read-only inputs/packed
weights. Parallelizing K requires private partial outputs and an explicit reduction;
apply bias and final clamp once. The
[multithreaded GEMM scenarios](../../examples/api/gemm/multithreaded_gemm/README.md)
show both patterns with caller-owned threads. Their timings include thread
creation and joining.

## See also

- [Choosing a kernel](../getting_started/choosing_a_kernel.md)
- [API reference](../reference/api_reference.md)
- [Quantization contracts](quantization.md)
- [Performance tuning](performance_tuning.md)
