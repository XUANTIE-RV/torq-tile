# Compile isolation checks

`python3 scripts/compile_isolation/check.py` is a standalone developer/CI check for
RISC-V targets. It does not use GTest/CTest or fetch dependencies. Target execution
is opt-in through `--qemu`; the original quick/full plans remain compile/link
checks. Run it from any directory; Python needs only its standard library.

## Choose a check

```bash
# Default: three profiles, small compilation probes only.
python3 scripts/compile_isolation/check.py
python3 scripts/compile_isolation/check.py --quick

# Complete configured matrix, with total and per-suite wall time.
time python3 scripts/compile_isolation/check.py --full -j8

# Compile every selected source for one CPU.
python3 scripts/compile_isolation/check.py --profile rvv-fp16 --suite source-compile -j8

# Check an installed library, including external C/C++ consumers.
python3 scripts/compile_isolation/check.py --profile rvv --suite install-package -j8
```

There are two built-in plans, `--quick` and `--full`; no arguments means quick.
There is no `--preset` option. Use `--profile` and `--suite` for focused checks.
Both are repeatable; explicit profiles default to `compile-probes`. `--profile`
cannot be combined with `--quick` or `--full`. An explicit `--suite` overrides the
chosen plan's suites; `--suite all` requests all suites for each selected profile.
Suites with unmet profile prerequisites report SKIP (for example, `cmake-config`
on baseline `rvv`); declared empty profiles use the rejection check described
below. `--full` selects suitable combinations and avoids unnecessary repetition.

| Plan | Profiles | Suites |
| --- | --- | --- |
| quick | `rv64-scalar`, `rvv`, `rvv-fp16-bf16` | `compile-probes` only. |
| full | `rv64-scalar` | `compile-probes`, `source-selection` (expected empty-build rejection). |
| full | `rvv` | `compile-probes`, `source-selection`, `source-compile`, `global-init`, `install-package`. |
| full | `rvv-fp16`, `rvv-bf16`, `rvv-bf16-zvfhmin` | `compile-probes`, `source-selection`, `source-compile`, `global-init`. |
| full | `rvv-fp16-bf16` | All original suites except `runtime-contract`. |

Full covers six profiles, including five non-empty RVV profiles. Package checks
use baseline RVV and RVV with FP16/BF16; configuration and incremental installation
checks use the latter. Ztt profiles remain opt-in through `ztt_profiles.json`.
Quick probes do not replace full source compilation or numerical kernel tests.

```bash
python3 scripts/compile_isolation/check.py --list
python3 scripts/compile_isolation/check.py --full --dry-run
```

These commands show the plans without invoking compilers or writing files.
`--help` groups the CLI options. Profiles run sequentially; `-j8` / `--jobs 8`
controls parallel probes, source compilation, ELF inspection and per-header
compilation. The default is up to eight jobs.

## Profiles and toolchains

`--build-type Debug|Release` selects O0/debug checks or O2/NDEBUG consistently for
standalone probes and core builds. `--freestanding AUTO|ON|OFF` selects the core
environment; `ON` also applies freestanding flags to standalone probes. Configuration
reuse includes the environment and build type. `cmake-config` also checks mode
changes in one cache, source-list stability, core-only Threads isolation and mixed
multi-config rejection (requires Ninja).

```bash
# All selected sources, no initializer sections, explicit runtime closure and traps.
python3 scripts/compile_isolation/check.py --profile rvv-fp16-bf16 \
    --suite global-init --suite runtime-contract --suite install-package \
    --freestanding ON --build-type Debug \
    --qemu qemu-riscv64 --qemu-cpu rv64,v=false -j8
```

`runtime-contract` records archive external symbols. In freestanding mode it
checks an explicit memory/compiler-helper allowlist, links the entire archive
with a C driver, `-nostdlib`, real test memory functions and explicit `-lgcc`, and
checks the final ELF for unresolved symbols, initializer sections and dynamic
dependencies. The linker map records helper providers. Its low-ISA startup does
not traverse constructors. It also tests macro diagnostics, memory/im2col results,
assertion evaluation and deliberate Debug/ALWAYS traps. Execution without `--qemu`
reports SKIP; QEMU proves these runtime checks, not native performance or a device
loader. Hosted archives may retain their existing C/C++ runtime dependencies.

In both modes, a target-ISA C++ probe calls every enabled common geometry getter
(RVV and/or Ztt). ELF inspection requires no load-time initializer sections;
hosted probes must retain one lazy-static guard per getter and C++ guard services,
while freestanding probes must have no guards or external runtime calls. This is
compile/symbol evidence, not a concurrent-first-use execution test. Hosted builds
must preserve the compiler's default thread-safe static initialization.

Use `--build-type Debug` and `--build-type Release` in separate invocations
when validating both runtime modes.
The original `--quick` and `--full` plans do not select `runtime-contract`; request
it explicitly when validating the execution-environment contract.

Freestanding installation checks use the same minimal final-link rules for C and
C++ consumers, including relocated CMake imports. Hosted installation checks keep
their existing C++ driver and `-lm`. The installed binary's environment does not
force the consumer's compilation mode. Runtime fixtures live in this directory
and are test-platform code, not implementations installed with the library.

| Runtime file | Responsibility |
| --- | --- |
| `runtime.py` | Symbol policy, minimal links, macro/configuration checks and execution results. |
| `runtime_probe.cpp` | Shared memory, im2col, assertion and clamp checks, plus ISA-guarded backend helper checks. |
| `runtime_memory.c` | Independent C implementations of the four memory services used by minimal links and installed consumers. |
| `runtime_start.S` | Linux user-mode entry and exit without libc startup or constructor traversal. |

The common fixture is built with baseline consumer flags. Profiles that need
backend helper checks build the same file again with target flags; currently
Ztt checks the F32, F16 and BF16 family-local scalar min/max helpers under its
ISA guard, including signed zeros, subnormals, infinities and NaNs. UDS128/A4
checks F32 only because that profile excludes the 16-bit kernels. Both the
scalar-F and bitwise implementations are exercised. Both use the same minimal
link and execution path. The Ztt check calls no matrix instructions and does not
require AME state. It checks the existing scalar compatibility implementation,
not a separate runtime model for Ztt.

| Profile | Target |
| --- | --- |
| `rv64-scalar` | `rv64gc`: scalar RISC-V, including scalar floating point; no vector or matrix kernels. |
| `rvv` | `rv64gcv`, without optional FP16/BF16 arithmetic. |
| `rvv-fp16` | RVV + `zfh_zvfh`, without BF16 arithmetic. |
| `rvv-bf16` | RVV + `zvfbfwma`, without vector FP16 conversions. |
| `rvv-bf16-zvfhmin` | RVV + `zvfbfwma_zvfhmin`, without full vector FP16 arithmetic. |
| `rvv-fp16-bf16` | RVV + `zfh_zvfh_zvfbfwma`. |

`profiles.json` defines flags, required/forbidden compiler macros, and selected
independent `capabilities` expectations. Capability keys are `TORQ_TILE_HAS_`
suffixes with integer 0/1 values. `source-selection` checks both the generated
macro and actual source-group membership: required groups must be complete;
forbidden groups must contribute no sources, even if their macro is 0. `RVV` and
`ZTT` describe any selected backend implementation. These assertions catch an
accidentally disabled group even when the manifest and generated header agree.
They intentionally cover key boundaries rather than list every source file.

`ztt_profiles.json` is a separate opt-in matrix for a Ztt-capable toolchain:
UDS8/16/32/64 with A4, plus UDS128 with A4 and A8. The UDS128 A4 profile
requires F16/BF16 exclusion and F32 inclusion. UDS128 A8 requires all three
datatype groups. F32 uses one capability group across these profiles. Use explicit
`--profile ztt-uds128-a8 --profiles-file scripts/compile_isolation/ztt_profiles.json`.
The `cmake-config` suite and the RVV downgrade part of `install-incremental`
report SKIP for these profiles; source selection, C/C++ header consumers,
installation, compilation and global-initializer checks still apply.

`rv64-scalar` has `expect_empty: true`. Its source-selection check
must fail configuration with `No buildable kernels`; an unrelated failure does
not pass. Library-dependent suites reduce to this rejection check, visible in
`--dry-run`. This profile does not provide a scalar fallback library. Quick only runs
their probes, so it does not check CMake's empty-build rejection.

Requirements: Python 3.8+, RISC-V C/C++ compilers; CMake and Make for build suites;
ELF `readelf` for `global-init`. Non-RISC-V native compilers are outside the check
scope; cross-compiling on an x86 host is supported. Defaults are
`riscv64-unknown-linux-gnu-gcc` and `riscv64-unknown-linux-gnu-g++` from PATH:

```bash
python3 scripts/compile_isolation/check.py --quick \
    --cc /path/to/toolchain/bin/riscv64-unknown-linux-gnu-gcc \
    --cxx /path/to/toolchain/bin/riscv64-unknown-linux-gnu-g++
```

For Clang, put target/sysroot/toolchain arguments in `--flags='...'`; ISA flags
belong in the profile or single-profile `--arch-flags='...'`. Macro and capability
assertions still apply to overrides. Never define/undefine compiler-owned
`__riscv*` macros. `--c-flags` and `--cxx-flags` add language-specific non-ISA flags.
Consumer flags default to `-march=rv64gc -mabi=lp64d` (`--consumer-flags`).
LTO (`-flto` / `-flto=...`) is unsupported and rejected, even when combined with
`-fno-lto`: object-level checks require native machine code. Source compilation
also rejects LTO in actual CMake commands; initializer checks reject LTO sections.

`--profiles-file` accepts the same JSON schema for another toolchain. Each profile
requires `flags`, `required`, `forbidden` string lists, with optional boolean
`expect_empty` and a `capabilities` object. Empty profiles cannot require a
capability header. Built-in plans need their named profiles; custom subsets can
use explicit `--profile`. This tool does not translate vendor extensions or
automatically test multiple compilers. A GCC result does not establish Clang
coverage.

## Suites, dependencies and cost

Costs below are rough orders of magnitude for one profile at about eight jobs,
not timing guarantees. A small source set is much cheaper than a full CPU build.

| Suite | Checks | Prerequisites and approximate cost |
| --- | --- | --- |
| `compile-probes` | GNU C11/C++17 common/conv/F32 declarations; helper includes; runtime-sized e8/e16/e32 pack calls; a BF16 packed kernel where enabled; explicit unsupported-backend guard diagnostics. | No CMake. Seconds to tens of seconds. |
| `source-selection` | Fresh CMake manifest, backend exclusion, generated capabilities against parsed source groups and independent profile expectations; empty-build rejection where expected. | CMake performs compilation probes. Seconds to tens of seconds. |
| `source-compile` | Every selected `.cpp` with its actual CMake command; detects ISA requirements missed by selection. | Runs source-selection first. Tens of seconds to minutes. |
| `global-init` | Rejects `.init_array`, `.preinit_array`, `.ctors` and suffixed variants in compiled objects. | Runs source-selection/source-compile if needed. Scanning adds seconds; standalone cost includes compilation. |
| `cmake-config` | C/C++ ISA flags disagree in both directions; same-cache high-to-baseline and hosted/freestanding reconfiguration; compare sources/capabilities with controls; reject mode conflicts and mixed multi-config environments; check core-only Threads isolation. | Requires a higher-ISA RVV profile; Ninja for multi-config checks. Multiple configurations: tens of seconds to minute scale. |
| `runtime-contract` | Archive symbols, C/C++ macro diagnostics and geometry getter guard policy in both modes; freestanding whole-archive minimal link, memory/im2col/clamp semantics and Debug/ALWAYS traps. | Runs source-selection/source-compile and builds a static core. Seconds after objects exist; QEMU is required for execution cases, otherwise they report SKIP. |
| `install-package` | Build/install, move prefix, check exact public header closure/content/config; compile each API/interface separately in C/C++; link all extracted public functions directly and with find_package, using baseline consumer flags and installed includes. | Runs source-selection and builds the library. Seconds to tens of seconds after objects exist; a fresh build can take minutes. |
| `install-incremental` | In a private source copy, downgrade to RVV and build/install; discover a new public include on ordinary build; preserve unchanged generated-header timestamps; reject missing dependencies. | Runs source-selection; requires a profile whose public header set strictly contains baseline RVV's. Tens of seconds to minutes. |

`global-init` detects load-time dynamic initialization sections, not ordinary
static constants or function-local lazy initialization. It does not inspect CSR
execution or run the loader. Incremental checks configure the higher ISA before
downgrading; they do not first build a high-ISA library. Each install uses a new
prefix; removal of stale files from an overlaid old prefix is not tested.
Installation checks require the library's resolved path (including symlinks) to
stay inside the relocated prefix. Direct consumers link that absolute path, and
the CMake imported target must resolve to the same installed library.

Within one run, source compilation uses CMake object paths so installation can
reuse those objects. Configuration checks can reuse a validated fresh baseline
RVV control only when the source root, compilers, flags and CMake options match.
When `cmake-config` and `install-incremental` run together, the latter performs the
same-cache downgrade once and compares its source selection with the independent
fresh control. Running `cmake-config` alone still checks the downgrade. There is
no cross-run cache or cross-profile parallelism.

For narrow changes:

```bash
# Compile actual raw-bit pack calls.
python3 scripts/compile_isolation/check.py --profile rvv --suite compile-probes \
    --probe-filter 'rvv-e(8|16|32)-call'

# Compile one source family.
python3 scripts/compile_isolation/check.py --profile rvv-fp16 --suite source-compile \
    --source-filter '/gemm_f16_f16/' -j8

# Validate shared-library installation explicitly (default builds are static).
python3 scripts/compile_isolation/check.py --profile rvv --suite install-package \
    --cmake-arg=-DTORQ_TILE_BUILD_SHARED=ON -j8
```

Filters mark the report as filtered. `--source-filter` limits source/global-init
checks only; selection, package and incremental checks remain complete. Full
means the configured matrix, not every ISA combination, compiler version or build
mode. Header closure follows literal local includes, including conditional
branches; it is not a preprocessor. Symbol extraction follows the current public
declaration grammar, not a general C/C++ parser. Native-type APIs still require
consumer compiler type support. Compile/link checks do not establish numerical
correctness, runtime dispatch or actual load-time ISA safety. With `--qemu`,
`runtime-contract` additionally executes its bounded helper fixtures and a low-ISA
whole-archive startup check. These do not cover complete kernel numerics, device
loaders, concurrent first use or native performance.

## Results and evidence

The console prints stages, counts, timings, progress and every FAIL/SKIP. Use `-v`
for every PASS. Each run creates a new `build/compile-isolation/run-*` directory,
or use `--output-dir /path/to/new-directory`. Existing output directories are
refused; output must be outside the input `src/` and `cmake/` trees.

Nothing is cleaned up automatically. Packages remain under
`<run>/<profile>/install-package/relocated/`; incremental copies and prefixes are
under `<run>/<profile>/install-incremental/`. Historical evidence keeps its old
profile/suite names and paths.

- `invocation.json`: command, profile definitions, flags and resolved plan.
- `results.json`: counts, exit code, cases, total/per-stage timings, filtered flag.
- `README.md`: human-readable results and timings.
- Per-case evidence: exact command/cwd, log, return code/duration, manifests,
  generated consumers, objects and libraries.

Exit 0 means all requested checks passed; 1 means FAIL; 2 means invalid CLI use or
incomplete coverage (SKIP without FAIL). Unsupported compiler profiles are SKIP;
macro mismatches, capability assertion violations and real compilation errors are
FAIL. `--allow-skip` is only for exploration, not admission. Expected empty-build
rejection is PASS. `--timeout` bounds each subprocess and its children. Ctrl-C
cancels active subprocess groups before joining workers.

## Implementation layout

`check.py` is the direct Python entry point; it needs no shell wrapper.

| File | Responsibility |
| --- | --- |
| `check.py` | CLI, plan validation and suite orchestration. |
| `common.py` | Toolchain, subprocesses/timeouts, parallelism, evidence and reports. |
| `compilation.py` | Probes, source compilation/selection, capability assertions and configuration checks. |
| `initialization.py` | ELF initialization-section inspection. |
| `installation.py` | Installed headers, relocation and external consumers. |
| `incremental.py` | Private source-copy regeneration and reduced installation. |
| `runtime.py` | Runtime symbols, geometry getter guards, macro/configuration policy, minimal links and opt-in execution. |
| `profiles.json`, `presets.json` | ISA contracts and the two built-in quick/full plans. |
| `pack_rvv.cpp` | Compile probes that instantiate packing operations. |
