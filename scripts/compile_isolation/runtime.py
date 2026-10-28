#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Freestanding symbol, minimal-link and runtime contract checks."""

import re
import shlex
import signal
import shutil

from common import HERE, LICENSE, source_path, write_json


MEMORY_SYMBOLS = {"memcpy", "memmove", "memset", "memcmp"}
# Explicit compiler helpers, resolved by the selected toolchain's libgcc at final link.
COMPILER_SYMBOLS = {
    "__extendhfsf2", "__extendhfdf2", "__truncsfhf2", "__truncdfhf2",
    "__extendbfsf2", "__extendbfdf2", "__truncsfbf2", "__truncdfbf2",
    "__muldi3", "__divdi3", "__udivdi3", "__moddi3", "__umoddi3",
    "__multi3", "__divti3", "__udivti3", "__modti3", "__umodti3",
    "__clzdi2", "__ctzdi2",
}


def is_freestanding(build):
    return bool(re.search(r"^#define TORQ_TILE_BUILT_FREESTANDING 1$",
                         (build / "include/torq_tile/tqt_compile_config.h").read_text(), re.M))


def symbols(log):
    defined, undefined = set(), set()
    for line in log.splitlines():
        fields = line.split()
        if len(fields) < 8 or not re.fullmatch(r"\d+:", fields[0]):
            continue
        section, symbol = fields[6:8]
        (undefined if section == "UND" else defined).add(symbol)
    return defined, undefined


def inspect_elf(ctx, name, directory, executable):
    code, log = ctx.command(directory, "elf", [ctx.args.readelf, "-W", "-S", "-s", "-d",
                                               str(executable)])
    _, undefined = symbols(log)
    initializers = re.findall(r"\]\s+(\.(?:init_array|preinit_array|ctors)(?:\.\S+)?)\s", log)
    needed = re.findall(r"\(NEEDED\).*", log)
    passed = not code and not undefined and not initializers and not needed
    ctx.record(name, "PASS" if passed else "FAIL",
               f"undefined={sorted(undefined)}, initializers={initializers}, needed={needed}", directory)
    return passed


def minimal_link(ctx, directory, sources, flags, library=None, include=None, extra=()):
    """Use a C driver, real memory services and an explicit compiler runtime only."""
    objects = []
    sources = list(sources) + [HERE / "runtime_memory.c", HERE / "runtime_start.S"]
    for source in sources:
        language = "c++" if source.suffix == ".cpp" else "c"
        options = ["-ffreestanding", "-fno-builtin"]
        if language == "c++":
            options += ["-fno-exceptions", "-fno-rtti"]
        subdir = directory / source.stem
        code, _ = ctx.compile(subdir, source, flags, language, include or ctx.root,
                              [*options, *extra])
        if code:
            return None
        objects.append(str(subdir / "probe.o"))
    executable = directory / "consumer.elf"
    libraries = (["-Wl,--whole-archive", str(library), "-Wl,--no-whole-archive"]
                 if library else [])
    code, _ = ctx.command(directory, "link", [ctx.cc, *ctx.extra, *ctx.c_extra, *flags,
                          "-nostdlib", "-static", "-no-pie", "-Wl,-e,_start",
                          "-Wl,--build-id=none", *objects, *libraries, "-lgcc",
                          "-Wl,-Map," + str(directory / "link.map"), "-o", str(executable)])
    return executable if not code else None


def execute(ctx, name, directory, executable, trap=False):
    if not ctx.args.qemu:
        ctx.record(name, "SKIP", "provide --qemu for execution evidence", directory)
        return
    code, _ = ctx.command(directory, "run", [ctx.args.qemu, "-cpu", ctx.args.qemu_cpu,
                                             str(executable)])
    trap_statuses = (-signal.SIGILL, -signal.SIGTRAP, 128 + signal.SIGILL, 128 + signal.SIGTRAP)
    passed = code in trap_statuses if trap else code == 0
    ctx.record(name, "PASS" if passed else "FAIL",
               f"exit={code}; expected {'SIGILL/SIGTRAP' if trap else '0'}", directory)


def macro_checks(ctx, name, flags):
    cases = [
        ("hosted-implicit", [], None),
        ("hosted-zero", ["-DTORQ_TILE_FREESTANDING=0"], None),
        ("free-implicit", ["-ffreestanding"], None),
        ("free-one", ["-ffreestanding", "-DTORQ_TILE_FREESTANDING=1"], None),
        ("hosted-conflict", ["-DTORQ_TILE_FREESTANDING=1"], "must match"),
        ("free-conflict", ["-ffreestanding", "-DTORQ_TILE_FREESTANDING=0"], "must match"),
        ("invalid-on", ["-DTORQ_TILE_FREESTANDING=ON"], "numeric literal"),
        ("invalid-auto", ["-DTORQ_TILE_FREESTANDING=AUTO"], "numeric literal"),
        ("invalid-two", ["-DTORQ_TILE_FREESTANDING=2"], "numeric literal"),
        ("invalid-expression", ["-DTORQ_TILE_FREESTANDING=1+0"], "numeric literal|invalid suffix"),
        ("sdk-injection", ["-DTORQ_TILE_PLATFORM_HEADER=unused"], "Remove TORQ_TILE_PLATFORM_HEADER"),
        ("free-abort", ["-ffreestanding", "-DTORQ_TILE_ERROR_TRAP=0"], "require TORQ_TILE_ERROR_TRAP=1"),
    ]
    for case, options, diagnostic in cases:
        for language, suffix in (("c", "c"), ("c++", "cpp")):
            directory = ctx.output / name / "runtime-contract" / "macros" / case / suffix
            directory.mkdir(parents=True)
            source = directory / ("probe." + suffix)
            source.write_text(LICENSE + '#include "src/common/tqt_common.h"\n')
            compiler = ctx.cc if language == "c" else ctx.cxx
            lang_flags = ctx.c_extra if language == "c" else ctx.cxx_extra
            # Select each environment independently; GCC and Clang differ in their
            # spelling of the inverse freestanding flag for C++.
            inherited = ctx.extra + lang_flags + ctx.probe_options(language)
            inherited = [x for x in inherited if x not in
                         ("-ffreestanding", "-fno-freestanding", "-fhosted")]
            code, log = ctx.command(directory, "compile", [compiler, *inherited, *flags,
                                    "-std=gnu11" if language == "c" else "-std=c++17",
                                    *options, "-I", str(ctx.root), "-c", str(source),
                                    "-o", str(directory / "probe.o")])
            passed = code == 0 if diagnostic is None else (
                code not in (0, 124, 127) and re.search(diagnostic, log))
            ctx.record(f"{name}/runtime-contract/macros/{case}/{suffix}",
                       "PASS" if passed else "FAIL", diagnostic or "accepted", directory)


def getter_policy(ctx, name, flags, free):
    """Check lazy hosted guards and freestanding closure without executing CSRs."""
    directory = ctx.output / name / "runtime-contract" / "getter-policy"
    directory.mkdir(parents=True)
    source = directory / "getters.cpp"
    body = [LICENSE, '#include "src/common/tqt_common.h"\n']
    for guard, getter in (("__riscv_vector", "vlenb"),
                          ("__riscv_ztt", "amenlen"),
                          ("__riscv_ztt", "ameudsz")):
        body.append(f'#if defined({guard})\n'
                    f'extern "C" size_t tqt_check_getter_{getter}(void)\n'
                    '{\n'
                    f'    return tqt_get_{getter}();\n'
                    '}\n#endif\n')
    source.write_text("".join(body))
    options = ["-ffreestanding", "-fno-exceptions", "-fno-rtti"] if free else []
    code, _ = ctx.compile(directory, source, flags, "c++", ctx.root, options)
    if code:
        ctx.record(f"{name}/runtime-contract/getter-policy", "FAIL",
                   "geometry getter probe failed to compile", directory)
        return
    code, log = ctx.command(directory, "elf", [ctx.args.readelf, "-W", "-S", "-s",
                                               str(directory / "probe.o")])
    defined, undefined = symbols(log)
    getters = sorted(x for x in defined if x.startswith("tqt_check_getter_"))
    guards = sorted(x for x in defined if x.startswith("_ZGV"))
    initializers = re.findall(r"\]\s+(\.(?:init_array|preinit_array|ctors)(?:\.\S+)?)\s", log)
    if free:
        policy = not guards and not (undefined - {"_GLOBAL_OFFSET_TABLE_"})
    else:
        policy = len(guards) == len(getters) and {
            "__cxa_guard_acquire", "__cxa_guard_release"}.issubset(undefined)
    passed = not code and bool(getters) and not initializers and policy
    ctx.record(f"{name}/runtime-contract/getter-policy", "PASS" if passed else "FAIL",
               f"freestanding={free}, getters={getters}, guards={len(guards)}, "
               f"initializers={initializers}, undefined={sorted(undefined)}", directory)


def configuration(ctx, name, flags, reference):
    """Exercise mode changes in one cache and check source selection independently."""
    directory = ctx.output / name / "cmake-config" / "runtime"
    expected_sources = {str(source_path(x)) for x in reference}
    cases = [("ON", "", 1, None), ("OFF", "", 0, None),
             ("AUTO", "-ffreestanding", 1, None), ("AUTO", "", 0, None),
             ("OFF", "-ffreestanding", None, "conflicts with"),
             ("INVALID", "", None, "must be AUTO, ON or OFF")]
    for index, (option, compiler_mode, expected, diagnostic) in enumerate(cases):
        base_flags = [x for x in ctx.extra + ctx.cxx_extra + flags if x not in
                      ("-ffreestanding", "-fno-freestanding", "-fhosted")]
        options = [f"-DTORQ_TILE_FREESTANDING={option}",
                   "-DCMAKE_CXX_FLAGS=" + shlex.join(base_flags + ([compiler_mode] if compiler_mode else []))]
        code, log = ctx.configure(directory, flags, options)
        evidence = directory / f"case-{index}-{option}"
        evidence.mkdir()
        for filename in ("configure.log", "configure.command.json", "CMakeCache.txt"):
            if (directory / filename).exists():
                shutil.copy2(directory / filename, evidence / filename)
        if diagnostic:
            passed = code not in (0, 124, 127) and diagnostic in log
        else:
            actual_sources = set() if code else {str(source_path(x)) for x in ctx.entries(directory)}
            passed = not code and actual_sources == expected_sources and is_freestanding(directory) == bool(expected)
            if not code:
                shutil.copy2(directory / "include/torq_tile/tqt_compile_config.h", evidence)
                shutil.copy2(directory / "compile_commands.json", evidence)
                cache = (directory / "CMakeCache.txt").read_text()
                passed = passed and not re.search(r"^(?:CMAKE_HAVE_LIBC_PTHREAD|CMAKE_THREAD_LIBS_INIT):", cache, re.M)
        ctx.record(f"{name}/cmake-config/runtime/{index}-{option}", "PASS" if passed else "FAIL",
                   diagnostic or f"build mode={expected}, source list preserved, no core Threads probe", evidence)
    ordering = ctx.output / name / "cmake-config" / "runtime-option-order"
    code, _ = ctx.configure(ordering, flags, ["-DTORQ_TILE_FREESTANDING=ON",
                            f"-DCMAKE_CXX_FLAGS_{ctx.args.build_type.upper()}=-fexceptions -frtti"])
    actual_sources = set() if code else {str(source_path(x)) for x in ctx.entries(ordering)}
    passed = not code and is_freestanding(ordering) and actual_sources == expected_sources
    ctx.record(f"{name}/cmake-config/runtime/option-order", "PASS" if passed else "FAIL",
               "private core options follow configuration flags in both probes and targets", ordering)
    mixed = ctx.output / name / "cmake-config" / "mixed-runtime"
    if not shutil.which("ninja"):
        ctx.record(f"{name}/cmake-config/runtime/mixed", "SKIP", "Ninja required for multi-config check", directory)
        return
    code, log = ctx.configure(mixed, flags, ["-DTORQ_TILE_FREESTANDING=AUTO",
                              "-DCMAKE_CONFIGURATION_TYPES=Debug;Release",
                              "-DCMAKE_CXX_FLAGS_DEBUG=-O0 -ffreestanding",
                              "-DCMAKE_CXX_FLAGS=" + shlex.join(base_flags),
                              "-DCMAKE_CXX_FLAGS_RELEASE=-O2"], generator="Ninja Multi-Config")
    passed = code not in (0, 124, 127) and "Mixed hosted/freestanding configurations" in log
    ctx.record(f"{name}/cmake-config/runtime/mixed", "PASS" if passed else "FAIL",
               "mixed compiler environments must be rejected", mixed)


def run(ctx, name, flags, entries, objects):
    directory = ctx.output / name / "runtime-contract"
    build = ctx.output / name / "build"
    if any(item["returncode"] for item in objects):
        ctx.record(f"{name}/runtime-contract", "FAIL", "core compilation failed", directory)
        return
    code, _ = ctx.command(directory, "build", ["cmake", "--build", str(build),
                                               "--target", "torq_tile", "-j", str(ctx.args.jobs)])
    library = build / "libtorq_tile.a"
    if code or not library.is_file():
        ctx.record(f"{name}/runtime-contract", "FAIL", "requires a complete static core", directory)
        return
    code, log = ctx.command(directory, "symbols", [ctx.args.readelf, "-Ws", str(library)])
    defined, undefined = symbols(log)
    external = undefined - defined
    free = is_freestanding(build)
    if ctx.args.freestanding != "AUTO" and free != (ctx.args.freestanding == "ON"):
        ctx.record(f"{name}/runtime-contract/environment", "FAIL",
                   "binary environment metadata disagrees with the requested mode", directory)
        return
    unexpected = external - MEMORY_SYMBOLS - COMPILER_SYMBOLS - {"_GLOBAL_OFFSET_TABLE_"}
    providers = {symbol: "runtime_memory.c" if symbol in MEMORY_SYMBOLS else
                 "linker" if symbol == "_GLOBAL_OFFSET_TABLE_" else "libgcc (see link.map)"
                 for symbol in sorted(external - unexpected)}
    write_json(directory / "runtime-symbols.json",
               dict(freestanding=free, external=sorted(external), unexpected=sorted(unexpected),
                    providers=providers))
    ctx.record(f"{name}/runtime-contract/symbols", "FAIL" if code or (free and unexpected) else "PASS",
               f"freestanding={free}, external={sorted(external)}", directory)
    low = shlex.split(ctx.args.consumer_flags)
    macro_checks(ctx, name, low)
    getter_policy(ctx, name, flags, free)
    if not free:
        return
    empty = directory / "addresses.cpp"
    empty.write_text(LICENSE + 'extern "C" int main() { return 0; }\n')
    executable = minimal_link(ctx, directory / "whole-archive", [empty], low, library)
    if not executable:
        ctx.record(f"{name}/runtime-contract/whole-archive", "FAIL", "minimal C-driver link failed", directory)
    else:
        inspect_elf(ctx, f"{name}/runtime-contract/whole-archive", executable.parent, executable)
        execute(ctx, f"{name}/runtime-contract/load-isolation", executable.parent, executable)
    for failure in range(4):
        case = directory / ("semantics" if not failure else f"failure-{failure}")
        executable = minimal_link(ctx, case, [HERE / "runtime_probe.cpp"], low,
                                  extra=[f"-DTQT_CHECK_FAILURE={failure}"])
        if not executable:
            ctx.record(f"{name}/runtime-contract/{case.name}", "FAIL", "fixture link failed", case)
            continue
        inspect_elf(ctx, f"{name}/runtime-contract/{case.name}/elf", case, executable)
        trap = failure == 2 or (failure in (1, 3) and ctx.args.build_type == "Debug")
        execute(ctx, f"{name}/runtime-contract/{case.name}/run", case, executable, trap)
    if any(source_path(entry).stem.endswith("_ztt") for entry in entries):
        case = directory / "ztt-number-semantics"
        # Reuse the common fixture with target flags to enable backend helper checks.
        executable = minimal_link(ctx, case, [HERE / "runtime_probe.cpp"], flags)
        if not executable:
            ctx.record(f"{name}/runtime-contract/ztt-number-semantics", "FAIL", "fixture link failed", case)
        else:
            inspect_elf(ctx, f"{name}/runtime-contract/ztt-number-semantics/elf", case, executable)
            execute(ctx, f"{name}/runtime-contract/ztt-number-semantics/run", case, executable)
