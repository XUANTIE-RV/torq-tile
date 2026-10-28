#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Compilation probes, source selection and CMake policy checks."""

from pathlib import Path
import re
import shlex
import shutil
import runtime

from common import HERE, LICENSE, lto_flags, source_path, write_json


COMMON = "src/common/tqt_common.h"
RELU = "src/activation/relu_f32_f32/relu_f32_f32/tqt_relu_f32_f32_1x8vl_rvv.h"
RVV_BASE_FLAGS = ["-march=rv64gcv", "-mabi=lp64d"]
BACKENDS = {"rvv": "__riscv_v", "ztt": "__riscv_ztt"}
RVV_BASE_CAPABILITIES = dict(RVV=1, RVV_F16=0, RVV_BF16=0, RVV_BF16_TO_F16=0,
                             RVV_F64=1)


def probes(ctx, name, flags, macros):
    cases = [("common-c", COMMON, "c", None, []),
             ("common-cxx", COMMON, "c++", None, []),
             ("conv-c", "src/common/tqt_conv.h", "c", None, []),
             ("conv-cxx", "src/common/tqt_conv.h", "c++", None, []),
             ("public-f32-c", RELU, "c", None, []),
             ("public-f32-cxx", RELU, "c++", None, [])]
    for backend, macro in (("rvv", "__riscv_v"),):
        header = f"src/common/tqt_gemm_pack_{backend}.h"
        if macro in macros:
            cases.append((f"{backend}-include", header, "c++", None, []))
            if backend == "rvv":
                cases.append(("rvv-math-include", "src/common/tqt_math_rvv.h", "c++", None, []))
                cases.append(("rvv-dwconv-include", "src/common/tqt_dwconv_pack_rvv.h", "c++", None, []))
            for bits in (8, 16, 32):
                cases.append((f"{backend}-e{bits}-call", None, "c++",
                              HERE / f"pack_{backend}.cpp", [f"-DTQT_CHECK_BITS={bits}"]))
        else:
            # An explicit architectural diagnostic is required, not just any failure.
            cases.append((f"{backend}-guard", header, "c++", None, ["reject-guard"]))
    if "__riscv_v" in macros and "__riscv_zvfbfwma" in macros:
        cases.append(("bf16-packed-kernel", None, "c++", ctx.root /
                      "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16p_bf16p/"
                      "tqt_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv.cpp", []))
    selected = [x for x in cases if re.search(ctx.args.probe_filter, x[0])]
    if not selected:
        ctx.record(f"{name}/compile-probes", "FAIL", "probe filter matched no cases", ctx.output / name)
    def check(item):
        case, header, language, fixture, extra = item
        directory = ctx.output / name / "compile-probes" / case
        directory.mkdir(parents=True)
        source = fixture or directory / ("probe.cpp" if language == "c++" else "probe.c")
        if header:
            source.write_text(LICENSE + f'#include "{header}"\n')
        reject = extra == ["reject-guard"]
        code, log = ctx.compile(directory, source, flags, language, ctx.root,
                                 [] if reject else extra)
        passed = (code == 0) if not reject else (
            code not in (0, 124, 127) and re.search(r"error[^\n]*This file must be compiled", log))
        ctx.record(f"{name}/compile-probes/{case}", "PASS" if passed else "FAIL",
                    "expected ISA guard rejection" if reject and passed else
                    "object compilation succeeded" if passed else "inspect compile.log", directory)
    ctx.parallel(check, selected)


def selected_build(ctx, name, flags, macros, expect_empty=False, capabilities=None):
    directory = ctx.output / name / "build"
    code, log = ctx.configure(directory, flags)
    if expect_empty:
        passed = code not in (0, 124, 127) and "No buildable kernels:" in log
        ctx.record(f"{name}/source-selection/empty", "PASS" if passed else "FAIL",
                   "expected empty-build diagnostic" if passed else
                   "expected No buildable kernels diagnostic; inspect configure.log", directory)
        return [] if passed else None
    if code:
        ctx.record(f"{name}/source-selection/configure", "FAIL", "inspect configure.log", directory)
        return None
    entries = ctx.entries(directory)
    ctx.record(f"{name}/source-selection/manifest", "PASS" if entries else "FAIL",
                f"{len(entries)} kernel compile commands recorded", directory)
    violations = []
    for entry in entries:
        source = source_path(entry)
        for backend, macro in BACKENDS.items():
            if source.stem.endswith(f"_{backend}") and macro not in macros:
                violations.append(f"{source.relative_to(ctx.root)} requires {macro}")
    write_json(directory / "architecture-violations.json", violations)
    ctx.record(f"{name}/source-selection/architecture", "FAIL" if violations else "PASS",
                f"{len(violations)} sources selected for an unavailable backend; use source-compile for exact ISA checks",
                directory)
    valid_config = compile_config(ctx, name + "/source-selection", directory, entries,
                                  capabilities=capabilities)
    if entries and not violations and valid_config:
        ctx.fresh_configurations[ctx.configuration_key(flags)] = (directory, entries)
    return entries


def compile_config(ctx, name, directory, entries, root=None, capabilities=None):
    root = root or ctx.root
    header = directory / "include/torq_tile/tqt_compile_config.h"
    actual = dict(re.findall(r"^#define TORQ_TILE_HAS_(\w+) ([01])$",
                             header.read_text() if header.exists() else "", re.M))
    selected = {source_path(x).relative_to(root).as_posix() for x in entries}
    expected = {}
    groups = {}
    # Parse the literal source groups independently of CMake's probe results.
    for group, body in re.findall(r"set\(TORQ_TILE_(\w+)_SOURCES\s+(.*?)\)",
                                  re.sub(r"#[^\n]*", "", (root / "CMakeLists.txt").read_text()), re.S):
        files = set(re.findall(r"src/[^\s\"]+\.cpp", body))
        if files:
            groups[group] = files
            expected[group] = str(int(files <= selected))
    for backend in (name.upper() for name in BACKENDS):
        expected[backend] = str(int(any(x.endswith(f"_{backend.lower()}.cpp") for x in selected)))
    passed = bool(actual) and actual == expected
    write_json(directory / "compile-config-check.json", dict(actual=actual, expected=expected))
    ctx.record(f"{name}/compile-config", "PASS" if passed else "FAIL",
               "generated 0/1 capabilities match final source groups" if passed else
               "missing or inconsistent generated compile configuration", directory)
    if capabilities:
        violations = {}
        for group, required in capabilities.items():
            files = ({x for x in selected if x.endswith(f"_{group.lower()}.cpp")}
                     if group.lower() in BACKENDS else groups.get(group, set()))
            available = bool(files) and files <= selected
            absent = not (files & selected)
            if (actual.get(group) != str(required) or
                    not (available if required else absent) or
                    (group.lower() not in BACKENDS and group not in groups)):
                violations[group] = dict(required=required, actual=actual.get(group),
                                         selected=sorted(files & selected), missing=sorted(files - selected))
        write_json(directory / "capability-contract.json",
                   dict(required=capabilities, violations=violations))
        ctx.record(f"{name}/capability-contract", "FAIL" if violations else "PASS",
                   f"{len(violations)} capability expectations violated", directory)
        passed = passed and not violations
    return passed


def sources(ctx, name, entries):
    selected = [x for x in entries if re.search(ctx.args.source_filter, str(source_path(x)))]
    if not selected:
        ctx.record(f"{name}/source-compile", "FAIL", "source filter matched no files", ctx.output / name)

    def check(entry):
        source = source_path(entry)
        relative = source.relative_to(ctx.root).as_posix()
        directory = ctx.output / name / "source-compile" / source.relative_to(ctx.root).with_suffix("")
        argv = entry.get("arguments") or shlex.split(entry["command"])
        # Use CMake's own object path so installation can reuse successful compilation.
        obj = (Path(entry["directory"]) / argv[argv.index("-o") + 1]).resolve()
        unsupported = lto_flags(argv)
        if unsupported:
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / "rejected-command.json", argv)
            code = 1
        else:
            obj.parent.mkdir(parents=True, exist_ok=True)
            code, _ = ctx.command(directory, "compile", argv, Path(entry["directory"]))
        ctx.record(f"{name}/source-compile/{relative}", "PASS" if not code else "FAIL",
                   f"LTO is unsupported: {unsupported}" if unsupported else
                   "compiled with actual target flags" if not code else "inspect compile.log", directory)
        return dict(source=relative, directory=str(directory), object=str(obj), returncode=code)

    objects = ctx.parallel(check, selected)
    write_json(ctx.output / name / "build" / "compiled-objects.json", objects)
    return objects

def rvv_control(ctx, name):
    key = ctx.configuration_key(RVV_BASE_FLAGS)
    cached = ctx.fresh_configurations.get(key)
    if cached:
        directory, entries = cached
    else:
        directory = ctx.output / name / "cmake-config" / "fresh-rvv"
        code, _ = ctx.configure(directory, RVV_BASE_FLAGS)
        if code:
            ctx.record(f"{name}/cmake-config/control", "FAIL", "fresh RVV control failed", directory)
            return None
        entries = ctx.entries(directory)
    valid = compile_config(ctx, name + "/cmake-config/control", directory, entries,
                           capabilities=RVV_BASE_CAPABILITIES)
    if not valid or not entries:
        return None
    ctx.fresh_configurations[key] = (directory, entries)
    ctx.record(f"{name}/cmake-config/control", "PASS",
               "reused matching fresh RVV configuration from this run" if cached else
               "created fresh RVV configuration", directory)
    return entries


def policy(ctx, name, flags, macros, base_entries, share_downgrade=False):
    if "__riscv_v" not in macros or not any(x in macros for x in
            ("__riscv_zvfh", "__riscv_zvfbfwma")):
        ctx.record(f"{name}/cmake-config", "SKIP", "requires a higher-ISA RVV profile; use rvv-fp16-bf16",
                   ctx.output / name)
        return
    # Use a fresh low-ISA build as an independent control for both stale caches and C-only probes.
    low = RVV_BASE_FLAGS
    control_entries = rvv_control(ctx, name)
    if control_entries is None:
        return
    expected = {str(source_path(x)) for x in control_entries}
    # Incremental checks exercise the same-cache downgrade in their private source copy.
    cases = ["c-cxx-flags", "cxx-high-c-low"]
    if not share_downgrade:
        cases.append("cache-downgrade")
    for case in cases:
        directory = ctx.output / name / "cmake-config" / case
        if case == "cache-downgrade":
            code, _ = ctx.configure(directory, flags)
            if code:
                ctx.record(f"{name}/cmake-config/{case}", "FAIL", "initial configure failed", directory)
                continue
            shutil.copy2(directory / "configure.log", directory / "initial-configure.log")
            shutil.copy2(directory / "configure.command.json", directory / "initial-configure.command.json")
            shutil.copy2(directory / "CMakeCache.txt", directory / "initial-CMakeCache.txt")
            shutil.copy2(directory / "include/torq_tile/tqt_compile_config.h",
                         directory / "initial-tqt_compile_config.h")
        reverse = case == "cxx-high-c-low"
        code, _ = ctx.configure(directory, flags if reverse else low,
                                c_flags=low if reverse else flags if case == "c-cxx-flags" else None)
        comparison = {str(source_path(x)) for x in base_entries} if reverse else expected
        entries = [] if code else ctx.entries(directory)
        actual = {str(source_path(x)) for x in entries}
        if not code:
            compile_config(ctx, name + "/cmake-config/" + case, directory, entries)
        write_json(directory / "selection-diff.json", dict(extra=sorted(actual - comparison),
                                                           missing=sorted(comparison - actual)))
        meaningful = "__riscv_zvfh" in macros or "__riscv_zvfbfwma" in macros
        ctx.record(f"{name}/cmake-config/{case}", "FAIL" if code or actual != comparison else
                    "PASS" if meaningful else "SKIP",
                    "requires a higher-ISA profile" if not meaningful and not code and actual == comparison else
                    f"compared with matching C++ flags: {len(actual - comparison)} extra, {len(comparison - actual)} missing",
                    directory)
    runtime.configuration(ctx, name, flags, base_entries)
    return control_entries
