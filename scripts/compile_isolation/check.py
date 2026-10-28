#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Standalone ISA isolation checks with quick, full and explicit plans."""

import argparse
import json
import os
from pathlib import Path
import re
import resource
import shlex
import sys
import tempfile
import time
import traceback

# Keep --list/--dry-run free of bytecode files in the source tree.
sys.dont_write_bytecode = True

import compilation
import initialization
import installation
import incremental
import runtime
from common import Context, HERE, SUITES, lto_flags, within, write_json


def main():
    parser = argparse.ArgumentParser(prog="scripts/compile_isolation/check.py",
        usage="%(prog)s [--quick | --full | --profile NAME] [options]",
        description="ISA isolation checks with optional freestanding runtime execution.",
        epilog="Examples:\n"
               "  python3 %(prog)s\n"
               "  time python3 %(prog)s --full -j8\n"
               "  python3 %(prog)s --profile rvv --suite install-package -j8\n"
               "  python3 %(prog)s --profile rvv-fp16 --suite source-compile -j8",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    basic = parser.add_argument_group("Check selection")
    selection = basic.add_mutually_exclusive_group()
    selection.add_argument("--quick", dest="preset", action="store_const", const="quick",
                           help="fast header/packing probes; no CMake build (default)")
    selection.add_argument("--full", dest="preset", action="store_const", const="full",
                           help="complete curated development matrix")
    selection.add_argument("--profile", action="append",
                           help="repeatable explicit profile; default suite: compile-probes")
    basic.add_argument("--suite", action="append", choices=(*SUITES, "all"),
                        help="repeatable; overrides the preset suites if supplied")
    basic.add_argument("-j", "--jobs", type=int, default=min(8, os.cpu_count() or 1),
                       help="parallel job limit (default: %(default)s); profiles run sequentially")
    basic.add_argument("--list", action="store_true", help="list presets, profiles and suites without running tools")
    basic.add_argument("--dry-run", action="store_true", help="preview exact plan; no tools or output files")
    basic.add_argument("--probe-filter", default=".", help="regex on probe names only; marks the run as filtered")
    basic.add_argument("--source-filter", default=".",
                       help="regex for source-compile/global-init; installation stays complete")
    basic.add_argument("--build-type", choices=("Debug", "Release"), default="Release")
    basic.add_argument("--freestanding", choices=("AUTO", "ON", "OFF"), default="AUTO",
                       help="core environment; ON also applies to standalone probes")
    toolchain = parser.add_argument_group("Toolchain (executables must be on PATH or absolute paths)")
    toolchain.add_argument("--cc", default="riscv64-unknown-linux-gnu-gcc", help="C compiler (default: %(default)s)")
    toolchain.add_argument("--cxx", default="riscv64-unknown-linux-gnu-g++", help="C++ compiler (default: %(default)s)")
    toolchain.add_argument("--flags", default="", help="extra shared flags, e.g. --sysroot/--target; use --flags='...'")
    toolchain.add_argument("--c-flags", default="")
    toolchain.add_argument("--cxx-flags", default="")
    toolchain.add_argument("--arch-flags", help="replace ISA flags for one profile; macro assertions still apply")
    toolchain.add_argument("--consumer-flags", default="-march=rv64gc -mabi=lp64d",
                          help="installed consumer flags (default: %(default)s)")
    advanced = parser.add_argument_group("Evidence and advanced options")
    advanced.add_argument("--profiles-file", type=Path, default=HERE / "profiles.json")
    advanced.add_argument("--cmake-arg", action="append", default=[], help="extra CMake argument; use --cmake-arg=-D...")
    advanced.add_argument("--timeout", type=int, default=300, help="per-command timeout in seconds (default: %(default)s)")
    advanced.add_argument("--readelf", default="readelf")
    advanced.add_argument("--qemu", help="optional RISC-V runner for runtime-contract executables")
    advanced.add_argument("--qemu-cpu", default="rv64", help="baseline CPU for runtime-contract")
    advanced.add_argument("--output-dir", type=Path, help="new output directory; logs/builds/install prefixes are retained")
    advanced.add_argument("--source-root", type=Path, default=HERE.parent.parent)
    advanced.add_argument("--allow-skip", action="store_true", help="exploration only: permit SKIP exit status 0")
    advanced.add_argument("-v", "--verbose", action="store_true", help="print every PASS (failures always print)")
    args = parser.parse_args()
    if args.qemu:
        # Expected assertion traps must not leave guest/host core dumps behind.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        profiles = json.loads(args.profiles_file.read_text())
        presets = json.loads((HERE / "presets.json").read_text())
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("profiles file must contain a non-empty object")
        if not isinstance(presets, dict) or set(presets) != {"quick", "full"}:
            raise ValueError("presets file must contain exactly quick and full")
        for groups in presets.values():
            if not isinstance(groups, list) or not groups:
                raise ValueError("each preset needs at least one group")
            for group in groups:
                if not isinstance(group, dict) or any(
                        not isinstance(group.get(key), list) or not group[key] or
                        not all(isinstance(x, str) for x in group[key]) for key in ("profiles", "suites")):
                    raise ValueError("preset groups need non-empty profiles and suites lists")
                if set(group["suites"]) - set(SUITES) - {"all"}:
                    raise ValueError("unknown suite in presets file")
        for profile in profiles.values():
            if not isinstance(profile, dict) or any(
                    not isinstance(profile.get(key), list) or
                    not all(isinstance(x, str) for x in profile[key])
                    for key in ("flags", "required", "forbidden")):
                raise ValueError("each profile needs flags/required/forbidden string lists")
            if not isinstance(profile.get("expect_empty", False), bool):
                raise ValueError("expect_empty must be a boolean")
            capabilities = profile.get("capabilities", {})
            if not isinstance(capabilities, dict) or any(
                    not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or
                    type(value) is not int or value not in (0, 1)
                    for key, value in capabilities.items()):
                raise ValueError("capabilities must map HAS_ suffixes to integer 0/1 values")
            if profile.get("expect_empty") and capabilities:
                raise ValueError("empty profiles cannot assert a generated capability header")
        for value in (args.flags, args.c_flags, args.cxx_flags,
                      args.consumer_flags, args.arch_flags or ""):
            shlex.split(value)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.list:
        print("Presets:")
        for name, groups in presets.items():
            print(f"  {name}:")
            for group in groups:
                print(f"    {', '.join(group['profiles'])}: {', '.join(group['suites'])}")
        print("\nProfiles:")
        for name, profile in profiles.items():
            note = " [expect empty-build rejection]" if profile.get("expect_empty") else ""
            print(f"  {name}: {shlex.join(profile['flags'])}{note}")
        print("\nSuites: " + ", ".join(SUITES))
        return 0
    preset = None if args.profile else args.preset or "quick"
    groups = presets[preset] if preset else [dict(profiles=args.profile, suites=["compile-probes"])]
    plan = []
    for group in groups:
        suites = args.suite or group["suites"]
        if "all" in suites:
            suites = SUITES
        suites = [suite for suite in SUITES if suite in suites]
        for name in group["profiles"]:
            plan.append(dict(profile=name, suites=suites,
                             cc=args.cc, cxx=args.cxx, flags=shlex.split(args.flags),
                             c_flags=shlex.split(args.c_flags), cxx_flags=shlex.split(args.cxx_flags)))
    names = [item["profile"] for item in plan]
    if not names or len(set(names)) != len(names) or any(x not in profiles for x in names):
        parser.error("profiles must be known, distinct names")
    if any(not re.fullmatch(r"[a-zA-Z0-9_-]+", x) for x in names):
        parser.error("profile names must contain only letters, digits, underscores and hyphens")
    if args.arch_flags is not None and len(names) != 1:
        parser.error("--arch-flags requires exactly one profile")
    if args.jobs < 1 or args.timeout < 1:
        parser.error("--jobs and --timeout must be positive")
    if not (args.source_root / "src").is_dir():
        parser.error("--source-root must contain a src directory")
    if args.output_dir and any(within(args.output_dir.resolve(), args.source_root.resolve() / folder)
                               for folder in ("src", "cmake")):
        parser.error("--output-dir must be outside src/ and cmake/ (incremental checks copy these inputs)")
    for option in args.cmake_arg:
        if not re.match(r"^-DTORQ_TILE_[A-Z0-9_]+(?::[A-Z]+)?=", option) or re.match(
                r"^-DTORQ_TILE_BUILD_(TEST|BENCHMARK)(?:[:=])", option):
            parser.error("--cmake-arg accepts TORQ_TILE options only, excluding tests/benchmarks")
    if re.search(r"(?:^|\s)-[DU]\s*__riscv", " ".join(
            [args.flags, args.c_flags, args.cxx_flags, args.arch_flags or "", args.consumer_flags] +
            [shlex.join(profiles[name]["flags"]) for name in names])):
        parser.error("do not define or undefine compiler-owned ISA macros")
    if re.search(r"(?:^|\s)-(?:march|mcpu|mabi)=", " ".join(
            [args.flags, args.c_flags, args.cxx_flags])):
        parser.error("ISA flags belong in the profile or --arch-flags")
    try:
        re.compile(args.probe_filter)
        re.compile(args.source_filter)
    except re.error as error:
        parser.error(str(error))
    for item in plan:
        item["arch_flags"] = (shlex.split(args.arch_flags) if args.arch_flags is not None
                              else profiles[item["profile"]]["flags"])
        item["expect_empty"] = profiles[item["profile"]].get("expect_empty", False)
        if item["expect_empty"] and set(item["suites"]) - {"compile-probes"}:
            item["suites"] = [x for x in ("compile-probes", "source-selection") if x == "source-selection" or x in item["suites"]]
    flag_sets = [shlex.split(value) for value in
                 (args.flags, args.c_flags, args.cxx_flags, args.consumer_flags)]
    flag_sets.extend(item["arch_flags"] for item in plan)
    if any(lto_flags(flags) for flags in flag_sets):
        parser.error("LTO is unsupported: remove -flto/-flto=... for object-level ISA checks")
    if args.dry_run:
        print(f"Preset: {preset or 'explicit profiles'}; jobs={args.jobs}")
        for item in plan:
            note = (" (expect empty-build rejection)" if item["expect_empty"] and
                    "source-selection" in item["suites"] else "")
            print(f"  {item['profile']}: {', '.join(item['suites'])}{note}\n"
                  f"    {item['cxx']} {shlex.join(item['flags'] + item['cxx_flags'] + item['arch_flags'])}")
        return 0
    if args.output_dir:
        output = args.output_dir.resolve()
        try:
            output.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            parser.error("--output-dir already exists; choose a new directory")
    else:
        parent = args.source_root.resolve() / "build" / "compile-isolation"
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    checker = Context(args, output)
    write_json(output / "invocation.json", dict(argv=sys.argv, source_root=str(checker.root),
               utc_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               profiles=profiles, preset=preset, plan=plan, environment_removed=[
                   "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH",
                   "CMAKE_PREFIX_PATH", "CMAKE_TOOLCHAIN_FILE", "CFLAGS", "CXXFLAGS",
                   "LIBRARY_PATH", "LDFLAGS"]))
    print(f"Evidence: {output}", flush=True)
    print("Preset: " + (preset or "explicit profiles"), flush=True)
    if args.probe_filter != "." or args.source_filter != ".":
        print("Coverage: FILTERED; this run does not establish the complete preset.", flush=True)
    try:
        for item in plan:
            name = item["profile"]
            suites = set(item["suites"])
            profile = profiles[name]
            flags = item["arch_flags"]
            checker.set_toolchain(item)
            note = ("; expect empty-build rejection" if item["expect_empty"] and
                    "source-selection" in suites else "")
            print(f"[PLAN] {name}: {', '.join(item['suites'])}; {shlex.join(flags)}{note}", flush=True)
            with checker.stage(name, "toolchain"):
                macros = checker.preflight(name, flags, profile)
            if macros is None:
                continue
            if "compile-probes" in suites:
                with checker.stage(name, "compile-probes"):
                    compilation.probes(checker, name, flags, macros)
            if suites - {"compile-probes"}:
                with checker.stage(name, "source-selection"):
                    entries = compilation.selected_build(checker, name, flags, macros,
                                                         item["expect_empty"], profile.get("capabilities", {}))
                if entries is None:
                    for suite in sorted(suites - {"compile-probes", "source-selection"}):
                        checker.record(f"{name}/{suite}", "SKIP", "configuration failed", output / name)
                    continue
                if not entries:
                    continue
                if suites & {"source-compile", "global-init", "runtime-contract"}:
                    with checker.stage(name, "source-compile"):
                        objects = compilation.sources(checker, name, entries)
                    if "global-init" in suites:
                        with checker.stage(name, "global-init"):
                            initialization.run(checker, name, objects)
                    if "runtime-contract" in suites:
                        with checker.stage(name, "runtime-contract"):
                            runtime.run(checker, name, flags, entries, objects)
                control_entries = None
                if "cmake-config" in suites:
                    with checker.stage(name, "cmake-config"):
                        control_entries = compilation.policy(checker, name, flags, macros, entries,
                                                             "install-incremental" in suites)
                if "install-package" in suites:
                    with checker.stage(name, "install-package"):
                        installation.run(checker, name, entries)
                if "install-incremental" in suites:
                    with checker.stage(name, "install-incremental"):
                        incremental.run(checker, name, flags, entries, control_entries)
    except (Exception, KeyboardInterrupt) as error:
        (output / "runner-error.log").write_text(traceback.format_exc())
        checker.record("runner", "FAIL", f"{type(error).__name__}: {error}", output)
    return checker.report()


if __name__ == "__main__":
    sys.exit(main())
