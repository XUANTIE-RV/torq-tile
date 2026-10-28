#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Installed header, relocation and external consumer checks."""

import json
from pathlib import Path
import re
import shlex
import shutil

from common import HERE, LICENSE, source_path, within, write_json
import runtime


def relocated_library(path, prefix):
    """Require an existing library whose resolved path stays inside the moved prefix."""
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"Installed library is missing or not absolute: {path}")
    resolved = path.resolve()
    if not within(resolved, prefix.resolve()):
        raise ValueError(f"Library resolves outside relocated prefix: {path} -> {resolved}")
    return resolved


def header_closure(root, entries):
    """Derive the public contract from actual compile commands, not installed files."""
    public = set()
    for entry in entries:
        source = source_path(entry)
        public.add(source.with_suffix(".h"))
        interface = source.parent / f"tqt_{source.parent.name}_interface.h"
        if interface.exists():
            public.add(interface)
    closure = set(public)
    pending = list(public)
    while pending:
        header = pending.pop()
        for include in re.findall(r'^\s*#\s*include\s*"([^"]+)"', header.read_text(), re.M):
            dep = (root / include if include.startswith("src/") else header.parent / include).resolve()
            if not dep.is_file() or not within(dep, root / "src"):
                raise ValueError(f"Unresolved public dependency: {header}: {include}")
            if dep not in closure:
                closure.add(dep)
                pending.append(dep)
    return public, closure


def check_headers(ctx, name, directory, build, relocated, entries, root=None):
    root = root or ctx.root
    public, closure = header_closure(root, entries)
    expected = {x.relative_to(root / "src").as_posix() for x in closure}
    include_root = relocated / "include"
    installed_root = include_root / "torq_tile"
    installed = {x.relative_to(installed_root).as_posix() for x in installed_root.rglob("*")
                 if x.suffix in (".h", ".hpp")}
    missing = sorted(expected - installed)
    leaked = sorted(installed - expected - {"tqt_compile_config.h"})
    write_json(directory / "header-manifest.json", dict(expected=sorted(expected),
               installed=sorted(installed), missing=missing, unexpected_source_headers=leaked))
    ctx.record(f"{name}/header-set", "FAIL" if missing or leaked else "PASS",
                f"{len(missing)} missing, {len(leaked)} unexpected source headers", directory)
    mismatches = []
    for header in closure:
        relative = header.relative_to(root / "src")
        expected_content = re.sub(r'(^[ \t]*#[ \t]*include[ \t]+")src/', r'\1torq_tile/',
                                  header.read_text(), flags=re.M)
        target = installed_root / relative
        if not target.is_file() or target.read_text() != expected_content:
            mismatches.append(relative.as_posix())
    write_json(directory / "header-content-differences.json", sorted(mismatches))
    ctx.record(f"{name}/header-content", "FAIL" if mismatches else "PASS",
               f"{len(mismatches)} headers differ beyond project include rewriting", directory)
    config_header = installed_root / "tqt_compile_config.h"
    generated = build / "include/torq_tile/tqt_compile_config.h"
    passed = config_header.is_file() and generated.is_file() and config_header.read_bytes() == generated.read_bytes()
    ctx.record(f"{name}/capability-header", "PASS" if passed else "FAIL",
               "installed configuration matches the selected build" if passed else
               "missing or mismatched compile configuration", directory)
    return public


def run(ctx, name, entries):
    build = ctx.output / name / "build"
    directory = ctx.output / name / "install-package"
    code, _ = ctx.command(directory, "build", ["cmake", "--build", str(build),
                                               "--target", "torq_tile", "-j", str(ctx.args.jobs)])
    if code:
        ctx.record(f"{name}/install-package/build", "FAIL", "library build failed", directory)
        return
    prefix = directory / "prefix"
    code, _ = ctx.command(directory, "install-package", ["cmake", "--install", str(build), "--prefix", str(prefix)])
    if code:
        ctx.record(f"{name}/install-package/stage", "FAIL", "install failed", directory)
        return
    relocated = directory / "relocated"
    prefix.rename(relocated)
    public = check_headers(ctx, name + "/install-package", directory, build, relocated, entries)
    candidates = [relocated / "lib" / filename for filename in
                  ("libtorq_tile.a", "libtorq_tile.so")]
    candidates = [path for path in candidates if path.exists() or path.is_symlink()]
    try:
        if len(candidates) != 1:
            raise ValueError(f"Expected one installed static or shared library, found: {candidates}")
        library = relocated_library(candidates[0], relocated)
    except ValueError as error:
        ctx.record(f"{name}/install-package/library-origin", "FAIL", str(error), directory)
        return
    ctx.record(f"{name}/install-package/library-origin", "PASS", str(library), directory)
    include_root = relocated / "include"
    # Every installed public declaration is checked without source-tree include paths.
    low = shlex.split(ctx.args.consumer_flags)
    def check_header(header):
        rel = header.relative_to(ctx.root / "src")
        for lang, suffix in (("c", "c"), ("c++", "cpp")):
            target = directory / "headers" / rel.with_suffix("") / suffix
            target.mkdir(parents=True)
            source = target / f"consumer.{suffix}"
            source.write_text(LICENSE + f"#include <torq_tile/{rel.as_posix()}>\n")
            code, _ = ctx.compile(target, source, low, lang, include_root)
            ctx.record(f"{name}/install-package/{suffix}/{rel}", "PASS" if not code else "FAIL",
                        "baseline consumer compiled" if not code else "inspect compile.log", target)
    ctx.parallel(check_header, sorted(public))
    # Refer to every public function, so static archive extraction cannot hide missing implementations.
    lines = [LICENSE, '#if defined(__riscv_vector) || defined(__riscv_ztt)',
             '#error Baseline consumer inherited backend ISA.', '#endif']
    for header in sorted(public):
        rel = header.relative_to(ctx.root / "src").as_posix()
        lines.append(f"#include <torq_tile/{rel}>")
    symbols = set()
    for header in public:
        content = re.sub(r"/\*.*?\*/|//[^\n]*", "", header.read_text(), flags=re.S)
        symbols.update(re.findall(r"\b(tqt_\w+)\s*\([^;{}]*\)\s*;", content))
    c_lines = list(lines)
    for index, symbol in enumerate(sorted(symbols)):
        lines.append(f"auto *volatile reference_{index} = &{symbol};")
        c_lines.append(f"void (*volatile reference_{index})(void) = (void (*)(void))&{symbol};")
    lines.append('extern "C" int main() { return 0; }')
    c_lines.append("int main(void) { return 0; }")
    consumer = directory / "consumer.cpp"
    consumer.write_text("\n".join(lines) + "\n")
    c_consumer = directory / "consumer.c"
    c_consumer.write_text("\n".join(c_lines) + "\n")
    free = runtime.is_freestanding(build)
    if free:
        for case, source in (("direct-link", consumer), ("c-direct-link", c_consumer)):
            target = directory / case
            executable = runtime.minimal_link(ctx, target, [source], low, library, include_root)
            if not executable or not symbols:
                ctx.record(f"{name}/install-package/{case}", "FAIL", "minimal C-driver link failed", target)
            else:
                runtime.inspect_elf(ctx, f"{name}/install-package/{case}", target, executable)
                if ctx.args.qemu:
                    runtime.execute(ctx, f"{name}/install-package/{case}/load", target, executable)
    else:
        code, _ = ctx.command(directory, "direct-link", [ctx.cxx, *ctx.extra, *ctx.cxx_extra, *low,
                              "-std=c++17", "-I", str(include_root), str(consumer),
                              str(library), "-lm", "-o", str(directory / "consumer")])
        ctx.record(f"{name}/install-package/direct-link", "PASS" if not code and symbols else "FAIL",
                    f"{len(symbols)} public function references; inspect direct-link.log", directory)
        c_directory = directory / "c-consumer"
        code, _ = ctx.compile(c_directory, c_consumer, low, "c", include_root)
        if not code:
            code, _ = ctx.command(c_directory, "link", [ctx.cxx, *ctx.extra, *ctx.cxx_extra, *low,
                                  str(c_directory / "probe.o"), str(library),
                                  "-lm", "-o", str(c_directory / "consumer")])
        ctx.record(f"{name}/install-package/c-direct-link", "PASS" if not code and symbols else "FAIL",
                   f"{len(symbols)} C declarations linked with the C++ runtime", c_directory)
    cmake_source = directory / "cmake-consumer"
    cmake_source.mkdir()
    shutil.copy2(consumer, cmake_source / "consumer.cpp")
    shutil.copy2(c_consumer, cmake_source / "consumer.c")
    cmake_content = (
        'cmake_minimum_required(VERSION 3.16)\nproject(consumer LANGUAGES C CXX)\n'
        'set(CMAKE_CXX_STANDARD 17)\n'
        'find_package(torq_tile CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${CHECK_PREFIX}")\n'
        'file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/imported-library.txt"\n'
        '     CONTENT "$<TARGET_FILE:torq_tile::torq_tile>\\n")\n'
        'add_executable(consumer consumer.cpp)\n'
        'target_link_libraries(consumer PRIVATE torq_tile::torq_tile)\n'
        'add_executable(c_consumer consumer.c)\n'
        'target_link_libraries(c_consumer PRIVATE torq_tile::torq_tile)\n')
    if free:
        for fixture in ("runtime_memory.c", "runtime_start.S"):
            shutil.copy2(HERE / fixture, cmake_source / fixture)
        cmake_content += (
            'enable_language(ASM)\n'
            'get_target_property(core_links torq_tile::torq_tile INTERFACE_LINK_LIBRARIES)\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/core-link-interface.txt" "${core_links}\\n")\n'
            'set_source_files_properties(runtime_memory.c PROPERTIES COMPILE_OPTIONS "-ffreestanding;-fno-builtin")\n'
            'foreach(target consumer c_consumer)\n'
            '  target_sources(${target} PRIVATE runtime_memory.c runtime_start.S)\n'
            '  target_link_options(${target} PRIVATE -nostdlib -static -no-pie -Wl,-e,_start)\n'
            '  target_link_libraries(${target} PRIVATE gcc)\n'
            'endforeach()\n')
    (cmake_source / "CMakeLists.txt").write_text(cmake_content)
    consumer_build = directory / "cmake-build"
    code, _ = ctx.command(directory, "consumer-configure", ["cmake", "-S", str(cmake_source),
                          "-B", str(consumer_build), f"-DCMAKE_CXX_COMPILER={ctx.cxx}",
                          f"-DCMAKE_C_COMPILER={ctx.cc}",
                          "-DCMAKE_SYSTEM_NAME=Linux", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                          f"-DCHECK_PREFIX={relocated}", "-DCMAKE_CXX_FLAGS=" +
                          shlex.join(ctx.extra + ctx.cxx_extra + low),
                          "-DCMAKE_C_FLAGS=" + shlex.join(ctx.extra + ctx.c_extra + low)])
    if not code:
        try:
            imported = Path((consumer_build / "imported-library.txt").read_text().strip())
            if relocated_library(imported, relocated) != library:
                raise ValueError(f"Imported target does not use the installed library: {imported}")
            ctx.record(f"{name}/install-package/imported-library-origin", "PASS", str(imported), directory)
        except (OSError, ValueError) as error:
            ctx.record(f"{name}/install-package/imported-library-origin", "FAIL", str(error), directory)
            code = 1
    if not code:
        code, _ = ctx.command(directory, "consumer-build", ["cmake", "--build", str(consumer_build)])
    if not code:
        # Exported source-tree include paths must not repair a broken installed include graph.
        commands = json.loads((consumer_build / "compile_commands.json").read_text())
        invalid = []
        for entry in commands:
            argv = entry.get("arguments") or shlex.split(entry["command"])
            for index, arg in enumerate(argv):
                path = None
                if arg in ("-I", "-isystem", "-iquote") and index + 1 < len(argv):
                    path = argv[index + 1]
                elif arg.startswith("-I") and len(arg) > 2:
                    path = arg[2:]
                if path:
                    resolved = (Path(entry["directory"]) / path).resolve()
                    if within(resolved, ctx.root) and not within(resolved, relocated):
                        invalid.append(str(resolved))
        write_json(directory / "consumer-include-leaks.json", invalid)
        if invalid:
            code = 1
    if not code and free:
        interface = (consumer_build / "core-link-interface.txt").read_text().strip()
        if interface and interface != "core_links-NOTFOUND":
            ctx.record(f"{name}/install-package/runtime-interface", "FAIL",
                       f"unexpected exported runtime dependencies: {interface}", directory)
            code = 1
        else:
            ctx.record(f"{name}/install-package/runtime-interface", "PASS",
                       "no exported hosted runtime dependencies", directory)
        for target in ("consumer", "c_consumer"):
            if not runtime.inspect_elf(ctx, f"{name}/install-package/{target}/elf",
                                       directory / target, consumer_build / target):
                code = 1
    ctx.record(f"{name}/install-package/cmake-consumer", "PASS" if not code else "FAIL",
                "relocated find_package and link" if not code else "inspect consumer logs", directory)
