#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Check install regeneration in a private source copy; never edit the checkout."""

import shutil

from common import LICENSE, source_path, write_json
from compilation import compile_config, RVV_BASE_FLAGS, RVV_BASE_CAPABILITIES
from installation import check_headers, header_closure


def run(ctx, name, flags, base_entries, control_entries=None):
    directory = ctx.output / name / "install-incremental"
    source = directory / "source"
    build = directory / "build"
    source.mkdir(parents=True)
    shutil.copy2(ctx.root / "CMakeLists.txt", source / "CMakeLists.txt")
    for folder in ("cmake", "src"):
        shutil.copytree(ctx.root / folder, source / folder)
    case = name + "/install-incremental"

    def check(label, passed, detail):
        ctx.record(f"{case}/{label}", "PASS" if passed else "FAIL", detail, directory)
        return passed

    code, _ = ctx.configure(build, flags, source_root=source)
    if not check("high-configure", code == 0, "configure private source copy with original flags"):
        return
    high_entries = ctx.entries(build, source)
    expected_high = {str(source_path(x).relative_to(ctx.root)) for x in base_entries}
    actual_high = {str(source_path(x).relative_to(source)) for x in high_entries}
    if not check("copy-selection", actual_high == expected_high, "copy selects the same sources as the checkout"):
        return
    _, high_headers = header_closure(source, high_entries)
    write_json(directory / "high-headers.json", sorted(str(x.relative_to(source)) for x in high_headers))
    for filename in ("configure.log", "configure.command.json", "CMakeCache.txt"):
        shutil.copy2(build / filename, directory / ("high-" + filename))
    shutil.copy2(build / "include/torq_tile/tqt_compile_config.h", directory / "high-tqt_compile_config.h")

    low = RVV_BASE_FLAGS
    code, _ = ctx.configure(build, low, source_root=source)
    if not check("low-configure", code == 0, "reconfigure the same build with baseline RVV flags"):
        return
    entries = ctx.entries(build, source)
    compile_config(ctx, case, build, entries, root=source, capabilities=RVV_BASE_CAPABILITIES)
    if control_entries is not None:
        expected = {source_path(x).relative_to(ctx.root).as_posix() for x in control_entries}
        actual = {source_path(x).relative_to(source).as_posix() for x in entries}
        write_json(directory / "selection-diff.json", dict(extra=sorted(actual - expected),
                                                           missing=sorted(expected - actual)))
        check("cache-downgrade", actual == expected,
              "shared cmake-config check: downgraded selection matches fresh RVV control")
    public, low_headers = header_closure(source, entries)
    if not low_headers < high_headers:
        ctx.record(f"{case}/downgrade", "SKIP", "requires an RVV superset; use rvv-fp16, rvv-fp16-bf16",
                   directory)
        return

    def build_install(label):
        stage = directory / label
        code, _ = ctx.command(stage, "build", ["cmake", "--build", str(build),
                              "--target", "torq_tile", "-j", str(ctx.args.jobs)])
        if not check(label + "-build", code == 0, "build the actual selected library"):
            return False
        prefix = stage / "prefix"
        code, _ = ctx.command(stage, "install-package", ["cmake", "--install", str(build), "--prefix", str(prefix)])
        if not check(label + "-install", code == 0, "install to a new retained prefix"):
            return False
        check_headers(ctx, case + "/" + label, stage, build, prefix, entries, root=source)
        return True

    if not build_install("downgraded"):
        return
    # A newly introduced dependency must be discovered by an ordinary build.
    header = sorted(public)[0]
    header.write_text(header.read_text() + '\n#include "src/common/tqt_install_regression.h"\n')
    dependency = source / "src/common/tqt_install_regression.h"
    dependency.write_text(LICENSE + "// Private installation-regeneration fixture.\n")
    if not build_install("changed"):
        return
    installed_dependency = directory / "changed/prefix/include/torq_tile/common/tqt_install_regression.h"
    if not check("new-dependency", installed_dependency.is_file(), "ordinary build discovers and installs a new dependency"):
        return
    generated = build / "install/include/torq_tile" / header.relative_to(source / "src")
    timestamp = generated.stat().st_mtime_ns
    code, _ = ctx.configure(build, low, source_root=source)
    check("stable-content", code == 0 and generated.stat().st_mtime_ns == timestamp,
          "unchanged reconfiguration preserves the generated header timestamp")

    # Keep a rejected reference as a negative case, then restore the private copy.
    content = header.read_text()
    header.write_text(content + '\n#include "src/common/tqt_missing_install_header.h"\n')
    code, log = ctx.configure(build, low, source_root=source)
    check("unresolved-include", code not in (0, 124, 127) and "Unresolved public include" in log,
          "a missing public dependency must fail configuration explicitly")
    shutil.copy2(build / "configure.log", directory / "unresolved-include.log")
    header.write_text(content)
    code, _ = ctx.configure(build, low, source_root=source)
    check("restored", code == 0, "private source copy restored; all installation prefixes retained")
