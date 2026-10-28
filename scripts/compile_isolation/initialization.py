#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Inspect already compiled objects for load-time initializer sections."""

from pathlib import Path
import re


def run(ctx, name, objects):
    def check(item):
        relative = item["source"]
        directory = Path(item["directory"])
        if item["returncode"]:
            ctx.record(f"{name}/global-init/{relative}", "SKIP", "object compilation failed", directory)
            return
        code, log = ctx.command(directory, "sections", [ctx.args.readelf, "-SW", item["object"]])
        found = re.findall(r"\]\s+(\.(?:init_array|preinit_array|ctors)(?:\.\S+)?)\s", log)
        lto = ".gnu.lto_" in log or ".llvmbc" in log or ".llvm.lto" in log
        ctx.record(f"{name}/global-init/{relative}",
                   "SKIP" if code == 127 else "FAIL" if code or found or lto else "PASS",
                   "LTO object is unsupported; initializer inspection is incomplete" if lto else
                   f"load-time initializer sections: {found}" if found else
                   "section inspection failed" if code else "no initializer sections", directory)
    ctx.parallel(check, objects)
