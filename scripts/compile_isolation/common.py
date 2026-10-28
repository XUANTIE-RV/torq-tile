#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Shared toolchain, build workspace, process and reporting services."""

import json
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import signal
import shlex
import shutil
import subprocess
import threading
import time


HERE = Path(__file__).resolve().parent
SUITES = ("compile-probes", "source-selection", "source-compile", "global-init", "cmake-config", "install-package", "install-incremental")
LICENSE = """//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//
"""


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def source_path(entry):
    return (Path(entry["directory"]) / entry["file"]).resolve()


def lto_flags(argv):
    return [arg for arg in argv if arg == "-flto" or arg.startswith("-flto=")]


def within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


class Context:
    def __init__(self, args, output):
        self.args = args
        self.root = args.source_root.resolve()
        self.output = output
        self.results = []
        self.timings = []
        self.started = time.monotonic()
        self.last_progress = self.started
        self.lock = threading.Lock()
        self.processes = set()
        self.cancelled = False
        self.fresh_configurations = {}
        self.cc = shutil.which(args.cc) or args.cc
        self.cxx = shutil.which(args.cxx) or args.cxx
        self.extra = shlex.split(args.flags)
        self.c_extra = shlex.split(args.c_flags)
        self.cxx_extra = shlex.split(args.cxx_flags)

    def set_toolchain(self, item):
        self.cc = shutil.which(item["cc"]) or item["cc"]
        self.cxx = shutil.which(item["cxx"]) or item["cxx"]
        self.extra = item["flags"]
        self.c_extra = item["c_flags"]
        self.cxx_extra = item["cxx_flags"]

    def configuration_key(self, flags):
        # Reuse only validated, fresh configurations in this invocation and source tree.
        return (str(self.root), self.cc, self.cxx, tuple(self.extra), tuple(self.c_extra),
                tuple(self.cxx_extra), tuple(flags), tuple(self.args.cmake_arg))


    def record(self, case, status, detail, directory):
        item = dict(case=case, status=status, detail=detail,
                    evidence=str(directory.relative_to(self.output)))
        with self.lock:
            self.results.append(item)
            now = time.monotonic()
            if self.args.verbose or status != "PASS":
                print(f"[{status}] {case}: {detail} ({item['evidence']})", flush=True)
                self.last_progress = now
            elif now - self.last_progress >= 10:
                print(f"[PROGRESS] {case}; {len(self.results)} checks completed", flush=True)
                self.last_progress = now

    @contextmanager
    def stage(self, profile, suite):
        started = time.monotonic()
        offset = len(self.results)
        print(f"[START] {profile}/{suite}", flush=True)
        try:
            yield
        finally:
            counts = {status: sum(x["status"] == status for x in self.results[offset:])
                      for status in ("PASS", "FAIL", "SKIP")}
            seconds = round(time.monotonic() - started, 3)
            self.timings.append(dict(profile=profile, suite=suite, seconds=seconds, **counts))
            print(f"[DONE] {profile}/{suite}: " + " ".join(f"{k}={v}" for k, v in counts.items()) +
                  f"; {seconds:.1f}s", flush=True)


    def command(self, directory, label, argv, cwd=None):
        if self.cancelled:
            raise InterruptedError("check cancelled")
        directory.mkdir(parents=True, exist_ok=True)
        cwd = cwd or directory
        prefix = directory / label
        write_json(prefix.with_suffix(".command.json"),
                   dict(argv=[str(x) for x in argv], cwd=str(cwd)))
        started = time.monotonic()
        proc = None
        with prefix.with_suffix(".log").open("w") as log:
            log.write(f"cwd: {cwd}\n$ {shlex.join([str(x) for x in argv])}\n\n")
            log.flush()
            try:
                with self.lock:
                    if self.cancelled:
                        raise InterruptedError("check cancelled")
                    proc = subprocess.Popen(argv, cwd=cwd, env=self.env(), stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    self.processes.add(proc)
                try:
                    while True:
                        remaining = self.args.timeout - (time.monotonic() - started)
                        try:
                            code = proc.wait(timeout=max(0, min(10, remaining)))
                            break
                        except subprocess.TimeoutExpired:
                            if time.monotonic() - started >= self.args.timeout:
                                raise
                            with self.lock:
                                if time.monotonic() - self.last_progress >= 10:
                                    print(f"[RUNNING] {directory.relative_to(self.output)}/{label}", flush=True)
                                    self.last_progress = time.monotonic()
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                    raise
                finally:
                    with self.lock:
                        self.processes.discard(proc)
            except FileNotFoundError as error:
                log.write(str(error) + "\n")
                code = 127
            except subprocess.TimeoutExpired:
                log.write("Command timed out.\n")
                code = 124
        write_json(prefix.with_suffix(".result.json"),
                   dict(returncode=code, seconds=round(time.monotonic() - started, 3)))
        return code, prefix.with_suffix(".log").read_text(errors="replace")

    def parallel(self, function, items):
        # Cancel active subprocess groups before joining workers on Ctrl-C.
        with ThreadPoolExecutor(max_workers=self.args.jobs) as pool:
            try:
                return list(pool.map(function, items))
            except KeyboardInterrupt:
                with self.lock:
                    self.cancelled = True
                    for proc in self.processes:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                raise


    @staticmethod
    def env():
        env = os.environ.copy()
        env["LC_ALL"] = "C"
        # Do not let the caller's source-tree include paths mask package defects.
        for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH",
                    "CMAKE_PREFIX_PATH", "CMAKE_TOOLCHAIN_FILE", "CFLAGS", "CXXFLAGS",
                    "LIBRARY_PATH", "LDFLAGS"):
            env.pop(key, None)
        return env


    def compile(self, directory, source, flags, language="c++", include=None, extra=()):
        compiler = self.cxx if language == "c++" else self.cc
        lang_flags = self.cxx_extra if language == "c++" else self.c_extra
        return self.command(directory, "compile", [compiler, *self.extra, *lang_flags, *flags,
                            "-std=c++17" if language == "c++" else "-std=gnu11",
                            "-O2", "-fno-lto", *extra,
                            *(["-I", str(include)] if include else []),
                            "-c", str(source), "-o", str(directory / "probe.o")])


    def preflight(self, name, flags, profile):
        directory = self.output / name / "toolchain"
        directory.mkdir(parents=True)
        self.command(directory, "version-cxx", [self.cxx, "--version"])
        self.command(directory, "version-cc", [self.cc, "--version"])
        macros = {}
        for language, suffix in (("c", "c"), ("c++", "cpp")):
            source = directory / f"sanity.{suffix}"
            source.write_text(LICENSE + "#include <stddef.h>\nint probe(void) { return 0; }\n")
            subdir = directory / suffix
            subdir.mkdir()
            code, _ = self.compile(subdir, source, flags, language)
            if code:
                self.record(f"{name}/toolchain/{suffix}", "SKIP",
                            "compiler cannot compile this profile; inspect compile.log", subdir)
                return None
            compiler = self.cc if language == "c" else self.cxx
            extra = self.c_extra if language == "c" else self.cxx_extra
            code, log = self.command(subdir, "macros", [compiler, *self.extra, *extra, *flags,
                                                       "-std=c++17" if language == "c++" else "-std=gnu11",
                                                       "-dM", "-E", str(source)])
            if code:
                self.record(f"{name}/toolchain/{suffix}", "SKIP", "macro dump failed", subdir)
                return None
            values = dict(re.findall(r"^#define (\w+)\s*(.*)$", log, re.M))
            missing = set(profile["required"]) - values.keys()
            unexpected = set(profile["forbidden"]) & values.keys()
            if missing or unexpected:
                self.record(f"{name}/toolchain/{suffix}", "FAIL",
                            f"profile mismatch: missing={sorted(missing)}, forbidden={sorted(unexpected)}",
                            subdir)
                return None
            macros[language] = values
        self.record(f"{name}/toolchain", "PASS", "C/C++ compile and ISA macros verified", directory)
        return macros["c++"]


    def configure(self, directory, flags, options=(), c_flags=None, source_root=None):
        directory.mkdir(parents=True, exist_ok=True)
        return self.command(directory, "configure", ["cmake", "-S", str(source_root or self.root),
                            "-B", str(directory), "-G", "Unix Makefiles",
                            "-DCMAKE_SYSTEM_NAME=Linux", "-DCMAKE_SYSTEM_PROCESSOR=riscv64",
                            f"-DCMAKE_C_COMPILER={self.cc}", f"-DCMAKE_CXX_COMPILER={self.cxx}",
                            "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                            "-DCMAKE_C_FLAGS=" + shlex.join(self.extra + self.c_extra +
                                                          (flags if c_flags is None else c_flags)),
                            "-DCMAKE_CXX_FLAGS=" + shlex.join(self.extra + self.cxx_extra + flags),
                            "-DCMAKE_C_FLAGS_RELEASE=-O2 -DNDEBUG",
                            "-DCMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG",
                            "-DTORQ_TILE_BUILD_TEST=OFF", "-DTORQ_TILE_BUILD_BENCHMARK=OFF",
                            *self.args.cmake_arg, *options])


    def entries(self, directory, root=None):
        root = root or self.root
        entries = json.loads((directory / "compile_commands.json").read_text())
        # Test and benchmark targets are disabled. Keep the kernel library's sources only.
        entries = [x for x in entries if within(source_path(x), root / "src")]
        write_json(directory / "selected-sources.json", [str(source_path(x).relative_to(root))
                                                       for x in entries])
        return entries


    def report(self):
        self.results.sort(key=lambda x: x["case"])
        counts = {x: sum(y["status"] == x for y in self.results) for x in ("PASS", "FAIL", "SKIP")}
        code = 1 if counts["FAIL"] else 2 if counts["SKIP"] and not self.args.allow_skip else 0
        seconds = round(time.monotonic() - self.started, 3)
        write_json(self.output / "results.json", dict(counts=counts, exit_code=code, cases=self.results,
                   wall_seconds=seconds, timings=self.timings,
                   filtered=self.args.probe_filter != "." or self.args.source_filter != "."))
        rows = ["# Compile isolation results", "", f"PASS={counts['PASS']} FAIL={counts['FAIL']} SKIP={counts['SKIP']}",
                "", f"Wall time: {seconds:.1f}s. Jobs: {self.args.jobs}.",
                "", "Coverage: FILTERED checks." if self.args.probe_filter != "." or self.args.source_filter != "."
                else "Coverage: requested plan (see invocation.json).",
                "", "| Profile | Suite | Seconds | PASS | FAIL | SKIP |",
                "| --- | --- | ---: | ---: | ---: | ---: |"]
        for timing in self.timings:
            rows.append("| " + " | ".join(str(timing[x]) for x in
                        ("profile", "suite", "seconds", "PASS", "FAIL", "SKIP")) + " |")
        rows += ["", "| Case | Result | Detail | Evidence |", "| --- | --- | --- | --- |"]
        for result in self.results:
            rows.append("| " + " | ".join(str(result[x]).replace("|", "\\|").replace("\n", " ")
                                           for x in ("case", "status", "detail", "evidence")) + " |")
        (self.output / "README.md").write_text("\n".join(rows) + "\n")
        print(f"\nPASS={counts['PASS']} FAIL={counts['FAIL']} SKIP={counts['SKIP']} exit={code}"
              f"\nWall time: {seconds:.1f}s; jobs={self.args.jobs}\nEvidence: {self.output}")
        return code
