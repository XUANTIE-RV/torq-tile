# Compile isolation checks

Run `python3 scripts/compile_isolation/check.py --quick` for scalar/RVV header
and packing probes, or `--full` for the six-profile build matrix, initializer,
configuration, installation and incremental-install checks. These checks use
only the Python standard library and do not execute target kernels.

Select the compiler with `--cc` and `--cxx`. Keep evidence outside the source
tree with `--output-dir /path/to/new-results`. Use `--list` or `--dry-run` to
inspect the plan without invoking the compiler. Run numerical tests separately
through CTest with a runner matching the selected target extensions.
