#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#
# This script formats all C++ source files in the project.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Find all C++ source and header files, excluding build and CMakeFiles directories.
FILES=$(find \
    "$PROJECT_ROOT/src" "$PROJECT_ROOT/examples" "$PROJECT_ROOT/test" "$PROJECT_ROOT/benchmark" \
    -type d \( -name "build" -o -name "CMakeFiles" \) -prune \
    -o -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.c" -o -name "*.cpp" \) -print \
    2>/dev/null)

if [ -z "$FILES" ]; then
    echo "No C/C++ files found to format."
    exit 0
fi

# Format the files in place.
clang-format-18 -i $FILES

echo "Formatting complete."
