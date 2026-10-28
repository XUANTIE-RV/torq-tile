//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

// Linux user-mode test entry: no libc startup or initializer traversal.
.text
.global _start
.type _start, @function
_start:
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop
    call main
    li a7, 93
    ecall
1:
    j 1b
.size _start, .-_start
.section .note.GNU-stack,"",@progbits
