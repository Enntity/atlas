// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

struct Exl3Expert {
    unsigned long long trellis;
    unsigned long long suh;
    unsigned long long svh;
    unsigned int bits;
    unsigned int reserved;
};
static_assert(sizeof(Exl3Expert) == 32, "EXL3 expert descriptor ABI");
