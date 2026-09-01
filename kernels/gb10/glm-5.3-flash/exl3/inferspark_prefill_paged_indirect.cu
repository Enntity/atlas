// SPDX-License-Identifier: AGPL-3.0-only
//
// GLM-5.3 Flash DFlash2 drafter: head_dim=128. The target model uses a
// different head width, but this indirect non-causal kernel is used only by
// the drafter. Compiling the common default (HDIM=256) makes every MMA tile
// read across adjacent Q/K/V heads even though the runtime argument is 128.
#define HDIM 128
#include "../../common/inferspark_prefill_paged_indirect.cu"
