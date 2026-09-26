// SPDX-License-Identifier: AGPL-3.0-only
// 128-dim-head build of the DFlash paged indirect (sink) attention kernel.
//
// prefill_paged_compute.cuh sizes its Q/K/V tiles by the compile-time HDIM
// (default 256) while addressing rows by the runtime head_dim. A drafter with
// head_dim 128 on a target built at HDIM 256 (GLM-5's MLA set) therefore mixes
// adjacent heads in every score. This module pins HDIM to 128 so such drafters
// select a matching kernel; targets already built at HDIM=128 are unaffected.
#undef HDIM
#define HDIM 128
#include "inferspark_prefill_paged_indirect_sink.cu"
