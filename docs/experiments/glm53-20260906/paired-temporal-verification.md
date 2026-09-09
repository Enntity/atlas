# Fixed two-owner temporal verification candidate

2026-09-09. Native source `5b4662a9` now passes the first full serving A/B:
C2=27.314→34.333 aggregate full-wall tok/s, C1 unchanged at approximately27.3.
See `paired-ffn-serving-results.md` for quality, safety, provenance and limits.

## Change and explicit controls

`ATLAS_GLM_C2_PAIRED_VERIFY=1` requires an explicit
`ATLAS_GLM_C2_PAIR_FFN=two-k5|joint`, identically configured on both ranks.
The absent/zero enable flag retains the existing serial path. Admission stays
at two physical owners, four drafts, eager TP2/EP2, BF16 KV, FP32 SSM and the
existing short-context bounds. Cold, single-owner and draining work stays serial;
there is no fallback after a pair transaction starts.

The actual model traverses layers with two independent five-token temporal
owners. Their KDA states, rollback snapshots and MLA cache maps remain separate.
The `two-k5` control uses each existing K5 FFN; `joint` combines the ten normalized
rows for one routed MoE pass and reduction, retaining two generic-T K5 shared calls.
Both modes retain two M5 final-normalization and vocabulary-head calls.
The first candidate changed the shared projection implementation and failed
native exact comparison. The corrected implementation preserves the control's
shared projection arithmetic; no comparison tolerance was relaxed.

No additional resident allocation is introduced: checked tails of the original
prefill arena save normalized rows10..19 and mHC rows5..14. This requires at
least20 allocated prefill rows. Two K5 attention-metadata planes use scratch
bytes32768..39424 at the128-block maximum, outside admitted route scratch.
Original buffer and request bindings are never replaced.

An exclusive paired producer owns both results until both checked selections,
accepted/bonus hidden-row detachments and target commits complete. Only then
can either request emit or start its next proposal. The fixed26-word E6 packet
includes mode, both generations/attempts, positions and actual issued tokens;
the worker reconstructs and compares the entire packet before execution.
Every post-header failure remains terminal under the existing paired supervisor.

## Evidence boundary and next gate

Retained external evidence directory:
`/home/abc/storage/models/atlas-campaigns/20260909/glm-native-controller/`.
Focused runtime REDs reached unsupported actual FFN, KDA, producer and Model
entries before implementation. Subsequent CPU checks execute the actual Model
and scheduler, not fabricated Model capabilities. They cover all25 acceptance
pairs in both owner orders, real worker packet replay, next E1 and a second
paired verification against the existing serial byte oracle. Scheduler checks
also execute both checked five-row selections and preserve detached peer state.
These byte fixtures do not establish KDA/MLA/CUDA numerical correctness.

The standalone `scripts/dev/bench_glm_pair_ffn.cu` compares the actual current
two-M5 routed kernels against the ten-row candidate from router through mHC.
Its explicit192MiB cap includes every guarded device allocation; it requires
explicit comparison thresholds and starts at zero absolute/relative tolerance.
Two ranks are simulated arithmetically on one GPU, so its timing is not NCCL
or serving throughput. Native comparison, full-model quality/rollback checks,
warmed C1/C2 evidence and a fully qualified paired exit remain required.

The Docker Running-to-clean-Exited observation race is fixed separately in
`7d0d220b`; its corrected observer now completes both native serving campaigns
with actual paired release and independently observed exit0. Neither that fix
nor this C2 improvement meets the full reference-parity goal by itself.

The first native `two-k5` startup reached actual model construction, then exited
before readiness. Both containers were stopped by the supervisor (137,
OOMKilled=false); minimum observed host available memory was9,970,536KiB and
10,233,252KiB, with zero swap. A CPU reproduction found that the new FFN guard
required `Skip` while the real base-model adapter resolver returns inert `Fold`
when no adapter is loaded. The correction admits that base state only with no
resident adapter and zero configured adapter rank; `Refuse` remains rejected.
Factory diagnostics now print the indexed layer error before fail-safe exit.
The corrected candidate subsequently passes both native modes and reversed-order
fresh-process repetition as detailed in `paired-ffn-serving-results.md`.
