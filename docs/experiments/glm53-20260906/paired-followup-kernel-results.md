# Paired FFN follow-up kernel candidates

2026-09-09. These are bounded, standalone one-GB10 measurements, not serving
rates. Both models were stopped; GPU/build activity was serialized. Retained
evidence root: `/home/abc/storage/models/atlas-campaigns/20260909/glm-native-controller/`.

## Shared expert: retain generic-T arithmetic, widen M5 to M10

Source `95b6b4f0`, `bench_glm_pair_ffn --compare joint-shared`. Both arms use
the qualified Joint router/routed GU/dense down/mHC. Only two generic-T M5
shared chains versus one generic-T M10 chain differ. The existing fixed
M64/N128/K32 kernel and elementwise FP8 activation conversion are unchanged.
This is not the previously rejected exact-K5 GEMV substitution.

All shared gate/up, post-SiLU, down and final mHC elements match bitwise, as
do routed intermediates. Six restored-input cases cover both owner orders,
local/remote boundary masks and both scale-load variants. Per-owner row maps,
fresh output poison, sampled full-K independent FP8 host dots, the existing
independent post/router oracles and all allocation canaries pass. Compute
Sanitizer reports `ERROR SUMMARY: 0 errors`; normal exit, no OOM/observed swap.

Full FFN arithmetic timing, including two ranks serialized on one GPU:

| Scale loads | Order | Two M5 shared, ms | One M10 shared, ms | Ratio |
| --- | --- | ---: | ---: | ---: |
| Scalar | Control first | 3.041731 | 2.207003 | 1.378218 |
| Scalar | Candidate first | 2.905459 | 2.144018 | 1.355147 |
| Vector | Control first | 2.719223 | 1.958922 | 1.388122 |
| Vector | Candidate first | 2.716841 | 1.958491 | 1.387212 |

Three warmups and20 measured repetitions per order; no timed D2H. Explicit
device allocations remain147,131,880 bytes under192MiB. Plain run has120s/2GiB
and memcheck180s/4GiB no-swap container bounds. CUDA13, `-O3 --fmad=false`,
`sm_121a`. ELF SHA256:
`271b92bdbf09d61a477dad5366a2685b40c81f13a9996218927345515b5dbaad`.
Evidence: `joint-shared-native-{build,timing,memcheck}.log`,
`joint-shared-source.tgz`. This warrants explicit opt-in serving integration;
no full-model or tok/s improvement is established by this experiment.
Subsequent explicit serving integration `290cf248` is separately qualified in
`shared-m10-serving-results.md`: repeated C2 full-wall36.76–36.96 versus
same-binary Joint34.243. Do not substitute the standalone1.36–1.39x FFN ratio
for that measured serving improvement.

## First-draft BF16 vocabulary projection

Source `74e3ef3c`, `bench_glm_pair_draft_head.cu`: two scalar projections versus
existing batch2 at N77428/K4096, output stride154856. The full634,290,176-byte
weight shard is resident; total guarded device allocation635,546,432 bytes,
under768MiB. Three distinct input profiles, both rank output offsets and owner
reversal pass complete BF16-logit equality, sampled exact-order host dots,
unused output-half checks and canaries. Memcheck reports zero errors.

Two execution orders give5.446805→2.740501ms and5.387651→2.735990ms
(1.987522x/1.969178x). This saves approximately2.7ms per two-row projection;
it does not measure paired proposal or model throughput. Integrating it would
require paired first-draft plumbing; shared-expert batching is prioritized.
ELF `070277b0e10f153aa73a7366d5b22a7e6b1133a1d48b0a0e5c2a4accfd3fe623`.
Evidence: `draft-head-native-{build,timing,memcheck}.log` and
`draft-head-74e3ef3c-source.tgz`. Plain run120s/3GiB; memcheck180s/4GiB;
normal exits with zero observed swap/OOM.

## Reused GU worklist for down: parked

Source `a31531ad`, FFN `--compare joint-down`. The new standalone CTA mapper
reuses the actual GU list, avoiding a second builder; the native M64 down
arithmetic remains unchanged. Maximum launched CTAs shrink9216→2560.
The eight-weight synthetic fixture, row mapping/count/capacity guards and full
routed/shared/final output equality pass. This does not cover eighty distinct
resident expert weights or production scratch integration.

Whole FFN ratios vary1.001400–1.030424 for scalar loads and
1.008368–1.011948 for vector loads across execution orders. This small/noisy
gain does not justify serving integration now. No memcheck or serving
qualification is claimed for this candidate; the historical compact-down
integration stall remains unresolved. Existing production dense down stays on.
ELF `7405c60eb5fd0e4534c5258a376603a70f0084d9c9b68d7d90974fc69a906533`.
Evidence: `reused-down-native-{build,timing}.log` and
`reused-down-a31531ad-source.tgz`; normal exit, no OOM/observed swap.
