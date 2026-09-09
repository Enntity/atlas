# Selected paired allocation: retirement-order-independent reuse

2026-09-09. Code `d5a1bb0e0f8e6fdff08dde63d05743c2acd05c45`.
CPU correctness infrastructure only; factory/admission remain disabled and v26
remains the last qualified native deployment. No new throughput measurement.

The actual private allocator chooses its lowest free slot; the target allocator
uses a LIFO list. Retiring requests 0 then 1, including on separate scheduler
ticks, made a later selected allocation refuse the differing identities.
The selected path now claims a genuine target guard for the actual private
candidate, using the existing locked specific-claim operation. Generic guarded
allocation stays LIFO. If that exact target is unavailable, the session becomes
terminal before GPU work without consuming an unrelated free target.

Actual Model churn tests cover both ranks, both retirement orders, repeated
allocation/priming/K5/repair, and alternating replacement while a peer stays live.
They check real guard identity, disjoint private reserves, exact returned reserve
union and unchanged peer tokens/KV/hidden rows. The original unrestricted
`[victim, peer]` lifecycle retirement order is restored. Pool tests additionally
check unavailable/out-of-range refusal, exactly-once Drop and explicit take.

The first test attempt found a wrong test expectation: Model retains a padding
block. That control failure is not the allocation RED. The corrected actual
churn run was 1 PASS / 1 FAIL at the identity mismatch; held-target refusal was
0 PASS / 1 FAIL at consumption of the unrelated slot. Final focused handoff
suite passed 120 tests, and the complete SSM pool suite passed 14 tests.

Independent source review and root review approved the eight-path frozen
manifest SHA-256 `6e62fa54ab57310dae6777a36787b80c603cf92ab759073a8674f370459dd4f5`.
At the exact code commit, root's model suite passed **1100 / 1100** in 85.89s
with four test threads; server passed **2363 / 2363**, 12 ignored, in 23.54s
with one test thread. These CPU elapsed times are not inference performance.
Non-test model check, formatting, diff whitespace and scoped SPDX checks passed.
New Rust files are <=500 lines; existing allowlisted `ssm_pool.rs` is 1246 lines.
Clippy still stops on four inherited runtime Metal-stub argument-count errors;
no CI-green or native arithmetic qualification is claimed.

Raw receipts: `/home/abc/storage/models/atlas-campaigns/20260908/glm-c2-aligned-allocation/`.
Exact commands and exclusions are in `postcommit-closure.md`. The detached-hash
archive is `glm-c2-aligned-allocation-d5a1bb0e-receipts.tar`.
Both nodes' Docker process lists were empty; head/worker MemAvailable was
121606912 / 121685400 kB, with SwapTotal=SwapFree=10485756 kB on both.
No image build, native execution, reset or service mutation occurred.

The next slice shares the existing actual model fixture with server tests via
an explicit non-default test feature. Selected scheduler dispatch, supervised
registration, complete strict teardown and native two-rank control remain
separate requirements; this fix does not enable concurrent serving by itself.
