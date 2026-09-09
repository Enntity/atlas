# Three/four-owner temporal layer integration

2026-09-09, source `bc3637fd`. Host-side compute integration only; no new
serving throughput or native numerical qualification is claimed.

The actual KDA and MLA layer entries now share a bounded traversal across
three/four temporal K5 owners. Each owner's original attention/state path is
preserved; saved normalized/mHC rows feed one M15/M20 routed and shared FFN,
then each owner receives its own post-FFN mHC update. Dense FFNs retain K5
arithmetic and assemble owner outputs in descending order. Existing pair
controls retain their literal TwoK5/Joint/JointSharedM10 semantics.

The shared arena validator requires 30/40 normalized rows and checks original
backend/arena identity, full-span nonaliasing and all selected owners before
state writes. KDA checks every owner's state/snapshot spans; MLA checks every
cross-owner writable KV slot, including nonadjacent owners. New traversal
storage uses bounded stack arrays. This is not a blanket allocation-free claim
for all existing kernel dispatch internals or the unfinished model transaction.

## Evidence and limits

Logs are under
`/home/abc/storage/models/atlas-campaigns/20260909/glm-native-controller/`.
Actual runtime refusal REDs preceded each wider workspace/FFN/KDA/MLA entry.
The precommit GREEN logs are `owner-{workspace,ffn,kda,mla}-green.log`.
Byte-backed workspace checks cover all four saved tails and last-owner
prewrite rejection. Dispatch recorders cover actual M15/M20 router, compact
gate/up, dense down, shared FFN and EP reduction launch geometry, not CUDA
arithmetic. The FFN oracle permits exactly the existing packed-A/scale staging
copy before down projection; treating that required copy as a regression was
an incorrect test assumption, corrected without changing production code.

Existing pair FFN/KDA controls and 144 model handoff/ownership tests pass before
commit. Non-test spark-model CUDA-feature host check and workspace formatting
pass. After source commit, `bc3637fd-owner-tests.log` records 66 passing tests
selected by `cpu-model.sh test owner`, including the wider entries and existing
owner controls. No full-workspace Clippy-clean or native model claim is made.

The prior standalone native M15/M20 arithmetic evidence remains separate in
`owner-batch-ffn-native-results.md`; it does not qualify this composed layer
implementation. The newest actual C4 serving result remains source `6e1e37f4`,
approximately36.9 aggregate full-wall tok/s, from one fresh process.

## Next integration

Connect a bounded producer retaining all selected owners until all commits,
then actual model traversal, a distinct fixed-width E7 operation, and scheduler
selection. Preserve E6 unchanged. In particular, avoid copying the pair model's
post-issue token/metadata Vec allocations into the wider path. New transport
must validate the whole message and physical slot identity before any writer.

Only then perform native numerical/quality checks, warmed C1/C2/C3/C4 timing,
fresh-process repetition, memory/no-swap observation and normal paired release.
Coherence, real tool calls and needle retrieval remain mandatory. C6/C8 MTP,
the full performance targets and staged large-context qualification remain open.
