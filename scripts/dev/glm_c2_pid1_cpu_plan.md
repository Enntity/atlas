# Local private PID1 death gate

2026-09-09. Root-owned CPU-only qualification after independent T3.1 review.
No Docker, Spark, GPU, node access or deployment authority. This supplements
the real guard lifecycle suite; namespace creation alone is not a death test.

Build a small standalone C++ Linux harness with the system compiler, outside
the Cargo/engine build. Source belongs under the standalone guard's tests.
Run only this exact local artifact with `sudo -n`, under a finite outer timeout.
No mount, network, credential, cgroup, host configuration or service changes.

The parent creates its exact private temporary witness directory and an unrelated
CPU decoy before `unshare(CLONE_NEWPID)`. The parent stays in its existing PID
namespace. Its next fork execs the actual frozen guard as the new namespace's
PID1, inheriting only the intended control socket. Verify the init's host PID
from the actual fork return, retain its pidfd, and check its `/proc` NSpid ends
in 1 and its PID namespace inode differs from the parent/decoy.

Use the actual 112-byte HELLO/START protocol, bounded socket operations and
explicit durations. Observe the guard's direct gated child using only that
owned init's kernel child relationship, acquire its pidfd before START, and
verify the same namespace. The executed CPU child then forks one harmless
descendant. It verifies PR_GET_PDEATHSIG is zero, as fork clears that setting;
observe it through the owned child's kernel relationship and retain its pidfd.
No arbitrary supplied PID or broad process-name search is authority.

Check all three processes are alive and the child remains gated before START.
The probe has real C++ destructor and atexit witnesses on ordinary termination;
confirm those witnesses with a separate healthy local probe control before the
namespace case. After actual exec and descendant readiness, send SIGKILL only
through the retained init pidfd. Require bounded observed death of init, child
and descendant while the outside-namespace decoy stays alive and no namespace
child/descendant graceful-cleanup witnesses exist. The descendant with no
PDEATHSIG separates namespace termination from parent-death-signal evidence.

Harness failure cleanup signals only retained CPU pidfds, with bounded waits.
Private init death is the containment fallback for descendants; never use a
process-name kill or recursive deletion of an unresolved directory. Keep exact
witness directories and raw command/output/source/binary hashes as receipts.
The harness uses no engine API and makes no CUDA recovery guarantee. Root must
read the complete harness and obtain independent review before privileged use.
This is a qualification gate, not a fabricated failing production test.
