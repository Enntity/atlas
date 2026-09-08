// SPDX-License-Identifier: AGPL-3.0-only

//! Explicit I/O failure policy; the existing immunity proof remains the SSOT.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CopyFailurePolicy {
    LegacyFallback,
    Propagate,
}

impl CopyFailurePolicy {
    pub(super) fn immune(
        self,
        token: u32,
        history: &[u32],
        probe: impl FnOnce() -> anyhow::Result<bool>,
    ) -> anyhow::Result<bool> {
        let mut failure = None;
        let immune =
            crate::scheduler::fast_greedy::argmax_immune(token, history, || match probe() {
                Ok(positive) => positive,
                Err(error) => {
                    if self == Self::Propagate {
                        failure = Some(error);
                    }
                    false
                }
            });
        match failure {
            Some(error) => Err(error),
            None => Ok(immune),
        }
    }
}
