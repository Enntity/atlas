// SPDX-License-Identifier: AGPL-3.0-only

//! Host-only GLM shifted-pair plans. No GPU allocation, execution, wire
//! identity, or live-state validation occurs here. Callers must bind counts,
//! generations and normalized hidden ownership to real request state, then
//! publish the returned state only after all planned cache writes succeed.
//! Fixed C1/four-draft continuous speculation only; no runtime callers yet.

use anyhow::{Context, Result, ensure};

/// Explicit profile assertion supplied by a future checked runtime boundary.
/// This is not a replacement for GLM/TP/EP/BF16/server capability validation.
#[derive(Clone, Copy, Debug)]
pub struct Profile {
    pub sequences: usize,
    pub drafts: usize,
    pub continuous: bool,
    pub grammar: bool,
    pub adaptive_depth: bool,
    pub catchup: bool,
    pub carry: bool,
    pub prefix_reuse: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    context_tokens: usize,
    cache_rows: usize,
    staging_rows: usize,
}

impl Limits {
    /// `cache_rows` is physically allocated private-KV row capacity, not an
    /// overcommit budget. `staging_rows` excludes the separately saved bonus.
    pub fn new(
        profile: Profile,
        context_tokens: usize,
        cache_rows: usize,
        staging_rows: usize,
    ) -> Result<Self> {
        ensure!(
            profile.sequences == 1
                && profile.drafts == 4
                && profile.continuous
                && !profile.grammar
                && !profile.adaptive_depth
                && !profile.catchup
                && !profile.carry
                && !profile.prefix_reuse,
            "GLM pair repair requires fixed continuous C1 MTP4 without grammar/adaptation/carry/catchup/prefix reuse"
        );
        let dense_bound = context_tokens
            .checked_add(4)
            .context("GLM context overflow")?;
        ensure!(
            context_tokens > 0 && dense_bound <= 2048,
            "GLM context + drafts must be <= 2048"
        );
        ensure!(cache_rows > 0, "GLM cache capacity must be nonzero");
        Ok(Self {
            context_tokens,
            cache_rows,
            staging_rows,
        })
    }

    /// Consume either all shifted prompt pairs (lazy) or just the missing
    /// terminal prompt pair after an eager P-1-row primer. Tokens include the
    /// first generated token x[P]; hidden rows are prompt H[0..P), post-norm.
    pub fn bootstrap(self, input: BootstrapInput) -> Result<FinishPlan> {
        ensure!(
            input.generation != 0 && input.generation == input.capture_generation,
            "GLM bootstrap generation does not own prompt capture"
        );
        ensure!(
            input.prompt_tokens > 0,
            "GLM bootstrap prompt must be nonempty"
        );
        let position = input
            .prompt_tokens
            .checked_add(1)
            .context("GLM bootstrap position overflow")?;
        ensure!(
            input.target_position == position && position <= self.context_tokens,
            "GLM bootstrap target position must immediately follow the first generated token"
        );
        ensure!(
            input.token_rows >= position,
            "GLM bootstrap token span misses first generated token"
        );
        ensure!(
            input.normalized_hidden_rows >= input.prompt_tokens,
            "GLM bootstrap hidden span misses prompt tail"
        );
        ensure!(
            input.cached_rows == 0 || input.cached_rows == input.prompt_tokens - 1,
            "GLM bootstrap cache must be empty or exactly the eager prompt prefix"
        );
        ensure!(
            input.prompt_tokens <= self.cache_rows,
            "GLM bootstrap exceeds cache capacity"
        );
        let rows = input.prompt_tokens - input.cached_rows;
        Ok(FinishPlan {
            state: PairState {
                generation: input.generation,
                cache_rows: input.prompt_tokens,
                target_position: position,
            },
            write: Some(PairWrite {
                cache_start: input.cached_rows,
                token_start: input.cached_rows + 1,
                hidden_start: input.cached_rows,
                rows,
            }),
            bonus_hidden_row: None,
            keep_seed: false,
        })
    }

    /// Prepare a proposal with a canonical committed prefix. The resulting
    /// transient cache extent is explicit so a later stale/partial write
    /// cannot be mistaken for the full proposal. This does not execute it.
    pub fn propose(
        self,
        state: PairState,
        generation: u64,
        position: usize,
        observed_cache_rows: usize,
        drafts: usize,
    ) -> Result<ProposalPlan> {
        ensure!(
            generation == state.generation,
            "GLM proposal generation mismatch"
        );
        ensure!(
            drafts == 4,
            "GLM proposal must retain fixed continuous four-draft depth"
        );
        ensure!(
            position == state.target_position && position < self.context_tokens,
            "GLM proposal position changed or reached context cap; uncovered serial gaps cannot resume"
        );
        ensure!(
            observed_cache_rows == state.cache_rows,
            "GLM proposal cache ownership mismatch"
        );
        let speculative_end = state
            .cache_rows
            .checked_add(drafts)
            .context("GLM proposal cache overflow")?;
        ensure!(
            speculative_end <= self.cache_rows,
            "GLM proposal exceeds cache capacity"
        );
        // Validate every possible verdict before the first transient write:
        // full acceptance needs one more pair than the proposal generated.
        let full_accept_end = speculative_end
            .checked_add(1)
            .context("GLM full-accept cache overflow")?;
        ensure!(
            full_accept_end <= self.cache_rows,
            "GLM possible full acceptance exceeds cache capacity"
        );
        ensure!(
            self.staging_rows >= drafts,
            "GLM possible full acceptance exceeds hidden staging capacity"
        );
        // A K5 forward processes the pending token and four draft inputs.
        let verify_end = position
            .checked_add(drafts + 1)
            .context("GLM verify position overflow")?;
        ensure!(
            verify_end <= 2048,
            "GLM verification exceeds dense 2048 bound"
        );
        Ok(ProposalPlan {
            limits: self,
            before: state,
            speculative_end,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BootstrapInput {
    pub generation: u64,
    pub capture_generation: u64,
    pub prompt_tokens: usize,
    pub target_position: usize,
    pub token_rows: usize,
    pub normalized_hidden_rows: usize,
    pub cached_rows: usize,
}

/// Canonical private cache covers every shifted pair before the next seed.
/// Fields are private: only a checked bootstrap/commit constructs this state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairState {
    generation: u64,
    cache_rows: usize,
    target_position: usize,
}

impl PairState {
    pub fn generation(self) -> u64 {
        self.generation
    }
    pub fn cache_rows(self) -> usize {
        self.cache_rows
    }
    pub fn target_position(self) -> usize {
        self.target_position
    }
}

/// Sources are indexed within the input's token and normalized-hidden spans.
/// Bootstrap uses full prompt spans; verified commit uses K5 verify spans.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairWrite {
    cache_start: usize,
    token_start: usize,
    hidden_start: usize,
    rows: usize,
}

impl PairWrite {
    pub fn cache_start(self) -> usize {
        self.cache_start
    }
    pub fn token_start(self) -> usize {
        self.token_start
    }
    pub fn hidden_start(self) -> usize {
        self.hidden_start
    }
    pub fn rows(self) -> usize {
        self.rows
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VerifiedCommit {
    pub generation: u64,
    pub capture_generation: u64,
    pub accepted: usize,
    pub verify_token_rows: usize,
    pub normalized_hidden_rows: usize,
    pub target_position: usize,
    pub observed_cache_rows: usize,
    /// Absolute target position represented by normalized hidden row zero.
    /// Distinguishes old captures from another cycle of the same generation.
    pub hidden_base_position: usize,
}

/// A discarded, unverified proposal has no committed seed. In particular it
/// is NOT synonymous with verifying the seed and accepting zero drafts.
#[derive(Clone, Copy, Debug)]
pub enum Finish {
    Verified(VerifiedCommit),
    DiscardUnverified {
        generation: u64,
        observed_cache_rows: usize,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct ProposalPlan {
    limits: Limits,
    before: PairState,
    speculative_end: usize,
}

impl ProposalPlan {
    pub fn position(self) -> usize {
        self.before.target_position
    }
    pub fn speculative_cache_end(self) -> usize {
        self.speculative_end
    }

    pub fn finish(self, event: Finish) -> Result<FinishPlan> {
        match event {
            Finish::DiscardUnverified {
                generation,
                observed_cache_rows,
            } => {
                self.validate_owner(generation, observed_cache_rows)?;
                Ok(FinishPlan {
                    state: self.before,
                    write: None,
                    bonus_hidden_row: None,
                    keep_seed: false,
                })
            }
            Finish::Verified(input) => self.commit(input),
        }
    }

    fn validate_owner(self, generation: u64, cache_rows: usize) -> Result<()> {
        ensure!(
            generation == self.before.generation,
            "GLM finish generation mismatch"
        );
        ensure!(
            cache_rows == self.speculative_end,
            "GLM finish cache extent does not match the proposal"
        );
        Ok(())
    }

    fn commit(self, input: VerifiedCommit) -> Result<FinishPlan> {
        self.validate_owner(input.generation, input.observed_cache_rows)?;
        ensure!(
            input.capture_generation == self.before.generation,
            "GLM verification hidden generation mismatch"
        );
        ensure!(
            input.accepted <= 4,
            "GLM accepted count exceeds four drafts"
        );
        ensure!(
            input.verify_token_rows == 5,
            "GLM verified token span must contain exactly K5 inputs"
        );
        ensure!(
            input.normalized_hidden_rows >= 5,
            "GLM normalized hidden span does not cover K5"
        );
        ensure!(
            input.hidden_base_position == self.before.target_position,
            "GLM verification hidden base position is stale"
        );
        let committed = input.accepted + 1;
        let next_position = self
            .before
            .target_position
            .checked_add(committed)
            .context("GLM committed position overflow")?;
        ensure!(
            input.target_position == next_position,
            "GLM committed target position mismatch"
        );
        let cache_end = self
            .before
            .cache_rows
            .checked_add(committed)
            .context("GLM committed cache overflow")?;
        ensure!(
            cache_end <= self.limits.cache_rows,
            "GLM full accepted prefix exceeds cache capacity"
        );
        ensure!(
            input.accepted <= self.limits.staging_rows,
            "GLM accepted hidden staging capacity is insufficient"
        );
        // Keep seed R; draft i reads TRUE target verify hidden i-1. The
        // correction/bonus belongs to the next proposal, not this write.
        let write = (input.accepted > 0).then_some(PairWrite {
            cache_start: self.before.cache_rows + 1,
            token_start: 1,
            hidden_start: 0,
            rows: input.accepted,
        });
        Ok(FinishPlan {
            state: PairState {
                generation: self.before.generation,
                cache_rows: cache_end,
                target_position: next_position,
            },
            write,
            bonus_hidden_row: Some(input.accepted),
            keep_seed: true,
        })
    }
}

/// Planned state is not live state. The adapter must stage source rows and
/// preserve the bonus before arena reuse, execute writes, then publish state.
/// Terminal requests may release the whole private cache instead of applying
/// a plan; this object has no cleanup, allocation or collective side effects.
#[derive(Clone, Copy, Debug)]
pub struct FinishPlan {
    state: PairState,
    write: Option<PairWrite>,
    bonus_hidden_row: Option<usize>,
    keep_seed: bool,
}

impl FinishPlan {
    pub fn state(self) -> PairState {
        self.state
    }
    pub fn write(self) -> Option<PairWrite> {
        self.write
    }
    pub fn bonus_hidden_row(self) -> Option<usize> {
        self.bonus_hidden_row
    }
    pub fn keep_seed(self) -> bool {
        self.keep_seed
    }
}

#[cfg(test)]
#[path = "glm_pair_plan_tests.rs"]
mod tests;
