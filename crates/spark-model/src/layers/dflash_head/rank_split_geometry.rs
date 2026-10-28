// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl Geometry {
    /// This plan for a propose of `rows` rows (`gamma` is the rows of a
    /// propose: a batched one runs B×gamma).
    pub(crate) fn with_rows(&self, rows: usize) -> Self {
        Self {
            gamma: rows,
            topk: self.topk && rows.is_multiple_of(self.gamma),
            ..*self
        }
    }

    /// Every row range is non-empty and starts on a 16-byte boundary of the
    /// packed twin (its K is a multiple of 32).
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.gamma > 0 && self.layers > 0 && (self.parts.mlp || self.parts.head),
            "ATLAS_GLM_DRAFT_TP: nothing to split ({self:?})"
        );
        ensure!(
            matches!(self.ranks, 2 | 4),
            "ATLAS_GLM_DRAFT_TP splits over the pair or the chain ({self:?})"
        );
        ensure!(
            self.hidden.is_multiple_of(32) && self.inter.is_multiple_of(32),
            "ATLAS_GLM_DRAFT_TP needs hidden and intermediate sizes in multiples of 32 ({self:?})"
        );
        ensure!(
            !self.parts.attn
                || self.parts.mlp
                    && self.q_dim.is_multiple_of(32)
                    && self.kv_dim.is_multiple_of(32)
                    && self.q_dim >= self.ranks * CTA_ROWS
                    && self.kv_dim >= self.ranks * CTA_ROWS,
            "ATLAS_GLM_DRAFT_TP attn needs mlp and projections the shares can serve ({self:?})"
        );
        let shares = self.ranks * CTA_ROWS;
        ensure!(
            self.vocab >= shares && self.hidden >= shares && self.inter >= shares,
            "ATLAS_GLM_DRAFT_TP: drafter too small to split {} ways ({self:?})",
            self.ranks
        );
        ensure!(
            (self.ctx_in, self.ctx_kv) == (0, 0)
                || self.ctx_in > 0 && self.ctx_in.is_multiple_of(32) && self.ctx_kv >= shares,
            "ATLAS_GLM_DRAFT_TP_CTX: context projections the shares cannot serve ({self:?})"
        );
        Ok(())
    }

    /// BF16 values per row of a swap's payload: every rank sends the widest
    /// share's width (a narrower share's tail is unused).
    fn width(&self, swap: Swap) -> usize {
        let widest = |n| widest(n, self.ranks);
        match swap {
            Swap::Input(_) | Swap::Hidden => self.hidden,
            Swap::Activation(_) => widest(self.inter),
            Swap::Output(_) => widest(self.hidden),
            Swap::Logits => widest(self.vocab),
            Swap::Candidates => CANDIDATE_ROW_BYTES / 2,
            Swap::CtxInput(_) => self.ctx_in,
            Swap::CtxHidden(_) => widest(self.hidden),
            Swap::CtxKv(_) => widest(self.ctx_kv),
            Swap::AttnIn(_) => self.hidden,
            Swap::Qkv(_) => widest(self.q_dim) + 2 * widest(self.kv_dim),
            Swap::AttnOut(_) => self.q_dim,
            Swap::OProj(_) => widest(self.hidden),
        }
    }

    /// Whether `swap` carries the head's rows out (the pair's one-way
    /// exchange, the chain's broadcast) rather than every rank's share.
    pub(crate) fn from_head(swap: Swap) -> bool {
        matches!(
            swap,
            Swap::Input(_) | Swap::Hidden | Swap::CtxInput(_) | Swap::AttnIn(_) | Swap::AttnOut(_)
        )
    }

    /// The rows a swap carries: a context append's own, else the propose's.
    pub(crate) fn rows(&self, swap: Swap) -> usize {
        match swap {
            Swap::CtxInput(i) | Swap::CtxHidden(i) | Swap::CtxKv(i) => self.ctx.get(i),
            _ => self.gamma,
        }
    }

    /// Bytes each rank sends and receives in `swap`.
    pub(crate) fn bytes(&self, swap: Swap) -> usize {
        self.rows(swap) * self.width(swap) * 2
    }

    /// Rank `rank`'s whole walk: the context appends (`ctx_steps`), then
    /// the layers and tail (`layer_steps`).
    pub(crate) fn steps(&self, rank: usize) -> Vec<Step> {
        let mut steps: Vec<Step> = self.ctx.appends().flat_map(ctx_steps).collect();
        steps.extend(self.layer_steps(rank));
        steps
    }

    /// Rank `rank`'s walk of the layers and tail. Rank 0 runs the whole
    /// propose forward; the other ranks only their shares.
    pub(crate) fn layer_steps(&self, rank: usize) -> Vec<Step> {
        use Piece::*;
        let head = rank == 0;
        let mut steps = Vec::new();
        // Pieces only the head runs, pieces both run, and swaps.
        let own = |steps: &mut Vec<Step>, piece| {
            if head {
                steps.push(Step::Run(piece));
            }
        };
        let both = |steps: &mut Vec<Step>, piece| steps.push(Step::Run(piece));
        let swap = |steps: &mut Vec<Step>, swap| steps.push(Step::Swap(swap));
        for l in 0..self.layers {
            if self.parts.attn {
                own(&mut steps, AttnPre(l));
                swap(&mut steps, Swap::AttnIn(l));
                both(&mut steps, Qkv(l));
                swap(&mut steps, Swap::Qkv(l));
                own(&mut steps, AttnCore(l));
                swap(&mut steps, Swap::AttnOut(l));
                both(&mut steps, OProj(l));
                swap(&mut steps, Swap::OProj(l));
                own(&mut steps, ProjectRest(l));
            } else {
                own(&mut steps, Attention(l));
            }
            if !self.parts.mlp {
                own(&mut steps, Post(l));
                continue;
            }
            if !self.parts.attn {
                own(&mut steps, Project(l));
            }
            swap(&mut steps, Swap::Input(l));
            both(&mut steps, GateUp(l));
            swap(&mut steps, Swap::Activation(l));
            both(&mut steps, Down(l));
            swap(&mut steps, Swap::Output(l));
            own(&mut steps, Residual(l));
        }
        if !self.parts.head {
            own(&mut steps, Tail);
            return steps;
        }
        own(&mut steps, Norm);
        swap(&mut steps, Swap::Hidden);
        both(&mut steps, Vocab);
        if self.topk {
            both(&mut steps, TopK);
            swap(&mut steps, Swap::Candidates);
        } else {
            swap(&mut steps, Swap::Logits);
        }
        own(&mut steps, Select);
        steps
    }

    /// The swaps of a propose, in order: the same list on every rank.
    pub(crate) fn swaps(&self) -> Vec<Swap> {
        swaps_of(&self.steps(1))
    }
}
