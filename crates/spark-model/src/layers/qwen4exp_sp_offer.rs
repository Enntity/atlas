// SPDX-License-Identifier: AGPL-3.0-only
//! Reduce-scatter offers between a caller and the site computing its partial
//! (`ATLAS_QWEN4EXP_PREFILL_SP_RS_PIPE`; split from `qwen4exp_sp_pipe`).

use std::cell::Cell;

thread_local! {
    /// 1: a caller offered its reduce-scatter to the site producing the
    /// partial ([`RsOffer`]); 2: the site took it.
    static OFFER: Cell<u8> = const { Cell::new(0) };
}

/// A caller about to reduce-scatter a partial lets the site that computes
/// it run `compute_and_reduce_scatter` itself; [`RsOffer::taken`] says
/// whether it did (then the caller must not reduce again). Withdrawn on drop,
/// so a site reached from any other caller never takes one.
pub struct RsOffer(());

impl RsOffer {
    /// Offer the caller's reduce-scatter until the offer drops.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        OFFER.with(|c| c.set(1));
        Self(())
    }
    /// The site ran the reduce-scatter.
    pub fn taken(&self) -> bool {
        OFFER.with(Cell::get) == 2
    }
}

impl Drop for RsOffer {
    fn drop(&mut self) {
        OFFER.with(|c| c.set(0));
    }
}

/// Whether the current caller offered its reduce-scatter.
pub fn rs_offered() -> bool {
    OFFER.with(Cell::get) == 1
}

/// The offered reduce-scatter ran in the site.
pub fn rs_took() {
    OFFER.with(|c| c.set(2));
}
