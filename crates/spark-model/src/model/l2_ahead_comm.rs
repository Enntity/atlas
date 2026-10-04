// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_L2_AHEAD` sites `a` and `f` (`layers::ops::l2_ahead`): the
//! communicator an eager K-gamma verify hands its layers. Every call goes to
//! the real communicator unchanged; before a layer's first all-reduce (its
//! attention's) it forks the prefetch of the layer's FFN leading weights,
//! before its second (the FFN's) the next layer's leading attention weights.
//! A layer with one all-reduce (a replicated FFN) only takes the first; a
//! third takes nothing. Wrong guesses only cost time: a prefetch reads, and
//! the main stream never waits on it.

use anyhow::Result;
use parking_lot::Mutex;
use spark_comm::CommBackend;

use crate::layers::ops::{L2Region, L2Site};

/// Forks one prefetch: `(site, regions, compute stream)`.
pub(crate) type Fire<'a> = dyn Fn(L2Site, &[L2Region], u64) -> Result<()> + Sync + 'a;

/// The sites in a layer's all-reduce order.
const SITES: [L2Site; 2] = [L2Site::Attn, L2Site::Ffn];

pub(crate) struct L2AheadComm<'a> {
    inner: &'a dyn CommBackend,
    fire: &'a Fire<'a>,
    /// The current layer's prefetches in [`SITES`] order, and its all-reduces
    /// so far.
    layer: Mutex<([Vec<L2Region>; 2], usize)>,
}

impl<'a> L2AheadComm<'a> {
    /// `None` when sites `a` and `f` are both off.
    pub(crate) fn new(inner: &'a dyn CommBackend, fire: &'a Fire<'a>) -> Option<Self> {
        SITES
            .iter()
            .any(|&site| crate::layers::ops::l2_ahead_enabled(site))
            .then(|| Self::with(inner, fire))
    }

    fn with(inner: &'a dyn CommBackend, fire: &'a Fire<'a>) -> Self {
        Self {
            inner,
            fire,
            layer: Mutex::new(Default::default()),
        }
    }

    /// A layer starts: what its FFN reads first, and what the next layer's
    /// attention reads first.
    pub(crate) fn begin_layer(&self, ffn: Vec<L2Region>, next_attention: Vec<L2Region>) {
        *self.layer.lock() = ([ffn, next_attention], 0);
    }

    fn before_all_reduce(&self, stream: u64) -> Result<()> {
        let (k, regions) = {
            let mut layer = self.layer.lock();
            let k = layer.1;
            layer.1 += 1;
            (
                k,
                layer.0.get_mut(k).map(std::mem::take).unwrap_or_default(),
            )
        };
        if regions.is_empty() {
            return Ok(());
        }
        (self.fire)(SITES[k], &regions, stream)
    }
}

impl CommBackend for L2AheadComm<'_> {
    fn all_reduce(&self, ptr: u64, bytes: usize) -> Result<()> {
        // No stream to fork from (the capture paths): its prefetch is skipped.
        self.layer.lock().1 += 1;
        self.inner.all_reduce(ptr, bytes)
    }
    fn all_reduce_async(&self, ptr: u64, bytes: usize, stream: u64) -> Result<()> {
        self.before_all_reduce(stream)?;
        self.inner.all_reduce_async(ptr, bytes, stream)
    }
    fn all_gather(&self, send: u64, recv: u64, bytes: usize) -> Result<()> {
        self.inner.all_gather(send, recv, bytes)
    }
    fn reduce_scatter(&self, send: u64, recv: u64, bytes: usize) -> Result<()> {
        self.inner.reduce_scatter(send, recv, bytes)
    }
    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()> {
        self.inner.broadcast(ptr, bytes, root)
    }
    fn receive_idle_command_word(&self, ptr: u64) -> Result<()> {
        self.inner.receive_idle_command_word(ptr)
    }
    fn command_words_max(&self) -> usize {
        self.inner.command_words_max()
    }
    fn send_command_words(&self, words: &[u32]) -> Result<()> {
        self.inner.send_command_words(words)
    }
    fn recv_command_words(&self, words: &mut [u32]) -> Result<()> {
        self.inner.recv_command_words(words)
    }
    fn barrier(&self) -> Result<()> {
        self.inner.barrier()
    }
    fn peer_exchange_async(&self, send: u64, recv: u64, bytes: usize, stream: u64) -> Result<()> {
        self.inner.peer_exchange_async(send, recv, bytes, stream)
    }
    fn supports_peer_exchange_async(&self) -> bool {
        self.inner.supports_peer_exchange_async()
    }
    fn exchange_async(
        &self,
        send: u64,
        dst: u64,
        bytes: usize,
        add: bool,
        stream: u64,
    ) -> Result<bool> {
        self.inner.exchange_async(send, dst, bytes, add, stream)
    }
    fn all_reduce_capturable(&self, ptr: u64, bytes: usize, stream: u64) -> Result<bool> {
        self.inner.all_reduce_capturable(ptr, bytes, stream)
    }
    fn peer_exchange_capturable(
        &self,
        send: u64,
        recv: u64,
        bytes: usize,
        stream: u64,
    ) -> Result<bool> {
        self.inner
            .peer_exchange_capturable(send, recv, bytes, stream)
    }
    fn capturable_all_reduce_max_bytes(&self) -> usize {
        self.inner.capturable_all_reduce_max_bytes()
    }
    fn set_oneshot_kernel(&self, handle: u64) {
        self.inner.set_oneshot_kernel(handle)
    }
    fn supports_exchange_async(&self, bytes: usize) -> bool {
        self.inner.supports_exchange_async(bytes)
    }
    fn register_buffer(&self, ptr: u64, bytes: usize) -> Result<u64> {
        self.inner.register_buffer(ptr, bytes)
    }
    fn deregister_buffer(&self, handle: u64) -> Result<()> {
        self.inner.deregister_buffer(handle)
    }
    fn symmetric_alloc(&self, bytes: usize) -> Result<u64> {
        self.inner.symmetric_alloc(bytes)
    }
    fn symmetric_free(&self, ptr: u64) -> Result<()> {
        self.inner.symmetric_free(ptr)
    }
    fn set_add_kernel(&self, handle: u64) {
        self.inner.set_add_kernel(handle)
    }
    fn send_to(&self, ptr: u64, bytes: usize, dest: usize, stream: u64) -> Result<()> {
        self.inner.send_to(ptr, bytes, dest, stream)
    }
    fn recv_from(&self, ptr: u64, bytes: usize, src: usize, stream: u64) -> Result<()> {
        self.inner.recv_from(ptr, bytes, src, stream)
    }
    fn group_start(&self) -> Result<()> {
        self.inner.group_start()
    }
    fn group_end(&self) -> Result<()> {
        self.inner.group_end()
    }
    fn is_healthy(&self) -> bool {
        self.inner.is_healthy()
    }
    fn attempt_reconnect(&self) -> Result<()> {
        self.inner.attempt_reconnect()
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn world_size(&self) -> usize {
        self.inner.world_size()
    }
}

#[cfg(test)]
#[path = "l2_ahead_comm_tests.rs"]
mod tests;
