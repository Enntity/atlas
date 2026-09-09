// SPDX-License-Identifier: AGPL-3.0-only
//! Pointer-free transcript of the shared recorder. No numerical or completion proof.
use super::inner;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    Alloc(usize),
    Free,
    Copy(usize, u64),
    Upload(usize, u64),
    Read(usize, u64),
    Sync(u64),
    RecordEvent(u64),
    WaitEvent(u64),
    BeginCapture(u64),
    EndCapture(u64),
    AbortCapture(u64),
    LaunchGraph(u64),
    DestroyGraph,
    AllocState(bool),
    Memset(usize, u64),
    Kernel(String, u64),
    Target(usize, usize, u64),
    Body(usize, u64),
    Kv(usize, u64),
}
impl From<inner::Event> for Event {
    fn from(e: inner::Event) -> Self {
        use inner::Event as E;
        match e {
            E::Alloc(_, n) => Self::Alloc(n),
            E::Free(_) => Self::Free,
            E::Copy(_, _, n, s) => Self::Copy(n, s),
            E::Upload(_, n, s) => Self::Upload(n, s),
            E::Read(_, n, s) => Self::Read(n, s),
            E::Sync(s) => Self::Sync(s),
            E::RecordEvent(_, s) => Self::RecordEvent(s),
            E::WaitEvent(s, _) => Self::WaitEvent(s),
            E::BeginCapture(s) => Self::BeginCapture(s),
            E::EndCapture(s) => Self::EndCapture(s),
            E::AbortCapture(s) => Self::AbortCapture(s),
            E::LaunchGraph(_, s) => Self::LaunchGraph(s),
            E::DestroyGraph(_) => Self::DestroyGraph,
            E::AllocState(b) => Self::AllocState(b),
            E::Memset(_, n, s) => Self::Memset(n, s),
            E::Kernel(name, _, s) => Self::Kernel(name, s),
            E::Target(n, p, s) => Self::Target(n, p, s),
            E::Body(p, s) => Self::Body(p, s),
            E::Kv(slots, s) => Self::Kv(slots.len(), s),
        }
    }
}
