// SPDX-License-Identifier: AGPL-3.0-only
//! Only cold factory/constructor ownership; no allocator or runtime error policy.
use std::ops::{Deref, DerefMut};

/// Cold-start ownership only. A retained failure must lead to terminal process
/// exit, never retry/reuse. This does not intercept cleanup inside a callee.
#[doc(hidden)]
pub struct ColdOwner<T> {
    value: Option<T>,
    retain_on_error: bool,
}
impl<T> ColdOwner<T> {
    pub fn new(value: T, retain_on_error: bool) -> Self {
        Self {
            value: Some(value),
            retain_on_error,
        }
    }
    pub fn into_inner(mut self) -> T {
        self.value.take().expect("live construction owner")
    }
}
impl<T> Deref for ColdOwner<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value.as_ref().expect("live construction owner")
    }
}
impl<T> DerefMut for ColdOwner<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value.as_mut().expect("live construction owner")
    }
}
impl<T> Drop for ColdOwner<T> {
    fn drop(&mut self) {
        if self.retain_on_error
            && let Some(value) = self.value.take()
        {
            // Selected startup must immediately reach its armed terminal sink.
            // Dropping native owners here could free work before that caller sees Err.
            std::mem::forget(value);
        }
    }
}
