// SPDX-License-Identifier: AGPL-3.0-only
//! Fixed CPU byte fixture, not native numerics or a selected-serving entry.
#[path = "glm_c2_handoff_test_fixture.rs"]
pub(crate) mod fixture;
#[path = "glm_c2_test_wire.rs"]
pub(crate) mod wire;

#[cfg(feature = "glm-c2-test-utils")]
#[path = "glm_c2_test_facade.rs"]
mod facade;
#[cfg(feature = "glm-c2-test-utils")]
pub use facade::{Event, Fixture, Observer, Snapshot, Wire};
