// SPDX-License-Identifier: AGPL-3.0-only

use crate::codec::{Fixed, Writer};
use crate::*;
use sha2::{Digest as _, Sha256};

fn domain(label: &str, bytes: &[u8]) -> Digest {
    let mut h = Sha256::new();
    h.update(label.as_bytes());
    h.update([0]);
    h.update((bytes.len() as u32).to_be_bytes());
    h.update(bytes);
    h.finalize().into()
}
fn record<T: Fixed>(label: &str, value: &T) -> Result<Digest> {
    let mut bytes = [0; MAX_ENCODED];
    value.put(&mut Writer(&mut bytes[..T::LEN]))?;
    Ok(domain(label, &bytes[..T::LEN]))
}
/// Hashes already-canonical bounded recipe bytes; does not validate the recipe schema.
pub fn recipe_digest(bytes: &[u8]) -> Result<Digest> {
    if bytes.is_empty() || bytes.len() > 65536 {
        return Err(Error("recipe size"));
    }
    Ok(domain("atlas.glm.pair.recipe.v1", bytes))
}
pub fn policy_digest(value: &Policy) -> Result<Digest> {
    value.validate()?;
    record("atlas.glm.pair.policy.v1", value)
}
pub fn manifest_digest(value: &Manifest) -> Result<Digest> {
    value.validate()?;
    record("atlas.glm.pair.manifest.v1", value)
}
pub fn ticket_digest(rank: u8, value: &ChildTicket) -> Result<Digest> {
    let bytes = Frame {
        rank,
        body: Body::ChildTicket(*value),
    }
    .encode()?;
    Ok(domain("atlas.glm.pair.ticket.v1", bytes.as_slice()))
}
pub fn quiescent_digest(value: &QuiescentFrame) -> Result<Digest> {
    value.receipt.validate_fields()?;
    record("atlas.glm.pair.quiescent.v1", value)
}
pub fn release_digest(value: &PairRelease) -> Result<Digest> {
    value.validate_fields()?;
    record("atlas.glm.pair.release.v1", value)
}
