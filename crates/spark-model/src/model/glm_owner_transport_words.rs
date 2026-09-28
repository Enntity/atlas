// SPDX-License-Identifier: AGPL-3.0-only
//! Exact codec family adapter; one shared owner transport state machine.
use super::*;
use crate::model::glm_owner_wire::Mode;
use crate::model::glm_owner8_wire as wide;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::model) enum Payload {
    E7([u32; wire::PAYLOAD_WORDS]),
    E8([u32; wide::PAYLOAD_WORDS]),
}

pub(super) struct Records {
    pub shape: GlmOwnerBatchShape,
    pub mode: Mode,
    pub owners: [Option<OwnerRecord>; 8],
}

impl Payload {
    pub(super) fn encode(
        shape: GlmOwnerBatchShape,
        mode: Mode,
        owners: [Option<OwnerRecord>; 8],
        bounds: Bounds,
    ) -> Result<Self> {
        if shape.owners() <= 4 {
            ensure!(
                owners[4..].iter().all(Option::is_none),
                "E7 excess owner records"
            );
            Ok(Self::E7(
                wire::Packet {
                    shape,
                    mode,
                    owners: owners[..4].try_into()?,
                }
                .encode(bounds)?,
            ))
        } else {
            Ok(Self::E8(
                wide::Packet {
                    shape,
                    mode,
                    owners,
                }
                .encode(bounds)?,
            ))
        }
    }

    pub(super) fn empty(command: u32) -> Result<Self> {
        match command {
            wire::EP_GLM_OWNER_VERIFY => Ok(Self::E7([0; wire::PAYLOAD_WORDS])),
            wide::EP_GLM_OWNER8_VERIFY => Ok(Self::E8([0; wide::PAYLOAD_WORDS])),
            _ => anyhow::bail!("unknown owner transport command"),
        }
    }

    pub(super) fn command(self) -> u32 {
        match self {
            Self::E7(_) => wire::EP_GLM_OWNER_VERIFY,
            Self::E8(_) => wide::EP_GLM_OWNER8_VERIFY,
        }
    }

    pub(super) fn exchange(&mut self, model: &TransformerModel) -> Result<()> {
        match self {
            Self::E7(words) => model.owner_exchange_words(words),
            Self::E8(words) => model.owner_exchange_words(words),
        }
    }

    pub(super) fn decode(&self, bounds: Bounds) -> Result<Records> {
        match self {
            Self::E7(words) => {
                let packet = wire::Packet::decode(words, bounds)?;
                let mut owners = [None; 8];
                owners[..4].copy_from_slice(&packet.owners);
                Ok(Records {
                    shape: packet.shape,
                    mode: packet.mode,
                    owners,
                })
            }
            Self::E8(words) => {
                let packet = wide::Packet::decode(words, bounds)?;
                Ok(Records {
                    shape: packet.shape,
                    mode: packet.mode,
                    owners: packet.owners,
                })
            }
        }
    }
}

pub(super) fn exchange_verdict(
    model: &TransformerModel,
    shape: GlmOwnerBatchShape,
    accepted: Option<&[usize]>,
) -> Result<[usize; 8]> {
    let mut counts = [0; 8];
    if let Some(accepted) = accepted {
        ensure!(
            accepted.len() == shape.owners(),
            "owner verdict count mismatch"
        );
        counts[..accepted.len()].copy_from_slice(accepted);
    }
    if shape.owners() <= 4 {
        let mut words = if accepted.is_some() {
            wire::encode_verdict(shape, counts[..4].try_into()?)?
        } else {
            [0; wire::VERDICT_WORDS]
        };
        model.owner_exchange_words(&mut words)?;
        counts[..4].copy_from_slice(&wire::decode_verdict(&words, shape)?);
    } else {
        let mut words = if accepted.is_some() {
            wide::encode_verdict(shape, counts)?
        } else {
            [0; wide::VERDICT_WORDS]
        };
        model.owner_exchange_words(&mut words)?;
        counts = wide::decode_verdict(&words, shape)?;
    }
    Ok(counts)
}
