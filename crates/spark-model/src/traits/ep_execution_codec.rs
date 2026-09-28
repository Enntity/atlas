// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only legacy EP transcripts. No transport, live binding or wire generations.
//! Separate scalar/bulk calls are protocol boundaries, not a flat message.
//! Native legacy bulk interoperability assumes little-endian peers; serialization
//! here is explicit LE. Parsing does not authorize allocation or GPU execution.

use super::execution_plan::{SequentialSubstep, ValidatedStepPlan, WorkItem};
use anyhow::{Result, ensure};

#[path = "ep_execution_parser.rs"]
mod parser;
pub use parser::{LeWords, ValidatedHeader, parse_header};

const RESERVED_START: u32 = 0xffffffe0;
const PREFILL: u32 = 0xfffffff0;
const REPLACE: u32 = 0xfffffff1;
const BATCH: u32 = 0xffffffe0;
const SHUTDOWN: u32 = u32::MAX;
const ABSOLUTE_CONTROL_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegacyDialect {
    V1,
    V2,
}

/// Trusted caller-resolved limits, not message fields or serving authorization.
/// Control bytes bound one command and the total emitted plan transcript;
/// staging_bytes additionally bounds EACH bulk call, independently.
#[derive(Clone, Copy, Debug)]
pub struct LegacyWireLimits {
    pub slot_capacity: u32,
    pub vocab_size: u32,
    pub max_prompt_tokens: u32,
    pub max_chunk_tokens: u32,
    pub max_decode_rows: u32,
    pub max_control_bytes: usize,
    pub staging_bytes: usize,
}

impl LegacyWireLimits {
    fn validate(self) -> Result<()> {
        ensure!(
            self.slot_capacity > 0
                && self.vocab_size > 0
                && self.max_prompt_tokens > 0
                && self.max_chunk_tokens > 0
                && self.max_decode_rows > 0
                && self.staging_bytes > 0
                && self.max_control_bytes > 0
                && self.max_control_bytes <= ABSOLUTE_CONTROL_BYTES,
            "legacy codec limits must be positive and control bytes <=1MiB"
        );
        ensure!(
            self.max_decode_rows <= self.slot_capacity
                && self.max_chunk_tokens <= self.max_prompt_tokens,
            "legacy codec row/chunk limits exceed slot/prompt capacity"
        );
        Ok(())
    }
    fn token(self, token: u32) -> Result<()> {
        ensure!(
            token < self.vocab_size && token < RESERVED_START,
            "legacy codec token outside vocabulary or in reserved control region"
        );
        Ok(())
    }
    fn slot(self, slot: u32, dialect: LegacyDialect) -> Result<()> {
        ensure!(
            slot < self.slot_capacity && (dialect != LegacyDialect::V1 || slot == 0),
            "legacy codec slot outside capacity or V1 singleton slot0"
        );
        Ok(())
    }
}

fn checked_bytes(words: usize) -> Result<usize> {
    words
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("legacy codec word-byte overflow"))
}

/// A single existing collective. Words remain borrowed from the validated plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WireCall<'a> {
    U32(u32),
    Words(&'a [u32]),
}

impl WireCall<'_> {
    pub fn byte_len(self) -> Result<usize> {
        match self {
            Self::U32(_) => Ok(4),
            Self::Words(words) => checked_bytes(words.len()),
        }
    }
    /// Write exactly one call into caller-owned storage, preserving its boundary.
    pub fn write_le(self, destination: &mut [u8]) -> Result<()> {
        ensure!(
            destination.len() == self.byte_len()?,
            "legacy codec destination length mismatch"
        );
        match self {
            Self::U32(value) => destination.copy_from_slice(&value.to_le_bytes()),
            Self::Words(words) => {
                for (word, bytes) in words.iter().zip(destination.chunks_exact_mut(4)) {
                    bytes.copy_from_slice(&word.to_le_bytes());
                }
            }
        }
        Ok(())
    }
}

/// Encode with &[u32] payloads; parse with borrowed LeWords payloads. This has no
/// RequestKey generation, prompt revision, session, or step identity on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegacyCommand<P> {
    Decode {
        slot: u32,
        token: u32,
    },
    Prefill {
        slot: u32,
        chunk_start: u32,
        chunk_len: u32,
        prompt: P,
    },
    DecodeBatch {
        slots: P,
        tokens: P,
    },
    ReplaceSlot {
        slot: u32,
    },
    Shutdown,
}

pub struct WireCommand<'a> {
    calls: Vec<WireCall<'a>>,
    wire_bytes: usize,
}
impl<'a> WireCommand<'a> {
    pub fn calls(&self) -> &[WireCall<'a>] {
        &self.calls
    }
    pub fn wire_bytes(&self) -> usize {
        self.wire_bytes
    }
}

/// Standalone compatibility encoding. E0/control operations are deliberately
/// separate from encode_plan: no decode regrouping or lifecycle synthesis.
pub fn encode_command<'a>(
    command: LegacyCommand<&'a [u32]>,
    dialect: LegacyDialect,
    limits: LegacyWireLimits,
) -> Result<WireCommand<'a>> {
    limits.validate()?;
    let mut header = [0u32; 5];
    let mut payloads: [&[u32]; 2] = [&[], &[]];
    let (slot, opcode, extra, payload_count) = match command {
        LegacyCommand::Decode { slot, token } => {
            limits.token(token)?;
            (slot, token, 0, 0)
        }
        LegacyCommand::Prefill {
            slot,
            chunk_start,
            chunk_len,
            prompt,
        } => {
            let offset = if dialect == LegacyDialect::V2 { 2 } else { 1 };
            header[offset] = chunk_len;
            header[offset + 1] = chunk_start;
            header[offset + 2] = u32::try_from(prompt.len())?;
            payloads[0] = prompt;
            (slot, PREFILL, 3, 1)
        }
        LegacyCommand::DecodeBatch { slots, tokens } => {
            ensure!(
                dialect == LegacyDialect::V2,
                "legacy codec E0 requires canonical V2 sender"
            );
            ensure!(
                slots.len() == tokens.len(),
                "legacy codec batch slot/token length mismatch"
            );
            header[2] = u32::try_from(slots.len())?;
            payloads = [slots, tokens];
            (0, BATCH, 1, 2)
        }
        LegacyCommand::ReplaceSlot { slot } => (slot, REPLACE, 0, 0),
        LegacyCommand::Shutdown => (0, SHUTDOWN, 0, 0),
    };
    let base = if dialect == LegacyDialect::V2 {
        header[0] = slot;
        header[1] = opcode;
        2
    } else {
        limits.slot(slot, dialect)?;
        header[0] = opcode;
        1
    };
    let header = &header[..base + extra];
    let shape = parser::parse_words_header(dialect, header, limits)?;
    match command {
        LegacyCommand::Prefill { prompt, .. } => {
            for &token in prompt {
                limits.token(token)?;
            }
        }
        LegacyCommand::DecodeBatch { slots, tokens } => {
            parser::validate_batch(
                slots.iter().copied(),
                tokens.iter().copied(),
                slots.len(),
                limits,
            )?;
        }
        _ => (),
    }
    let mut calls = Vec::new();
    calls.try_reserve_exact(header.len() + payload_count)?;
    calls.extend(header.iter().copied().map(WireCall::U32));
    calls.extend(
        payloads[..payload_count]
            .iter()
            .map(|words| WireCall::Words(words)),
    );
    Ok(WireCommand {
        calls,
        wire_bytes: shape.wire_bytes(),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranscriptOp<'a> {
    Wire(WireCall<'a>),
    Local(SequentialSubstep),
}

pub struct LegacyTranscript<'a> {
    ops: Vec<TranscriptOp<'a>>,
    wire_bytes: usize,
}
impl<'a> LegacyTranscript<'a> {
    pub fn ops(&self) -> &[TranscriptOp<'a>] {
        &self.ops
    }
    pub fn wire_bytes(&self) -> usize {
        self.wire_bytes
    }
}

/// Preserve every stage-A work/substep boundary. Legacy worker prefill always
/// normalizes SSM, so a plan requesting Preserve for prefill is not encodable.
/// Host-only identities remain in the borrowed plan, not in this wire format.
pub fn encode_plan<'a>(
    plan: &'a ValidatedStepPlan,
    dialect: LegacyDialect,
    limits: LegacyWireLimits,
) -> Result<LegacyTranscript<'a>> {
    limits.validate()?;
    let mut ops = Vec::new();
    let mut wire_bytes = 0usize;
    for (index, &step) in plan.substeps().iter().enumerate() {
        let command = match step {
            SequentialSubstep::DecodeOne { work_index, .. }
            | SequentialSubstep::PrefillChunk { work_index, .. } => {
                let work = &plan.work()[work_index];
                let slot = work.key.wire_slot;
                Some(match work.kind {
                    WorkItem::DecodeOne { token } => LegacyCommand::Decode {
                        slot,
                        token: plan.tokens()[token.offset],
                    },
                    WorkItem::PrefillChunk {
                        prompt,
                        chunk_start,
                        chunk_len,
                        ..
                    } => {
                        ensure!(
                            plan.substeps().get(index + 1)
                                == Some(&SequentialSubstep::NormalizeSsm { work_index }),
                            "legacy worker prefill requires NormalizeSsm before next substep"
                        );
                        LegacyCommand::Prefill {
                            slot,
                            chunk_start: u32::try_from(chunk_start)?,
                            chunk_len: u32::try_from(chunk_len)?,
                            prompt: &plan.tokens()[prompt.offset..prompt.offset + prompt.count],
                        }
                    }
                })
            }
            _ => None,
        };
        if let Some(command) = command {
            let encoded = encode_command(command, dialect, limits)?;
            wire_bytes = wire_bytes
                .checked_add(encoded.wire_bytes())
                .ok_or_else(|| anyhow::anyhow!("legacy codec transcript byte overflow"))?;
            ensure!(
                wire_bytes <= limits.max_control_bytes,
                "legacy codec transcript exceeds control budget"
            );
            ops.try_reserve(encoded.calls.len() + 1)?;
            ops.extend(encoded.calls.into_iter().map(TranscriptOp::Wire));
        } else {
            ops.try_reserve(1)?;
        }
        ops.push(TranscriptOp::Local(step));
    }
    Ok(LegacyTranscript { ops, wire_bytes })
}

#[cfg(test)]
#[path = "ep_execution_codec_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "ep_execution_codec_edge_tests.rs"]
mod edge_tests;
