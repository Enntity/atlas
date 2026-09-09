// SPDX-License-Identifier: AGPL-3.0-only

use crate::codec::{Fixed, Reader, Writer};
use crate::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    StartupFile,
    ControllerToGuard,
    GuardToController,
    ChildToGuard,
    GuardToChild,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Body {
    Startup(StartupRecord),
    GatedReport(GatedReport),
    PairedStart(Manifest),
    ChildHello(ChildHello),
    ChildTicket(ChildTicket),
    Quiescent(Quiescent),
    PairRelease(PairRelease),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame {
    pub rank: u8,
    pub body: Body,
}

pub struct Encoded {
    bytes: [u8; MAX_ENCODED],
    len: usize,
}
impl Encoded {
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

pub(crate) fn put_header(w: &mut Writer<'_>, len: usize, kind: u8, rank: u8) -> Result<()> {
    if rank > 1 {
        return Err(Error("invalid rank"));
    }
    w.field(&((len - 4) as u32))?;
    w.bytes(b"GLP2")?;
    w.bytes(&[0, 1, kind, rank])?;
    w.field(&0u32)
}

pub(crate) fn get_header(r: &mut Reader<'_>, len: usize, kind: u8) -> Result<u8> {
    if r.field::<u32>()? as usize != len - 4 || r.bytes::<4>()? != *b"GLP2" {
        return Err(Error("invalid frame prefix or magic"));
    }
    let [v0, v1, actual_kind, rank] = r.bytes()?;
    if [v0, v1] != [0, 1] || actual_kind != kind || rank > 1 || r.field::<u32>()? != 0 {
        return Err(Error("invalid frame header"));
    }
    Ok(rank)
}

impl Frame {
    pub fn encode(&self) -> Result<Encoded> {
        self.validate_fields()?;
        let (kind, len) = match &self.body {
            Body::Startup(_) => (0x01, StartupRecord::LEN),
            Body::GatedReport(_) => (0x10, GatedReport::LEN),
            Body::PairedStart(_) => (0x11, Manifest::LEN),
            Body::ChildHello(_) => (0x12, ChildHello::LEN),
            Body::ChildTicket(_) => (0x13, ChildTicket::LEN),
            Body::Quiescent(_) => (0x14, Quiescent::LEN),
            Body::PairRelease(_) => (0x15, PairRelease::LEN),
        };
        let mut out = Encoded {
            bytes: [0; MAX_ENCODED],
            len: HEADER_LEN + len,
        };
        let mut w = Writer(&mut out.bytes[..out.len]);
        put_header(&mut w, HEADER_LEN + len, kind, self.rank)?;
        match &self.body {
            Body::Startup(v) => w.field(v),
            Body::GatedReport(v) => w.field(v),
            Body::PairedStart(v) => w.field(v),
            Body::ChildHello(v) => w.field(v),
            Body::ChildTicket(v) => w.field(v),
            Body::Quiescent(v) => w.field(v),
            Body::PairRelease(v) => w.field(v),
        }?;
        if !w.0.is_empty() {
            return Err(Error("internal encoded length mismatch"));
        }
        Ok(out)
    }
    pub fn decode(bytes: &[u8], direction: Direction) -> Result<Self> {
        if bytes.len() < HEADER_LEN || bytes.len() > MAX_ENCODED {
            return Err(Error("frame size"));
        }
        let kind = bytes[10];
        let len = kind_len(kind)?;
        if len != bytes.len() || !direction.allows(kind) {
            return Err(Error("frame length or direction"));
        }
        let mut r = Reader(bytes);
        let rank = get_header(&mut r, len, kind)?;
        let body = match kind {
            0x01 => Body::Startup(r.field()?),
            0x10 => Body::GatedReport(r.field()?),
            0x11 => Body::PairedStart(r.field()?),
            0x12 => Body::ChildHello(r.field()?),
            0x13 => Body::ChildTicket(r.field()?),
            0x14 => Body::Quiescent(r.field()?),
            0x15 => Body::PairRelease(r.field()?),
            _ => return Err(Error("unknown frame kind")),
        };
        if !r.0.is_empty() {
            return Err(Error("trailing frame bytes"));
        }
        let value = Self { rank, body };
        value.validate_fields()?;
        Ok(value)
    }
    fn validate_fields(&self) -> Result<()> {
        if self.rank > 1 {
            return Err(Error("invalid rank"));
        }
        match &self.body {
            Body::Startup(v) => v.validate(),
            Body::GatedReport(v) => {
                crate::validate::nonzero(&v.pair_session)?;
                v.record.validate()
            }
            Body::PairedStart(v) => v.validate(),
            Body::ChildHello(v) => v.validate(),
            Body::ChildTicket(v) => v.validate(),
            Body::Quiescent(v) => v.validate_fields(),
            Body::PairRelease(v) => v.validate_fields(),
        }
    }
}

/// Only bounds a control-stream frame; legacy bytes still require the old parser.
pub fn control_frame_len(prefix: [u8; 4], direction: Direction) -> Result<usize> {
    if !matches!(
        direction,
        Direction::ControllerToGuard | Direction::GuardToController
    ) {
        return Err(Error("not a control direction"));
    }
    let len = u32::from_be_bytes(prefix)
        .checked_add(4)
        .ok_or(Error("frame length overflow"))?;
    if len == 112 {
        return Ok(112);
    }
    for kind in [0x10, 0x11, 0x14, 0x15] {
        if direction.allows(kind) && kind_len(kind)? == len as usize {
            return Ok(len as usize);
        }
    }
    Err(Error("invalid control prefix"))
}

fn kind_len(kind: u8) -> Result<usize> {
    let body = match kind {
        0x01 => StartupRecord::LEN,
        0x10 => GatedReport::LEN,
        0x11 => Manifest::LEN,
        0x12 => ChildHello::LEN,
        0x13 => ChildTicket::LEN,
        0x14 => Quiescent::LEN,
        0x15 => PairRelease::LEN,
        _ => return Err(Error("unknown frame kind")),
    };
    Ok(HEADER_LEN + body)
}
impl Direction {
    fn allows(self, kind: u8) -> bool {
        match self {
            Self::StartupFile => kind == 0x01,
            Self::ControllerToGuard => matches!(kind, 0x11 | 0x15),
            Self::GuardToController => matches!(kind, 0x10 | 0x14),
            Self::ChildToGuard => matches!(kind, 0x12 | 0x14),
            Self::GuardToChild => matches!(kind, 0x13 | 0x15),
        }
    }
}
