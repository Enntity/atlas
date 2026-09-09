// SPDX-License-Identifier: AGPL-3.0-only

//! T3.1 local control only: these bytes confer no Docker or Model authority.
pub const LEN: usize = 112;
pub const HELLO: u8 = 1;
pub const START: u8 = 2;
pub const RENEW: u8 = 3;
pub const CHALLENGE: u8 = 4;
pub const REVOKE: u8 = 5;

/// Check the original frame window at the next observed clock, including after
/// I/O/decode. A partial transfer never earns a new deadline.
pub fn check_deadline(started: u64, observed: u64, limit: u64) -> Result<(), &'static str> {
    if observed
        .checked_sub(started)
        .is_some_and(|elapsed| elapsed < limit)
    {
        Ok(())
    } else {
        Err("frame deadline or clock regression")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub kind: u8,
    pub session: [u8; 32],
    pub instance: [u8; 32],
    pub ordinal: u64,
    pub challenge: [u8; 32],
}

impl Frame {
    pub fn encode(&self) -> [u8; LEN] {
        let mut out = [0; LEN];
        out[..4].copy_from_slice(&((LEN - 4) as u32).to_be_bytes());
        out[4] = 1;
        out[5] = self.kind;
        out[8..40].copy_from_slice(&self.session);
        out[40..72].copy_from_slice(&self.instance);
        out[72..80].copy_from_slice(&self.ordinal.to_be_bytes());
        out[80..].copy_from_slice(&self.challenge);
        out
    }

    pub fn decode(bytes: &[u8; LEN]) -> Result<Self, &'static str> {
        if bytes[..4] != ((LEN - 4) as u32).to_be_bytes()
            || bytes[4] != 1
            || bytes[6..8] != [0, 0]
            || !(HELLO..=REVOKE).contains(&bytes[5])
        {
            return Err("invalid frame header");
        }
        Ok(Self {
            kind: bytes[5],
            session: bytes[8..40].try_into().unwrap(),
            instance: bytes[40..72].try_into().unwrap(),
            ordinal: u64::from_be_bytes(bytes[72..80].try_into().unwrap()),
            challenge: bytes[80..].try_into().unwrap(),
        })
    }
}

/// Fixed storage, no length-driven allocation and no arrival-time extension.
pub struct Reader {
    bytes: [u8; LEN],
    used: usize,
    pub started: Option<u64>,
}

impl Reader {
    pub fn new() -> Self {
        Self {
            bytes: [0; LEN],
            used: 0,
            started: None,
        }
    }
    pub fn remaining(&mut self) -> &mut [u8] {
        &mut self.bytes[self.used..]
    }
    pub fn received(&mut self, n: usize, now: u64) -> Result<Option<Frame>, &'static str> {
        if n == 0 || n > LEN - self.used {
            return Err("invalid receive count");
        }
        self.started.get_or_insert(now);
        self.used += n;
        if self.used >= 4 && self.bytes[..4] != ((LEN - 4) as u32).to_be_bytes() {
            return Err("invalid frame length");
        }
        if self.used != LEN {
            return Ok(None);
        }
        let frame = Frame::decode(&self.bytes)?;
        self.used = 0;
        self.started = None;
        Ok(Some(frame))
    }
}
