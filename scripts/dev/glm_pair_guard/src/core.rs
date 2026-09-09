// SPDX-License-Identifier: AGPL-3.0-only

use crate::frame::{Frame, CHALLENGE, HELLO, RENEW, REVOKE, START};

#[derive(Clone, Copy)]
pub struct Policy {
    pub startup: u64,
    pub lease: u64,
    pub challenge: u64,
    pub frame: u64,
    pub campaign: u64,
    pub poll: u64,
    pub reap: u64,
}

impl Policy {
    pub fn validate(self) -> Result<Self, &'static str> {
        let all = [
            self.startup,
            self.lease,
            self.challenge,
            self.frame,
            self.campaign,
            self.poll,
            self.reap,
        ];
        if all.contains(&0)
            || all.iter().any(|&x| x > 86_400_000)
            || self.poll > 250
            || self.challenge >= self.lease
            || self.frame > self.lease
            || self.startup > self.campaign
            || self.lease > self.campaign
            || self.reap > 10_000
        {
            return Err("invalid explicit policy");
        }
        Ok(self)
    }
}

pub struct State {
    policy: Policy,
    session: [u8; 32],
    instance: [u8; 32],
    ordinal: u64,
    outstanding: Option<(u64, [u8; 32])>,
    next_issue: u64,
    deadline: u64,
    campaign_end: u64,
    last_now: u64,
    running: bool,
    terminal: bool,
}

impl State {
    pub fn new(
        now: u64,
        policy: Policy,
        session: [u8; 32],
        instance: [u8; 32],
        challenge: [u8; 32],
    ) -> Result<(Self, Frame), &'static str> {
        let policy = policy.validate()?;
        let deadline = now.checked_add(policy.startup).ok_or("time overflow")?;
        let campaign_end = now.checked_add(policy.campaign).ok_or("time overflow")?;
        let state = Self {
            policy,
            session,
            instance,
            ordinal: 0,
            outstanding: Some((now, challenge)),
            next_issue: now,
            deadline,
            campaign_end,
            last_now: now,
            running: false,
            terminal: false,
        };
        let hello = state.frame(HELLO, challenge);
        Ok((state, hello))
    }
    fn frame(&self, kind: u8, challenge: [u8; 32]) -> Frame {
        Frame {
            kind,
            session: self.session,
            instance: self.instance,
            ordinal: self.ordinal,
            challenge,
        }
    }
    pub fn stop(&mut self) {
        self.terminal = true;
    }
    pub fn check(&mut self, now: u64) -> Result<(), &'static str> {
        if self.terminal || now < self.last_now || now >= self.deadline || now >= self.campaign_end
        {
            self.stop();
            return Err("terminal or expired lease");
        }
        self.last_now = now;
        Ok(())
    }
    pub fn accept(&mut self, now: u64, frame: &Frame) -> Result<bool, &'static str> {
        let result = self.accept_inner(now, frame);
        if result.is_err() {
            self.stop();
        }
        result
    }
    fn accept_inner(&mut self, now: u64, frame: &Frame) -> Result<bool, &'static str> {
        self.check(now)?;
        if frame.session != self.session || frame.instance != self.instance {
            return Err("wrong local identity");
        }
        if frame.kind == REVOKE {
            return Err("revoked");
        }
        let expected = if self.running { RENEW } else { START };
        let (issued, nonce) = self.outstanding.ok_or("no outstanding challenge")?;
        if frame.kind != expected || frame.ordinal != self.ordinal || frame.challenge != nonce {
            return Err("invalid challenge response");
        }
        let deadline = issued
            .checked_add(self.policy.lease)
            .ok_or("time overflow")?
            .min(self.campaign_end);
        if now >= deadline {
            return Err("expired challenge response");
        }
        let release = !self.running;
        self.running = true;
        self.deadline = deadline;
        self.outstanding = None;
        self.next_issue = now
            .checked_add(self.policy.challenge)
            .ok_or("time overflow")?;
        Ok(release)
    }
    pub fn needs_challenge(&self, now: u64) -> bool {
        self.running && self.outstanding.is_none() && now >= self.next_issue
    }
    pub fn issue(&mut self, now: u64, nonce: [u8; 32]) -> Result<Frame, &'static str> {
        self.check(now)?;
        if !self.needs_challenge(now) {
            self.stop();
            return Err("challenge not due");
        }
        self.ordinal = match self.ordinal.checked_add(1) {
            Some(value) => value,
            None => {
                self.stop();
                return Err("ordinal overflow");
            }
        };
        self.outstanding = Some((now, nonce));
        Ok(self.frame(CHALLENGE, nonce))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn policy() -> Policy {
        Policy {
            startup: 100,
            lease: 1000,
            challenge: 100,
            frame: 50,
            campaign: 2000,
            poll: 10,
            reap: 100,
        }
    }
    fn started() -> State {
        let (mut s, mut hello) = State::new(0, policy(), [1; 32], [2; 32], [3; 32]).unwrap();
        hello.kind = START;
        assert!(s.accept(0, &hello).unwrap());
        s
    }
    #[test]
    fn exact_deadline_is_terminal_before_renewal_or_parse() {
        let mut s = started();
        assert!(s.check(1000).is_err());
        assert!(s.check(999).is_err());
    }
    #[test]
    fn issue_time_not_arrival_and_no_replay() {
        let mut s = started();
        let mut f = s.issue(100, [4; 32]).unwrap();
        f.kind = RENEW;
        assert!(!s.accept(900, &f).unwrap());
        assert!(s.check(1101).is_err());
        assert!(s.accept(1101, &f).is_err());
        let mut s = started();
        let mut f = s.issue(100, [5; 32]).unwrap();
        f.kind = RENEW;
        s.accept(101, &f).unwrap();
        assert!(s.accept(102, &f).is_err());
    }
    #[test]
    fn wrong_identity_campaign_regression_and_overflow_are_terminal() {
        for field in 0..3 {
            let mut s = started();
            let mut f = s.issue(100, [4; 32]).unwrap();
            f.kind = RENEW;
            match field {
                0 => f.session[0] ^= 1,
                1 => f.instance[0] ^= 1,
                _ => f.challenge[0] ^= 1,
            };
            assert!(s.accept(101, &f).is_err());
            assert!(s.check(102).is_err());
        }
        let mut s = started();
        s.ordinal = u64::MAX;
        assert!(s.issue(100, [4; 32]).is_err());
        assert!(s.check(101).is_err());
        let mut s = started();
        s.check(50).unwrap();
        assert!(s.check(49).is_err());
        let mut s = started();
        for now in [100, 500, 900, 1300, 1700] {
            let mut f = s.issue(now, [4; 32]).unwrap();
            f.kind = RENEW;
            s.accept(now, &f).unwrap();
        }
        assert!(s.check(2000).is_err());
        assert!(State::new(u64::MAX, policy(), [1; 32], [2; 32], [3; 32]).is_err());
    }
}
