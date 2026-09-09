// SPDX-License-Identifier: AGPL-3.0-only
//! Real local pipe children exchange existing codec bytes, not Guard authority.
use super::*;
use std::ops::{Deref, DerefMut};
use std::time::Duration;

fn limits() -> Limits {
    Limits {
        frame_ms: 1000,
        poll_ms: 5,
        campaign_ms: 5000,
        stderr_bytes: 64,
        stderr_per_turn: 8,
        reap_ms: 1000,
    }
}
fn script(code: &str, args: Vec<String>) -> process::Spec {
    process::Spec {
        program: "/usr/bin/python3".into(),
        args: [vec!["-c".into(), code.into()], args]
            .concat()
            .into_iter()
            .map(Into::into)
            .collect(),
        env: vec![],
        stdin: vec![],
    }
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn hello() -> frame::Frame {
    frame::Frame {
        kind: frame::HELLO,
        session: [1; 32],
        instance: [2; 32],
        ordinal: 0,
        challenge: [3; 32],
    }
}
struct Held(Relay);
impl Deref for Held {
    type Target = Relay;
    fn deref(&self) -> &Relay {
        &self.0
    }
}
impl DerefMut for Held {
    fn deref_mut(&mut self) -> &mut Relay {
        &mut self.0
    }
}
impl Drop for Held {
    fn drop(&mut self) {
        let _ = self.0.abort();
        while matches!(self.0.reap(), Ok(None)) {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}
fn spawn(spec: process::Spec) -> Held {
    Held(Relay::spawn(spec, 0, limits(), Io::now().unwrap()).unwrap())
}
fn receive(p: &mut Relay) -> (Incoming, u64) {
    let until = Io::now().unwrap() + 2000;
    loop {
        let progress = p.poll().unwrap();
        if let Some(frame) = progress.incoming {
            return frame;
        }
        assert!(Io::now().unwrap() < until, "actual child frame deadline");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn actual_persistent_child_keeps_stdin_and_reports_local_frame_completion() {
    let mut renew = hello();
    renew.kind = frame::RENEW;
    let live = wire::Frame {
        rank: 0,
        body: wire::Body::DrainRequest(wire::DrainRequest {
            pair_digest: [4; 32],
            epoch: wire::DRAIN_EPOCH,
        }),
    };
    let q = wire::Frame {
        rank: 0,
        body: wire::Body::Quiescent(wire::Quiescent {
            pair_digest: [4; 32],
            child_instance: [5; 32],
            epoch: wire::DRAIN_EPOCH,
            last_command: wire::SHUTDOWN_COMMAND,
            receipt_nonce: [6; 32],
        }),
    };
    let mut p = spawn(script(
        r#"
import os,sys
def exact(n):
    b=b''
    while len(b)<n:
        x=os.read(0,n-len(b))
        if not x: os._exit(81)
        b+=x
    return b
os.write(1,bytes.fromhex(sys.argv[1]))
assert exact(112)==bytes.fromhex(sys.argv[2])
os.write(1,bytes.fromhex(sys.argv[3]))
assert exact(56)==bytes.fromhex(sys.argv[4])
os.write(2,b'final-note')
os._exit(74)
"#,
        vec![
            hex(&hello().encode()),
            hex(&renew.encode()),
            hex(q.encode().unwrap().as_slice()),
            hex(live.encode().unwrap().as_slice()),
        ],
    ));
    let (Incoming::Legacy(value), _) = receive(&mut p) else {
        panic!("real HELLO")
    };
    assert_eq!(value, hello());
    assert!(
        p.stdin.is_some(),
        "empty queue must not close persistent stdin"
    );
    p.queue_lease(&renew, Io::now().unwrap()).unwrap();
    let (Incoming::Live(value), _) = receive(&mut p) else {
        panic!("real Q frame")
    };
    assert_eq!(value, q);
    let began = Io::now().unwrap();
    p.queue_live(&live, began).unwrap();
    let until = began + 2000;
    let (mut completion, mut eof, mut exit, mut stderr) = (None, false, None, Vec::new());
    loop {
        let progress = p.poll().unwrap();
        if progress.sent_live.is_some() {
            assert!(completion.is_none());
            completion = progress.sent_live;
        }
        eof |= progress.eof_idle;
        if progress.exit.is_some() {
            exit = progress.exit;
        }
        stderr.extend(progress.stderr);
        if progress.done {
            break;
        }
        assert!(Io::now().unwrap() < until);
        std::thread::sleep(Duration::from_millis(1));
    }
    let sent = completion.unwrap();
    assert_eq!((sent.kind, sent.started), (0x16, began));
    assert!(sent.completed >= began);
    assert!(eof);
    assert_eq!(exit.unwrap().code(), Some(74));
    assert_eq!(stderr, b"final-note");
}

#[test]
fn actual_exited_child_buffered_frame_is_drained_before_idle_eof() {
    let mut p = spawn(script(
        "import os,sys; os.write(1,bytes.fromhex(sys.argv[1])); os._exit(74)",
        vec![hex(&hello().encode())],
    ));
    let until = Io::now().unwrap() + 2000;
    while p.reap().unwrap().is_none() {
        assert!(Io::now().unwrap() < until);
        std::thread::sleep(Duration::from_millis(1));
    }
    let (Incoming::Legacy(value), _) = receive(&mut p) else {
        panic!("buffered complete frame")
    };
    assert_eq!(value, hello());
    let mut renew = hello();
    renew.kind = frame::RENEW;
    assert!(p.queue_lease(&renew, Io::now().unwrap()).is_err());
}
