// SPDX-License-Identifier: AGPL-3.0-only
//! Fixed remote node verbs. No launch-supplied shell command is interpreted.
use crate::{docker, error, process};
use std::{io, path::Path};

#[derive(Clone, Copy)]
pub enum Verb {
    Prepare,
    Create,
    Seal,
    Start,
    Observe,
    Socket,
    Relay,
    Kill,
}
impl Verb {
    fn name(self) -> &'static str {
        match self {
            Self::Prepare => "node-prepare",
            Self::Create => "node-create",
            Self::Seal => "node-seal",
            Self::Start => "node-start",
            Self::Observe => "node-observe",
            Self::Socket => "node-socket",
            Self::Relay => "node-relay",
            Self::Kill => "node-kill",
        }
    }
}
pub struct Endpoint<'a> {
    pub executable: &'a Path,
    pub key: &'a Path,
    pub known_hosts: &'a Path,
    pub environment: &'a [(String, String)],
    pub destination: &'a str,
    pub supervisor: &'a Path,
    pub self_sha256: &'a str,
}

pub fn spec(
    endpoint: Endpoint<'_>,
    verb: Verb,
    session: crate::wire::Digest,
    rank: u8,
    id: Option<crate::wire::Digest>,
    input: Vec<u8>,
    connect_ms: u64,
) -> io::Result<process::Spec> {
    let e = endpoint;
    docker::parse_id(e.self_sha256)?;
    if session == [0; 32]
        || rank > 1
        || connect_ms == 0
        || connect_ms > 86_400_000
        || id.is_some_and(|v| v == [0; 32])
        || matches!(
            verb,
            Verb::Seal | Verb::Start | Verb::Observe | Verb::Socket | Verb::Relay | Verb::Kill
        ) != id.is_some()
        || e.destination.starts_with('-')
        || e.destination.bytes().filter(|b| *b == b'@').count() != 1
        || !e
            .destination
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._@:-[]".contains(&b))
    {
        return Err(error("invalid fixed SSH target, identity or command bound"));
    }
    for p in [e.executable, e.key, e.known_hosts, e.supervisor] {
        if !p.is_absolute() || p.to_str().is_none_or(|s| s.contains('\0')) {
            return Err(error("explicit absolute UTF8 SSH paths required"));
        }
    }
    let mut words = vec![
        e.supervisor.to_str().unwrap().to_owned(),
        verb.name().into(),
        docker::hex(&session),
        rank.to_string(),
    ];
    if let Some(id) = id {
        words.push(docker::hex(&id));
    }
    words.extend(["--self-sha256".into(), e.self_sha256.into()]);
    let command = format!(
        "exec sudo -n -- {}",
        words
            .iter()
            .map(|w| format!("'{}'", w.replace('\'', "'\\''")))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let mut args: Vec<std::ffi::OsString> = [
        "-F",
        "/dev/null",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
    ]
    .into_iter()
    .map(Into::into)
    .collect();
    args.push(format!("UserKnownHostsFile={}", e.known_hosts.display()).into());
    args.extend([
        "-o".into(),
        format!("ConnectTimeout={}", connect_ms.div_ceil(1000)).into(),
        "-i".into(),
        e.key.as_os_str().to_owned(),
        "--".into(),
        e.destination.into(),
        command.into(),
    ]);
    Ok(process::Spec {
        program: e.executable.to_owned(),
        args,
        env: e
            .environment
            .iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect(),
        stdin: input,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn endpoint() -> Endpoint<'static> {
        Endpoint {
            executable: Path::new("/usr/bin/ssh"),
            key: Path::new("/key"),
            known_hosts: Path::new("/known_hosts"),
            environment: &[],
            destination: "mangokid@192.168.8.181",
            supervisor: Path::new("/opt/atlas/supervisor"),
            self_sha256: "0101010101010101010101010101010101010101010101010101010101010101",
        }
    }
    #[test]
    fn actual_command_builder_pins_ssh_config_and_only_fixed_remote_verbs() {
        let s = spec(
            endpoint(),
            Verb::Observe,
            [2; 32],
            0,
            Some([3; 32]),
            vec![4],
            1200,
        )
        .unwrap();
        assert_eq!(s.program, Path::new("/usr/bin/ssh"));
        assert_eq!(s.stdin, vec![4]);
        assert!(s.env.is_empty());
        let args: Vec<_> = s.args.iter().map(|s| s.to_str().unwrap()).collect();
        assert!(args.windows(2).any(|w| w == ["-F", "/dev/null"]));
        for option in [
            "BatchMode=yes",
            "StrictHostKeyChecking=yes",
            "IdentitiesOnly=yes",
            "ControlMaster=no",
            "ConnectTimeout=2",
        ] {
            assert!(args.contains(&option), "{option}");
        }
        let remote = args.last().unwrap();
        assert!(remote.starts_with("exec sudo -n -- '/opt/atlas/supervisor' 'node-observe' "));
        assert!(remote.contains(&format!("'{}'", "03".repeat(32))));
        assert!(remote.contains("'--self-sha256'"));
    }
    #[test]
    fn command_builder_rejects_target_or_identity_substitution_and_quotes_path() {
        let mut e = endpoint();
        e.destination = "-oProxyCommand=bad";
        assert!(spec(e, Verb::Create, [2; 32], 0, None, vec![], 1000).is_err());
        assert!(spec(endpoint(), Verb::Start, [2; 32], 0, None, vec![], 1000).is_err());
        assert!(spec(
            endpoint(),
            Verb::Prepare,
            [2; 32],
            0,
            Some([3; 32]),
            vec![],
            1000
        )
        .is_err());
        assert!(spec(endpoint(), Verb::Create, [0; 32], 0, None, vec![], 1000).is_err());
        assert!(spec(endpoint(), Verb::Create, [2; 32], 2, None, vec![], 1000).is_err());
        let mut e = endpoint();
        e.supervisor = Path::new("/opt/atlas/one'two");
        let s = spec(e, Verb::Create, [2; 32], 0, None, vec![], 1000).unwrap();
        assert!(s
            .args
            .last()
            .unwrap()
            .to_str()
            .unwrap()
            .contains("'/opt/atlas/one'\\''two'"));
    }
}
