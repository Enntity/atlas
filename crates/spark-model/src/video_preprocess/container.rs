// SPDX-License-Identifier: AGPL-3.0-only

/// Best-effort container name for error messages. Cosmetic only — nothing
/// branches on it — so an unrecognized blob is described as such rather than
/// guessed at.
pub(super) fn sniff_container(bytes: &[u8], mime: &str) -> String {
    let by_magic = if bytes.len() > 12 && &bytes[4..8] == b"ftyp" {
        Some("an MP4/MOV container")
    } else if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        Some("a Matroska/WebM container")
    } else if bytes.starts_with(b"RIFF") {
        Some("an AVI container")
    } else {
        None
    };
    match (by_magic, mime.is_empty()) {
        (Some(k), _) => k.to_string(),
        (None, false) => format!("a {mime} payload"),
        (None, true) => "an unrecognized container".to_string(),
    }
}
