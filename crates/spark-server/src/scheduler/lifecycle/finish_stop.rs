// SPDX-License-Identifier: AGPL-3.0-only

//! Which EOS-set token ended a finished sequence (split for the 500-LoC cap).

/// The EOS-set token (model EOS or single-token client stop) that ended the
/// sequence, if that is what ended it. Mirrors `derive_finish_reason`'s
/// token rule: a "stop" whose last token is in `eos_tokens`. Any other
/// "stop" (client stop-STRING watch, cancel, drain) ends on an ordinary
/// text token, which the API layer must keep.
pub(in crate::scheduler) fn finish_stop_token(
    reason: &str,
    last_tok: Option<u32>,
    eos_tokens: &[u32],
) -> Option<u32> {
    last_tok.filter(|t| reason == "stop" && eos_tokens.contains(t))
}
