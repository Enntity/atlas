// SPDX-License-Identifier: AGPL-3.0-only

//! Poolside v1 argument streaming (`set_stream_poolside_strings`): a string
//! argument reaches the client while the model is still writing it.
//!
//! The fragments of one call concatenate to the JSON object the held path
//! delivers for the same text. A value streams only where the end-of-call
//! chain (`backfill_required_params`, `coerce_all`) leaves it as written: a
//! declared, string-typed parameter with a non-blank value. Any other declared
//! value waits for its `</arg_value>` and goes out typed, in one piece. What
//! the chain alone decides (undeclared or renamed keys, blank values, required
//! parameters the model left out) follows at `</tool_call>`, before the closing
//! brace. A call that does not parse there gets no closing brace, so what the
//! client holds is not a JSON object and the API ends the response as rejected.

use super::*;

const KEY_OPEN: &str = "<arg_key>";
const KEY_CLOSE: &str = "</arg_key>";
const VALUE_OPEN: &str = "<arg_value>";
const VALUE_CLOSE: &str = "</arg_value>";

/// The argument value the scan is inside, for the call being streamed.
#[derive(Debug, Default)]
pub(super) enum PoolsideValue {
    /// Between pairs: the next thing is `<arg_key>` or the end of the call.
    #[default]
    None,
    /// A string argument, sent as it arrives. `sent` is false until its first
    /// non-blank text: a blank value is the end-of-call chain's to fill.
    Text { key: String, sent: bool },
    /// Any other argument: sent typed, in one piece, at `</arg_value>`.
    Whole { key: String },
    /// The body is not `<arg_key>K</arg_key><arg_value>V</arg_value>` pairs
    /// with distinct keys. Nothing more is sent; the call fails at its end.
    Malformed,
}

/// `text` as it reads inside a JSON string.
fn json_text(text: &str) -> String {
    let quoted = serde_json::to_string(text).unwrap_or_default();
    quoted
        .get(1..quoted.len().saturating_sub(1))
        .unwrap_or_default()
        .to_string()
}

/// Bytes at the end of `text` that could be the start of `</arg_value>`.
fn partial_value_close(text: &str) -> usize {
    (1..VALUE_CLOSE.len())
        .rev()
        .find(|&n| text.ends_with(&VALUE_CLOSE[..n]))
        .unwrap_or(0)
}

/// What `full` (a call's finished arguments) adds to `sent` (the fragments
/// already out, which open an object and never close it): the members still
/// missing, in `full`'s order, and the closing brace. `None` when a member
/// already sent is not in `full` with the same value.
fn unsent_args(sent: &str, full: &str) -> Option<String> {
    if sent.is_empty() {
        return Some(full.to_string());
    }
    type Object = serde_json::Map<String, serde_json::Value>;
    let sent: Object = serde_json::from_str(&format!("{sent}}}")).ok()?;
    let full: Object = serde_json::from_str(full).ok()?;
    if sent.iter().any(|(key, value)| full.get(key) != Some(value)) {
        return None;
    }
    let mut rest = String::new();
    for (key, value) in full.iter().filter(|(key, _)| !sent.contains_key(*key)) {
        rest.push(',');
        rest.push_str(&serde_json::to_string(key).ok()?);
        rest.push(':');
        rest.push_str(&serde_json::to_string(value).ok()?);
    }
    rest.push('}');
    Some(rest)
}

impl StreamingToolDetector {
    /// Send Poolside v1 string arguments as they are written, and the call's
    /// header as soon as its name is complete, instead of holding the call
    /// until `</tool_call>`. Request policy: it survives `reset()`.
    pub fn set_stream_poolside_strings(&mut self, on: bool) {
        self.stream_poolside_strings = on;
    }

    /// This detector streams Poolside v1 arguments (see the module doc).
    pub fn streams_poolside_args(&self) -> bool {
        self.promote_bare_names && self.stream_poolside_strings && !self.buffer_args
    }

    /// The end-of-call chain the API runs on a held Poolside call.
    fn finish_poolside_args(&self, call: &mut ToolCall) {
        backfill_required_params(std::slice::from_mut(call), &self.tools);
        coerce_all(std::slice::from_mut(call), &self.tools);
    }

    /// The declared type of parameter `key` of the call being streamed, when
    /// the end-of-call chain keeps that key under its own name.
    fn poolside_param_type(&self, key: &str) -> Option<&str> {
        let name = self.current_tc_name.as_deref()?;
        let tool = self.tools.iter().find(|t| t.function.name == name)?;
        let props = tool.function.parameters.as_ref()?.get("properties")?;
        super::validation::resolve_schema_type(props.get(key)?)
    }

    /// `"key":value` as the end-of-call chain types a lone non-blank `raw`.
    fn poolside_member(&self, key: &str, raw: &str) -> Option<String> {
        self.poolside_param_type(key)?;
        if raw.trim().is_empty() {
            return None;
        }
        let mut call = ToolCall {
            id: String::new(),
            call_type: "function".into(),
            function: FunctionCall {
                name: self.current_tc_name.clone()?,
                arguments: serde_json::to_string(&serde_json::json!({ key: raw })).ok()?,
            },
        };
        self.finish_poolside_args(&mut call);
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments).ok()?;
        let value = serde_json::to_string(args.get(key)?).ok()?;
        Some(format!("{}:{value}", serde_json::to_string(key).ok()?))
    }

    fn push_poolside(&mut self, outputs: &mut Vec<DetectorOutput>, fragment: String) {
        if fragment.is_empty() {
            return;
        }
        self.poolside_sent.push_str(&fragment);
        outputs.push(DetectorOutput::ToolCallArgsFragment {
            fragment,
            idx: self.call_counter as usize,
        });
    }

    /// `{` before a call's first member, `,` before the others.
    fn poolside_separator(&self) -> &'static str {
        if self.poolside_sent.is_empty() {
            "{"
        } else {
            ","
        }
    }

    /// Fragments for what `self.buffer[..limit]` adds to the call being
    /// streamed. With `final_close`, `limit` is the call's `</tool_call>`: the
    /// rest of the finished arguments follows, when the call parses.
    pub(super) fn poolside_fragments(
        &mut self,
        limit: usize,
        final_close: bool,
    ) -> Vec<DetectorOutput> {
        let mut outputs = Vec::new();
        if self.current_tc_emitted == 0 {
            // The header went out at the first `<arg_key>`; the pairs start there.
            self.current_tc_emitted = self.buffer[..limit].find(KEY_OPEN).unwrap_or(limit);
        }
        while self.poolside_step(limit, &mut outputs) {}
        if final_close {
            let mut call = parse_poolside_v1_call(self.buffer[..limit].trim());
            if let Some(call) = call.as_mut() {
                self.finish_poolside_args(call);
            }
            match call.and_then(|c| unsent_args(&self.poolside_sent, &c.function.arguments)) {
                Some(rest) => self.push_poolside(&mut outputs, rest),
                None => tracing::warn!(
                    tool = self.current_tc_name.as_deref().unwrap_or_default(),
                    "a streamed Poolside call did not parse at its end; left unclosed"
                ),
            }
        }
        outputs
    }

    /// One step of the scan over `self.buffer[self.current_tc_emitted..limit]`.
    /// False when it needs more text or the body is malformed.
    fn poolside_step(&mut self, limit: usize, outputs: &mut Vec<DetectorOutput>) -> bool {
        // A value's text may have gone out past a `</tool_call>` that only
        // then completed (the call fails: its value never closed).
        let rest = &self.buffer[self.current_tc_emitted.min(limit)..limit];
        match std::mem::take(&mut self.poolside_value) {
            PoolsideValue::Malformed => {
                self.poolside_value = PoolsideValue::Malformed;
                false
            }
            PoolsideValue::None => {
                // `<arg_key>K</arg_key><arg_value>`, as `parse_poolside_v1_call`
                // reads it. An incomplete pair header waits.
                let Some(key_end) = rest.find(KEY_CLOSE) else {
                    return false;
                };
                let after_key = &rest[key_end + KEY_CLOSE.len()..];
                if after_key.len() < VALUE_OPEN.len() && VALUE_OPEN.starts_with(after_key) {
                    return false;
                }
                let key = rest
                    .strip_prefix(KEY_OPEN)
                    .map(|r| r[..key_end - KEY_OPEN.len()].trim().to_string())
                    .filter(|key| is_tool_name_component(key) && !self.emitted_keys.contains(key))
                    .filter(|_| after_key.starts_with(VALUE_OPEN));
                let Some(key) = key else {
                    self.poolside_value = PoolsideValue::Malformed;
                    return false;
                };
                self.current_tc_emitted += key_end + KEY_CLOSE.len() + VALUE_OPEN.len();
                self.emitted_keys.push(key.clone());
                self.poolside_value = if self.poolside_param_type(&key) == Some("string") {
                    PoolsideValue::Text { key, sent: false }
                } else {
                    PoolsideValue::Whole { key }
                };
                true
            }
            PoolsideValue::Whole { key } => {
                let Some(end) = rest.find(VALUE_CLOSE) else {
                    self.poolside_value = PoolsideValue::Whole { key };
                    return false;
                };
                let member = self.poolside_member(&key, &rest[..end]);
                self.current_tc_emitted += end + VALUE_CLOSE.len();
                if let Some(member) = member {
                    let fragment = format!("{}{member}", self.poolside_separator());
                    self.push_poolside(outputs, fragment);
                }
                true
            }
            PoolsideValue::Text { key, sent } => {
                let end = rest.find(VALUE_CLOSE);
                // Without the closer, text that may be its start stays back.
                let text = &rest[..end.unwrap_or(rest.len() - partial_value_close(rest))];
                let mut fragment = String::new();
                let sending = sent || !text.trim().is_empty();
                if sending {
                    if !sent {
                        let quoted_key = serde_json::to_string(&key).unwrap_or_default();
                        fragment = format!("{}{quoted_key}:\"", self.poolside_separator());
                    }
                    fragment.push_str(&json_text(text));
                    self.current_tc_emitted += text.len();
                }
                if end.is_some() {
                    if sending {
                        fragment.push('"');
                    } else {
                        self.current_tc_emitted += text.len();
                    }
                    self.current_tc_emitted += VALUE_CLOSE.len();
                } else {
                    self.poolside_value = PoolsideValue::Text { key, sent: sending };
                }
                self.push_poolside(outputs, fragment);
                end.is_some()
            }
        }
    }
}

#[cfg(test)]
#[path = "tests/poolside_stream_args.rs"]
mod tests;
