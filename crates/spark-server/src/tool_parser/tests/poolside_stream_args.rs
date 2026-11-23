// SPDX-License-Identifier: AGPL-3.0-only

//! Poolside v1 argument streaming at the detector: what goes out before a
//! call closes, and that the fragments add up to the blocking path's arguments.

use super::super::*;

fn tools() -> Vec<ToolDefinition> {
    serde_json::from_value(serde_json::json!([
        {"type": "function", "function": {"name": "Say", "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "voice": {"type": "string"},
                "rate": {"type": "number"},
                "loud": {"type": "boolean"},
                "tags": {"type": "array"}
            },
            "required": ["text"]}}},
        {"type": "function", "function": {"name": "Bash", "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}, "description": {"type": "string"}},
            "required": ["command", "description"]}}}
    ]))
    .unwrap()
}

fn detector(streaming: bool) -> StreamingToolDetector {
    let mut detector = StreamingToolDetector::new_with_tools(tools());
    detector.set_promote_bare_names(true);
    detector.set_stream_poolside_strings(streaming);
    detector.buffer_args = false;
    detector
}

fn call(name: &str, pairs: &[(&str, &str)]) -> String {
    let mut text = format!("<tool_call>{name}");
    for (key, value) in pairs {
        text += &format!("<arg_key>{key}</arg_key><arg_value>{value}</arg_value>");
    }
    text + "</tool_call>"
}

/// The end-of-call chain the API runs on a whole Poolside call.
fn finish(call: &mut ToolCall) {
    backfill_required_params(std::slice::from_mut(call), &tools());
    coerce_all(std::slice::from_mut(call), &tools());
}

/// The arguments the API delivers for `text` without streaming them: the
/// blocking parser and the end-of-call chain (`build_choice_message`, and
/// `handle_tool_call_delta` for a held stream). `None` when it delivers no
/// call: nothing parsed, or validation fails it outright.
fn blocking_args(text: &str) -> Option<String> {
    let (_, mut calls) = parse_tool_calls_promoting_bare_names(text);
    assert!(calls.len() <= 1);
    let mut call = calls.pop()?;
    finish(&mut call);
    let hard = matches!(
        assess_tool_call(&call, &tools()),
        Err(ToolCallIssue::Hard(_))
    );
    (!hard).then_some(call.function.arguments)
}

/// What a client holds for call 0: its header, its argument text, whether it
/// ended, and any text that came out as content.
#[derive(Debug, Default, PartialEq)]
struct Seen {
    name: Option<String>,
    args: String,
    fragments: usize,
    ended: bool,
    content: String,
}

fn see(seen: &mut Seen, outputs: Vec<DetectorOutput>) {
    for output in outputs {
        match output {
            DetectorOutput::ToolCallStart { name, idx, .. } => {
                assert_eq!(idx, 0);
                assert!(seen.name.replace(name).is_none(), "one header");
            }
            DetectorOutput::ToolCallArgsFragment { fragment, idx } => {
                assert_eq!(idx, 0);
                assert!(seen.name.is_some(), "header before arguments");
                assert!(!seen.ended && !fragment.is_empty());
                seen.args += &fragment;
                seen.fragments += 1;
            }
            DetectorOutput::ToolCallEnd { idx } => {
                assert_eq!(idx, 0);
                assert!(!std::mem::replace(&mut seen.ended, true), "one end");
            }
            DetectorOutput::Content(text) => seen.content += &text,
            // A call that opens and closes within one piece of text has no
            // header out yet, so it is delivered whole, as when held.
            DetectorOutput::ToolCall(mut call, idx) => {
                assert_eq!(idx, 0);
                assert!(seen.name.is_none() && !seen.ended);
                finish(&mut call);
                seen.name = Some(call.function.name);
                seen.args = call.function.arguments;
                seen.ended = true;
            }
            DetectorOutput::ToolCallDelta { .. } => {
                panic!("a streamed call's arguments come as fragments")
            }
        }
    }
}

fn feed(chunks: &[&str]) -> Seen {
    let mut detector = detector(true);
    let mut seen = Seen::default();
    for chunk in chunks {
        see(&mut seen, detector.process(chunk));
    }
    seen
}

fn chars(text: &str) -> Vec<&str> {
    text.char_indices()
        .map(|(at, c)| &text[at..at + c.len_utf8()])
        .collect()
}

/// `text` one character at a time, and cut in two at every character.
fn every_feed(text: &str) -> Vec<Seen> {
    let mut all = vec![feed(&chars(text))];
    for (at, _) in text.char_indices() {
        all.push(feed(&[&text[..at], &text[at..]]));
    }
    all
}

const AWKWARD: &str = "She said \"wait\" \\ then:\n\tcaf\u{e9} \u{65e5}\u{672c}\u{8a9e} \u{1f389} \
                       </arg_valu and </arg_value still </tool_cal text";

#[test]
fn a_string_argument_goes_out_while_it_is_written() {
    let text = call("Say", &[("text", "Hello there, world")]);
    let cut = text.find(", world").unwrap();
    let seen = feed(&chars(&text[..cut]));
    assert_eq!(seen.name.as_deref(), Some("Say"));
    assert_eq!(seen.args, r#"{"text":"Hello there"#);
    assert!(seen.fragments > 5, "piece by piece: {seen:?}");
    assert!(!seen.ended);

    // The header alone, as soon as the name is complete.
    let seen = feed(&chars("<tool_call>Say<arg_key>"));
    assert_eq!(
        (seen.name.as_deref(), seen.args.as_str()),
        (Some("Say"), "")
    );
}

#[test]
fn fragments_add_up_to_the_blocking_arguments() {
    for pairs in [
        vec![("text", "Hello there, world")],
        vec![("text", AWKWARD)],
        vec![("text", "one"), ("voice", "two \"quoted\"")],
        vec![("text", "five"), ("rate", "1.5")],
        vec![
            ("rate", "2"),
            ("loud", "true"),
            ("text", "after typed values"),
        ],
        vec![
            ("tags", "[\"a\", {\"b\": 1}]"),
            ("text", "x"),
            ("loud", "maybe"),
        ],
        // Left to the end of the call: an undeclared key, a blank value, a
        // key the schema spells differently, a required parameter left out.
        vec![("text", "hi"), ("mood", "calm")],
        vec![("text", "  "), ("voice", "v")],
        vec![("Text", "renamed key"), ("voice", "v")],
        vec![("voice", "no text")],
        vec![("text", "true"), ("voice", "42")],
    ] {
        let text = call("Say", &pairs);
        let expected = blocking_args(&text).expect("a valid call");
        for seen in every_feed(&text) {
            assert_eq!(seen.name.as_deref(), Some("Say"));
            assert!(seen.ended && seen.content.is_empty(), "{seen:?}");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&seen.args).unwrap(),
                serde_json::from_str::<serde_json::Value>(&expected).unwrap(),
                "{pairs:?}"
            );
        }
        // Compact, and in the same key order wherever the model's order is
        // the blocking path's (it renames `Text` to the end of the object).
        if pairs[0].0 != "Text" && pairs != [("text", "  "), ("voice", "v")] {
            assert_eq!(feed(&chars(&text)).args, expected, "{pairs:?}");
        }
    }
}

#[test]
fn a_blank_required_value_is_filled_as_the_blocking_path_fills_it() {
    let text = call("Bash", &[("command", "ls -la"), ("description", "")]);
    let expected = blocking_args(&text).unwrap();
    assert!(expected.contains("Run: ls -la"), "{expected}");
    for seen in every_feed(&text) {
        assert_eq!(seen.args, expected);
    }
}

#[test]
fn text_that_may_be_the_closer_waits() {
    let mut detector = detector(true);
    let mut seen = Seen::default();
    see(
        &mut seen,
        detector.process("<tool_call>Say<arg_key>text</arg_key><arg_value>abc</arg_val"),
    );
    assert_eq!(seen.args, r#"{"text":"abc"#);
    see(&mut seen, detector.process("ley</arg_"));
    assert_eq!(seen.args, r#"{"text":"abc</arg_valley"#);
    see(&mut seen, detector.process("value>"));
    assert_eq!(seen.args, r#"{"text":"abc</arg_valley""#);
    see(&mut seen, detector.process("</tool_call>"));
    assert_eq!(seen.args, r#"{"text":"abc</arg_valley"}"#);
    assert!(seen.ended);
}

#[test]
fn a_typed_value_goes_out_whole_at_its_close() {
    let mut detector = detector(true);
    let mut seen = Seen::default();
    let open = "<tool_call>Say<arg_key>rate</arg_key><arg_value>1.2";
    see(&mut seen, detector.process(open));
    assert_eq!(seen.args, "");
    see(&mut seen, detector.process("5</arg_value><arg_key>text"));
    assert_eq!((seen.args.as_str(), seen.fragments), (r#"{"rate":1.25"#, 1));
    see(&mut seen, detector.process("</arg_key><arg_value>go"));
    assert_eq!(seen.args, r#"{"rate":1.25,"text":"go"#);
}

#[test]
fn consecutive_calls_each_stream() {
    let mut detector = detector(true);
    let text = call("Say", &[("text", "one")]) + &call("Say", &[("text", "two")]);
    let mut args = [String::new(), String::new()];
    let mut ends = vec![];
    for chunk in chars(&text) {
        for output in detector.process(chunk) {
            match output {
                DetectorOutput::ToolCallArgsFragment { fragment, idx } => args[idx] += &fragment,
                DetectorOutput::ToolCallEnd { idx } => ends.push(idx),
                DetectorOutput::ToolCallStart { .. } => {}
                _ => panic!("unexpected output"),
            }
        }
    }
    assert_eq!(args, [r#"{"text":"one"}"#, r#"{"text":"two"}"#]);
    assert_eq!(ends, [0, 1]);
}

#[test]
fn an_unregistered_name_sends_no_call() {
    let text = call("Shout", &[("text", "hello")]);
    assert!(blocking_args(&text).is_none());
    for seen in every_feed(&text) {
        assert_eq!(
            (seen.name, seen.args.as_str(), seen.ended),
            (None, "", false)
        );
        assert_eq!(seen.content, text, "returned as the prose it is");
    }
}

#[test]
fn a_call_that_does_not_parse_is_left_unclosed() {
    for text in [
        // A second `text`, a value with no closer, a stray tail, a bad key.
        call("Say", &[("text", "one"), ("text", "two")]),
        "<tool_call>Say<arg_key>text</arg_key><arg_value>never closed</tool_call>".to_string(),
        call("Say", &[("text", "one")]).replace("</tool_call>", "tail</tool_call>"),
        call("Say", &[("text", "one"), ("not a key", "x")]),
    ] {
        assert!(blocking_args(&text).is_none(), "{text}");
        for seen in every_feed(&text) {
            assert!(
                serde_json::from_str::<serde_json::Value>(&seen.args).is_err(),
                "not arguments a client can run: {seen:?}"
            );
            // Once its header is out the call still ends, unclosed; a call
            // that arrived in one piece was never a call at all.
            match seen.name.as_deref() {
                Some(name) => assert!(name == "Say" && seen.ended, "{seen:?}"),
                None => assert_eq!(seen.content, text),
            }
        }
    }
}

#[test]
fn a_call_the_stream_cuts_off_never_ends() {
    let text = call("Say", &[("text", "Hello there"), ("rate", "2")]);
    for cut in ["</tool_call>", "</arg_value></tool_call>", "there"] {
        let mut detector = detector(true);
        let mut seen = Seen::default();
        for chunk in chars(&text[..text.find(cut).unwrap()]) {
            see(&mut seen, detector.process(chunk));
        }
        assert!(seen.name.is_some() && !seen.ended);
        assert!(serde_json::from_str::<serde_json::Value>(&seen.args).is_err());
        let flushed = detector.flush();
        assert!(
            flushed
                .iter()
                .all(|output| matches!(output, DetectorOutput::Content(_))),
            "the end of the stream completes nothing"
        );
    }
}

#[test]
fn off_by_default_the_call_is_held_as_before() {
    let text = call("Say", &[("text", AWKWARD), ("rate", "2")]);
    let raw = parse_tool_calls_promoting_bare_names(&text).1[0]
        .function
        .arguments
        .clone();
    for mut detector in [
        detector(false),
        {
            let mut plain = StreamingToolDetector::new_with_tools(tools());
            plain.set_promote_bare_names(true);
            plain.buffer_args = false;
            plain
        },
        {
            // The ATLAS_BUFFER_TOOL_ARGS switch holds a call whatever is asked.
            let mut buffered = detector(true);
            buffered.buffer_args = true;
            buffered
        },
    ] {
        assert!(!detector.streams_poolside_args());
        let pieces = chars(&text);
        let (last, before) = pieces.split_last().unwrap();
        let mut early = vec![];
        for chunk in before {
            early.extend(detector.process(chunk));
        }
        assert!(
            matches!(early.as_slice(), [DetectorOutput::ToolCallStart { name, .. }] if name == "Say"),
            "only the header before the call closes"
        );
        let closing = detector.process(last);
        assert!(
            matches!(closing.as_slice(), [
                DetectorOutput::ToolCallDelta { args, idx: 0 },
                DetectorOutput::ToolCallEnd { idx: 0 },
            ] if *args == raw),
            "the raw arguments, once, at the close"
        );
    }
}
