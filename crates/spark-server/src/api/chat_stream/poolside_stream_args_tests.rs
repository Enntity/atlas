// SPDX-License-Identifier: AGPL-3.0-only
//! `--tool-guard stream` on Poolside v1: string arguments reach the client as
//! they are written, and a call that fails or is cut off never finishes
//! `tool_calls`. Real detector, tool handlers and terminal wire contract.
use super::tool_security_tests::{dispatch, finish};
use super::*;

const AWKWARD: &str = "She said \"wait\" \\ then:\n\tcaf\u{e9} \u{65e5}\u{672c}\u{8a9e} \u{1f389} \
                       </arg_valu and </arg_value still";

#[derive(Clone, Copy, PartialEq, Debug)]
enum Guard {
    Off,
    On,
    Stream,
}

fn request() -> ir::ChatRequest {
    let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(json!({
        "model":"fixture", "messages":[{"role":"user","content":"speak"}],
        "tools":[{"type":"function","function":{"name":"Say",
            "parameters":{"type":"object","properties":{
                "text":{"type":"string"},"voice":{"type":"string"},"rate":{"type":"number"},
                "url":{"type":"string"},"path":{"type":"string"}},
                "required":["text"]}}}],
        "max_tokens":192
    }))
    .unwrap();
    wire.into()
}

/// A stream set up as `chat_stream` sets it up for this guard setting.
fn setup(glm: bool, guard: Guard, cwd: Option<&str>) -> (StreamCtx, StreamState) {
    let (mut c, mut s, _) = stream_for_request(app(glm), true, vec![], request());
    let policy = &mut Arc::get_mut(&mut c.state).unwrap().tool_guard;
    policy.enabled = guard != Guard::Off;
    policy.stream = guard == Guard::Stream;
    c.cwd_for_normalize = cwd.map(String::from);
    let detector = s.detector.as_mut().unwrap();
    detector.set_stream_poolside_strings(c.streams_poolside_strings());
    // `handle_token` asks a GLM detector for complete calls on every token.
    detector.set_complete_poolside_calls(c.state.glm_tool_boundary.is_some());
    (c, s)
}

fn say(pairs: &[(&str, &str)]) -> String {
    let mut text = "<tool_call>Say".to_string();
    for (key, value) in pairs {
        text += &format!("<arg_key>{key}</arg_key><arg_value>{value}</arg_value>");
    }
    text + "</tool_call>"
}

/// `text` a character at a time: the deltas of each character, in order.
fn run(s: &mut StreamState, c: &StreamCtx, text: &str) -> Vec<Vec<ir::StreamDelta>> {
    text.chars()
        .map(|ch| dispatch(s, c, &ch.to_string()))
        .collect()
}

fn args(deltas: &[ir::StreamDelta]) -> String {
    deltas
        .iter()
        .filter_map(|d| match d {
            ir::StreamDelta::ToolCallArgs { fragment, .. } => Some(fragment.as_str()),
            _ => None,
        })
        .collect()
}

fn headers(deltas: &[ir::StreamDelta]) -> Vec<&str> {
    deltas
        .iter()
        .filter_map(|d| match d {
            ir::StreamDelta::ToolCallStart { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect()
}

fn finish_reason(done: &[ir::StreamDelta]) -> ir::FinishReason {
    done.iter()
        .find_map(|d| match d {
            ir::StreamDelta::Finish { reason, .. } => Some(reason.clone()),
            _ => None,
        })
        .expect("a finish delta")
}

/// The response ended as a blocked call the client must not run.
fn assert_not_runnable(done: &[ir::StreamDelta]) {
    assert_eq!(
        finish_reason(done),
        ir::FinishReason::ContentFilter,
        "{done:?}"
    );
    assert!(
        done.iter()
            .any(|d| matches!(d, ir::StreamDelta::Content { text, .. }
            if text == atlas_tool_guard::STREAMED_REFUSAL)),
        "{done:?}"
    );
    assert!(args(done).is_empty() && headers(done).is_empty());
}

#[test]
fn string_arguments_stream_and_match_the_held_call() {
    for glm in [false, true] {
        for pairs in [
            vec![("text", "Hello there, world")],
            vec![("text", AWKWARD)],
            vec![("text", "first"), ("voice", "second")],
            vec![("text", "then a number"), ("rate", "1.5")],
            vec![("rate", "2"), ("text", "after a number")],
            vec![("voice", "text left out")],
        ] {
            let text = say(&pairs);
            let (c, mut s) = setup(glm, Guard::Stream, None);
            let steps = run(&mut s, &c, &text);
            let close = steps.len() - "</tool_call>".len();
            let early = steps[..close].concat();
            let all = steps.concat();
            assert_eq!(headers(&early), ["Say"], "glm={glm} {pairs:?}");
            assert!(
                matches!(early[0], ir::StreamDelta::ToolCallStart { .. }),
                "the header leads: {early:?}"
            );
            assert!(args(&early).len() > 10, "arguments before the close");
            assert_eq!(headers(&all), ["Say"]);
            assert_eq!(
                finish_reason(&finish(&mut s, &c)),
                ir::FinishReason::ToolCalls
            );

            // The same text held (`--tool-guard on`) delivers the same bytes.
            let (c, mut s) = setup(glm, Guard::On, None);
            let held = run(&mut s, &c, &text).concat();
            assert_eq!(args(&all), args(&held), "glm={glm} {pairs:?}");
            serde_json::from_str::<serde_json::Value>(&args(&all)).unwrap();
        }
    }
}

#[test]
fn a_string_value_arrives_in_pieces() {
    let (c, mut s) = setup(true, Guard::Stream, None);
    let text = say(&[("text", "Hello there, world")]);
    let cut = text.find(", world").unwrap();
    let steps = run(&mut s, &c, &text[..cut]);
    assert_eq!(args(&steps.concat()), r#"{"text":"Hello there"#);
    assert!(steps.iter().filter(|step| !args(step).is_empty()).count() > 5);
}

#[test]
fn without_the_stream_setting_the_call_is_held_as_before() {
    let text = say(&[("text", AWKWARD), ("rate", "2")]);
    let expected = serde_json::to_string(&json!({"text": AWKWARD, "rate": 2.0})).unwrap();
    for glm in [false, true] {
        for (guard, cwd) in [
            (Guard::Off, None),
            (Guard::On, None),
            // Path rewriting changes arguments at the end of a call.
            (Guard::Stream, Some("/work")),
        ] {
            let (c, mut s) = setup(glm, guard, cwd);
            let steps = run(&mut s, &c, &text);
            let (last, before) = steps.split_last().unwrap();
            assert!(before.concat().is_empty(), "{guard:?}: nothing early");
            assert!(
                matches!(last.as_slice(), [
                    ir::StreamDelta::ToolCallStart { name, index: 0, .. },
                    ir::StreamDelta::ToolCallArgs { fragment, index: 0, .. },
                ] if name == "Say" && *fragment == expected),
                "{guard:?} glm={glm}: {last:?}"
            );
            assert_eq!(
                finish_reason(&finish(&mut s, &c)),
                ir::FinishReason::ToolCalls
            );
        }
    }
}

#[test]
fn a_cut_off_call_does_not_finish_tool_calls() {
    let text = say(&[("text", "Hello there"), ("rate", "2")]);
    for glm in [false, true] {
        for cut in ["</tool_call>", "</arg_value></tool_call>", " there"] {
            let (c, mut s) = setup(glm, Guard::Stream, None);
            let sent = run(&mut s, &c, &text[..text.find(cut).unwrap()]).concat();
            assert_eq!(headers(&sent), ["Say"]);
            assert!(!args(&sent).is_empty());
            assert_not_runnable(&finish(&mut s, &c));
        }
    }
}

#[test]
fn a_call_that_fails_at_its_end_does_not_finish_tool_calls() {
    for text in [
        say(&[("text", "one"), ("text", "two")]),
        "<tool_call>Say<arg_key>text</arg_key><arg_value>never closed</tool_call>".to_string(),
        say(&[("text", "one")]).replace("</tool_call>", "tail</tool_call>"),
        // A complete call the guard blocks.
        say(&[("text", "fetch"), ("url", "http://169.254.169.254/latest/")]),
    ] {
        for glm in [false, true] {
            let (c, mut s) = setup(glm, Guard::Stream, None);
            let sent = run(&mut s, &c, &text).concat();
            assert_eq!(headers(&sent), ["Say"], "{text}");
            assert!(
                serde_json::from_str::<serde_json::Value>(&args(&sent)).is_err()
                    || text.contains("169.254"),
                "{sent:?}"
            );
            assert!(s.cancel_flag.load(Ordering::Acquire));
            assert_not_runnable(&finish(&mut s, &c));
        }
    }
}

#[test]
fn an_unregistered_name_starts_no_call() {
    let text = say(&[("text", "hello")]).replace("Say", "Shout");
    let (c, mut s) = setup(false, Guard::Stream, None);
    let sent = run(&mut s, &c, &text).concat();
    assert!(
        headers(&sent).is_empty() && args(&sent).is_empty(),
        "{sent:?}"
    );
    let done = finish(&mut s, &c);
    assert_ne!(
        finish_reason(&done),
        ir::FinishReason::ToolCalls,
        "{done:?}"
    );
    assert!(headers(&done).is_empty() && args(&done).is_empty());
}

/// The whole token path (`handle_token`, which asks a GLM detector for
/// complete calls) with the guard set to stream before the stream is built.
#[test]
fn glm_tokens_stream_through_handle_token() {
    for complete in [true, false] {
        let mut state = app(true);
        Arc::get_mut(&mut state).unwrap().tool_guard.stream = true;
        // plan </think> <tool_call> get_weather <arg_key> city </arg_key> <arg_value> Oslo
        let (c, mut s, mut sent) =
            super::stream_for_app(state, true, vec![1, END, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(headers(&sent), ["get_weather"], "{sent:?}");
        assert_eq!(args(&sent), r#"{"city":"Oslo"#);
        if !complete {
            assert_not_runnable(&finish(&mut s, &c));
            continue;
        }
        sent.extend(handle_token(&mut s, &c, 10));
        sent.extend(handle_token(&mut s, &c, 11));
        assert_eq!(args(&sent), r#"{"city":"Oslo"}"#);
        assert_eq!(
            finish_reason(&finish(&mut s, &c)),
            ir::FinishReason::ToolCalls
        );
    }
}
