// SPDX-License-Identifier: AGPL-3.0-only
//! `--tool-guard stream`: a request that asks (`stream_tool_arguments`) gets
//! Poolside v1 string arguments as they are written, and a call that fails or
//! is cut off never finishes `tool_calls`; a request that does not ask is held
//! exactly as under `--tool-guard on`. Real detector, tool handlers and
//! terminal wire contract.
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

/// A stream set up as `chat_stream` sets it up for this guard setting, for a
/// request that does or does not ask for streamed tool calls (`opt_in`).
fn setup(glm: bool, guard: Guard, opt_in: bool, cwd: Option<&str>) -> (StreamCtx, StreamState) {
    setup_app(app(glm), guard, opt_in, cwd)
}

fn setup_app(
    app: Arc<AppState>,
    guard: Guard,
    opt_in: bool,
    cwd: Option<&str>,
) -> (StreamCtx, StreamState) {
    let mut req = request();
    req.stream_tool_arguments = opt_in;
    let (mut c, mut s, _) = stream_for_request(app, true, vec![], req);
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
            let (c, mut s) = setup(glm, Guard::Stream, true, None);
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
            let (c, mut s) = setup(glm, Guard::On, false, None);
            let held = run(&mut s, &c, &text).concat();
            assert_eq!(args(&all), args(&held), "glm={glm} {pairs:?}");
            serde_json::from_str::<serde_json::Value>(&args(&all)).unwrap();
        }
    }
}

#[test]
fn a_string_value_arrives_in_pieces() {
    let (c, mut s) = setup(true, Guard::Stream, true, None);
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
            let (c, mut s) = setup(glm, guard, true, cwd);
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
            let (c, mut s) = setup(glm, Guard::Stream, true, None);
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
            let (c, mut s) = setup(glm, Guard::Stream, true, None);
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
    let (c, mut s) = setup(false, Guard::Stream, true, None);
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
        let mut req = super::request();
        req.stream_tool_arguments = true;
        // plan </think> <tool_call> get_weather <arg_key> city </arg_key> <arg_value> Oslo
        let (c, mut s, mut sent) =
            stream_for_request(state, true, vec![1, END, 3, 4, 5, 6, 7, 8, 9], req);
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

/// Each delta as the client sees it (call ids are random, so left out).
fn wire(deltas: &[ir::StreamDelta]) -> Vec<String> {
    deltas
        .iter()
        .map(|d| match d {
            ir::StreamDelta::ToolCallStart { index, name, .. } => format!("start {index} {name}"),
            d => format!("{d:?}"),
        })
        .collect()
}

/// A whole response to `text`, fed a character at a time: what each character
/// sent, then what the end sent.
fn respond(app: App, guard: Guard, opt_in: bool, text: &str) -> Vec<Vec<String>> {
    let (c, mut s) = setup_app(app(), guard, opt_in, None);
    let mut out: Vec<_> = run(&mut s, &c, text).iter().map(|d| wire(d)).collect();
    out.push(wire(&finish(&mut s, &c)));
    out
}

/// A fresh app for each response: `setup_app` sets its guard.
type App = fn() -> Arc<AppState>;

/// Every format, with a call that passes, one the guard blocks, one that does
/// not parse and one the response cuts off.
fn every_case() -> Vec<(App, String)> {
    let qwen = |v: &str| {
        format!("<tool_call><function=Say><parameter=text>{v}</parameter></function></tool_call>")
    };
    let mut cases = vec![];
    let apps: [App; 2] = [|| app(false), || app(true)];
    for app in apps {
        for text in [
            say(&[("text", AWKWARD), ("rate", "2")]),
            say(&[("text", "fetch"), ("url", "http://169.254.169.254/latest/")]),
            say(&[("text", "one"), ("text", "two")]),
            say(&[("text", "Hello there")]).replace("</arg_value></tool_call>", ""),
        ] {
            cases.push((app, text));
        }
    }
    for text in [
        qwen("Hello there"),
        qwen("fetch</parameter><parameter=url>http://169.254.169.254/latest/"),
        qwen("Hello there").replace("</parameter></function></tool_call>", ""),
    ] {
        cases.push((
            || app_for_model("qwen4_exp", tool_parser::ToolCallFormat::Qwen3Coder),
            text,
        ));
    }
    cases
}

#[test]
fn stream_mode_without_the_field_is_held_exactly_as_on() {
    for (app, text) in every_case() {
        let held = respond(app, Guard::On, false, &text);
        assert_eq!(respond(app, Guard::Stream, false, &text), held, "{text}");
        // and the field is what changes it
        assert_ne!(respond(app, Guard::Stream, true, &text), held, "{text}");
    }
}

#[test]
fn the_field_is_ignored_under_on_and_off() {
    for (app, text) in every_case() {
        for guard in [Guard::On, Guard::Off] {
            assert_eq!(
                respond(app, guard, true, &text),
                respond(app, guard, false, &text),
                "{guard:?} {text}"
            );
        }
    }
}
