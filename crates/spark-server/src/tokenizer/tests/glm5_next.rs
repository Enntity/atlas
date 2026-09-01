// SPDX-License-Identifier: AGPL-3.0-only

#[test]
fn loop_break_matches_official_glm_template_requirement() {
    let env = super::super::jinja_helpers::build_jinja_env(
        "{% for value in values %}{{ value }}{% if value == 2 %}{% break %}{% endif %}{% endfor %}",
    )
    .expect("loop-controls template compiles");
    let rendered = env
        .get_template("chat")
        .unwrap()
        .render(minijinja::context! { values => vec![1, 2, 3] })
        .expect("break renders");
    assert_eq!(rendered, "12");
}
