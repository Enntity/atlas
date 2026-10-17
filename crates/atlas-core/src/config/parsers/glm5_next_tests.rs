// SPDX-License-Identifier: AGPL-3.0-only

use super::{parse_hc_post_multiplier, parse_text_swiglu_limit};
use serde_json::json;

#[test]
fn declared_mhc_post_multiplier_must_be_finite_and_positive() {
    assert_eq!(parse_hc_post_multiplier(&json!({})).unwrap(), None);
    for value in [2.0, 2.5] {
        assert_eq!(
            parse_hc_post_multiplier(&json!({"mhc_post_mult_value": value})).unwrap(),
            Some(value as f32)
        );
    }
    for value in [json!(null), json!("2"), json!(0), json!(-2), json!(1e100)] {
        assert!(parse_hc_post_multiplier(&json!({"mhc_post_mult_value": value})).is_err());
    }
}

#[test]
fn text_clamp_is_explicit_and_independent_of_vision() {
    assert_eq!(parse_text_swiglu_limit(&json!({})).unwrap(), None);
    assert_eq!(
        parse_text_swiglu_limit(&json!({"swiglu_limit": 10.0})).unwrap(),
        Some(10.0)
    );
    assert_eq!(
        parse_text_swiglu_limit(&json!({"vision_config":{"swiglu_limit": 10.0}})).unwrap(),
        None
    );
}

#[test]
fn invalid_declared_text_clamps_fail_instead_of_disabling() {
    for value in [
        json!(null),
        json!(false),
        json!("10"),
        json!(0.0),
        json!(-1.0),
        json!(1e100),
        json!(1e-100),
    ] {
        assert!(
            parse_text_swiglu_limit(&json!({"swiglu_limit":value})).is_err(),
            "{value}"
        );
    }
}
