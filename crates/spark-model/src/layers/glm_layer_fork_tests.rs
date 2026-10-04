// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::layers::ops::shard_test_gpu::ShardGpu;

#[test]
fn the_switch_is_explicit() {
    let off = Mode::default();
    assert_eq!(parse(None).unwrap(), off);
    assert_eq!(parse(Some("0")).unwrap(), off);
    let both = Mode {
        moe: true,
        index: true,
    };
    assert_eq!(parse(Some("1")).unwrap(), both);
    assert_eq!(
        parse(Some("moe")).unwrap(),
        Mode {
            index: false,
            ..both
        }
    );
    assert_eq!(parse(Some("index")).unwrap(), Mode { moe: false, ..both });
    for value in ["", "true", "2", " 1", "MOE", "moe,index"] {
        assert!(parse(Some(value)).is_err(), "{value:?}");
    }
}

#[test]
fn a_fork_orders_the_side_after_main_and_a_join_main_after_the_side() {
    let gpu = ShardGpu::default();
    let lane = ForkLane {
        side: 7,
        fork: 21,
        join: 22,
    };
    assert_eq!(lane.fork(&gpu, 3).unwrap(), 7);
    lane.join(&gpu, 3).unwrap();
    // A second pair on the same lane re-records both events.
    lane.fork(&gpu, 3).unwrap();
    lane.join(&gpu, 3).unwrap();
    let pair = [
        "record e21 s3",
        "wait s7 e21",
        "record e22 s7",
        "wait s3 e22",
    ];
    assert_eq!(gpu.order(), [pair, pair].concat());
}
