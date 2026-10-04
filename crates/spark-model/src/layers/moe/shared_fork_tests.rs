// SPDX-License-Identifier: AGPL-3.0-only
//! ATLAS_GLM_LAYER_FORK through the actual grouped verify FFN: with `moe` the
//! TP-split shared expert's launches leave the compute stream for the side
//! stream between the router GEMV and the unpermute, unchanged; everything
//! else launches exactly as without the switch.
use super::Event;
use super::Gpu;
use super::decode_m16::{Entry, ffn_child, verify_ffn};
use spark_runtime::gpu::GpuBackend;

const STREAM: u64 = 91;
/// The recording backend's `create_stream` / `create_event` (trait defaults).
const SIDE: u64 = 0;

/// The child's trace as lines: launches by stream, then the fences.
fn render(trace: &[Event]) -> Vec<String> {
    trace
        .iter()
        .filter_map(|e| match e {
            Event::Launch(k, g, b, m, s, a) => Some(format!("L{s} {k} {g:?} {b:?} {m} {a:?}")),
            Event::Record(event, s) => Some(format!("R{s} {event}")),
            Event::Wait(s, event) => Some(format!("W{s} {event}")),
            _ => None,
        })
        .collect()
}

fn on(lines: &[String], prefix: &str) -> Vec<String> {
    lines
        .iter()
        .filter(|l| l.starts_with(prefix))
        .map(|l| l[prefix.len()..].to_owned())
        .collect()
}

#[test]
fn moe_fork_moves_only_the_split_shared_expert_to_the_side_stream() {
    const SENTINEL: &str = "ATLAS_TEST_LAYER_FORK";
    let Ok(rows) = std::env::var(SENTINEL) else {
        let name = concat!(
            module_path!(),
            "::moe_fork_moves_only_the_split_shared_expert_to_the_side_stream"
        );
        let name = name.split_once("::").unwrap().1;
        let run = |rows: &str, mode: &str| {
            let mut cmd = ffn_child(name, SENTINEL, rows);
            cmd.env("ATLAS_GLM_SHARED_TP_SPLIT", "1")
                .env("ATLAS_W4A16_TC", "1")
                .env("ATLAS_GLM_MOE_DECODE_M16", "1")
                .env("ATLAS_GLM_MOE_DOWN_ZSKIP", "1")
                .env("ATLAS_GLM_MOE_DECODE_STREAM", "1")
                .env("ATLAS_GLM_MOE_STREAM_NOSYNC", "1")
                .env("ATLAS_GLM_LAYER_FORK", mode);
            let out = cmd.output().unwrap();
            let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
            assert!(
                out.status.success(),
                "{mode}\n{stdout}\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
            stdout
                .lines()
                .filter_map(|l| l.strip_prefix("TRACE "))
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        for rows in ["3", "12"] {
            let off = run(rows, "0");
            // `index` forks no MoE work.
            assert_eq!(run(rows, "index"), off, "{rows} rows");
            let fork = run(rows, "moe");
            let main = format!("L{STREAM} ");
            let side = on(&fork, &format!("L{SIDE} "));
            assert!(side.len() >= 3, "{rows} rows: shared gate/up, SiLU, down");
            assert!(
                off.iter()
                    .all(|l| l.starts_with(&main) || l.starts_with("U "))
            );
            // Off, the shared launches run first, in the same order, and the
            // router GEMV follows them. Forked, the compute stream runs the
            // rest unchanged, the router first.
            let (serial, n) = (on(&off, &main), side.len());
            assert_eq!(serial[..n], side[..], "{rows} rows");
            assert_eq!(on(&fork, &main), serial[n..], "{rows} rows");
            // Fork right after the router GEMV, the side launches next; join
            // right before the unpermute, their first reader.
            let at = |prefix: &str| fork.iter().position(|l| l.starts_with(prefix)).unwrap();
            let router = at(&format!("{main}{}", serial[n]));
            let unpermute = at(&format!("{main}{} ", on(&fork, "U ")[0]));
            let fences = [
                format!("R{STREAM} 0"),
                format!("W{SIDE} 0"),
                format!("R{SIDE} 0"),
                format!("W{STREAM} 0"),
            ];
            let order: Vec<usize> = fences.iter().map(|f| at(f)).collect();
            assert_eq!(
                order,
                [router + 1, router + 2, unpermute - 2, unpermute - 1]
            );
            assert_eq!(
                on(&fork[router + 3..router + 3 + n], &format!("L{SIDE} ")),
                side
            );
        }
        return;
    };
    let gpu = Gpu::new();
    let entry = if rows == "3" {
        Entry::Prefill
    } else {
        Entry::Owner
    };
    let trace = verify_ffn(&gpu, 0, false, true, entry, rows.parse().unwrap());
    for line in render(&trace) {
        println!("TRACE {line}");
    }
    let unpermute = gpu
        .kernel("moe", "moe_unpermute_reduce_indexed_ep")
        .unwrap();
    println!("TRACE U {}", unpermute.0);
}
