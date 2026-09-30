// SPDX-License-Identifier: AGPL-3.0-only

//! Source contract between [`PDL_KERNELS`] and every kernel a PDL target serves.

use super::{PDL_KERNELS, PDL_TARGETS};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// `src` with its `//` and `/* */` comments removed.
fn without_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(at) = rest.find("//").into_iter().chain(rest.find("/*")).min() {
        out.push_str(&rest[..at]);
        let end = if rest[at..].starts_with("//") {
            rest[at..].find('\n')
        } else {
            rest[at..].find("*/").map(|end| end + 2)
        };
        rest = end.map_or("", |end| &rest[at + end..]);
    }
    out + rest
}

fn is_name(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Whether a function body (the text after its `{`) waits before anything
/// else: `atlas_pdl_enter();`, or `atlas_pdl_enter_touch(..);`, the same
/// entry with discarded weight loads ahead of its wait.
fn waits_first(body: &str) -> bool {
    let body = body.trim_start();
    body.starts_with("atlas_pdl_enter();") || body.starts_with("atlas_pdl_enter_touch(")
}

/// Whether `src` defines a function `name` whose body waits first.
fn defines_waiting(src: &str, name: &str) -> bool {
    !name.is_empty()
        && src.match_indices(name).any(|(at, _)| {
            let rest = src[at + name.len()..].trim_start();
            !src[..at].ends_with(is_name)
                && rest.starts_with('(')
                && rest
                    .split_once('{')
                    .is_some_and(|(signature, body)| !signature.contains(';') && waits_first(body))
        })
}

/// `(name, waits)` for every `__global__` kernel defined in comment-free
/// `src`. It waits when its first statement is `atlas_pdl_enter();`, or a call
/// of a function in `src` whose first statement that is (the grouped MXFP8
/// kernels enter through `mxfp8_gemv_tc_grouped`).
fn kernels(src: &str) -> Vec<(&str, bool)> {
    src.match_indices("__global__")
        .filter_map(|(at, _)| {
            let (signature, body) = src[at..].split_once('{')?;
            if signature.contains(';') {
                return None;
            }
            // The parameter list is the last `(` of the signature; an earlier
            // one belongs to `__launch_bounds__(…)`.
            let head = signature[..signature.rfind('(')?].trim_end();
            let name = &head[head.rfind(|c| !is_name(c)).map_or(0, |i| i + 1)..];
            let body = body.trim_start();
            let callee = &body[..body.find(|c| !is_name(c)).unwrap_or(body.len())];
            Some((name, waits_first(body) || defines_waiting(src, callee)))
        })
        .collect()
}

/// The kernel sources a quant dir serves: `common/` overlaid by file name
/// (atlas-kernels `collect_cu_files`).
fn served_sources(common: &Path, quant: &Path) -> Vec<PathBuf> {
    let mut files = BTreeMap::new();
    for dir in [common, quant] {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            // `.cuh` too: a kernel may be defined in a header its `.cu` includes
            // (the GLM M16 MoE decode kernels are).
            if path.extension().is_some_and(|e| e == "cu" || e == "cuh") {
                files.insert(path.file_name().unwrap().to_owned(), path);
            }
        }
    }
    files.into_values().collect()
}

#[test]
fn parser_sees_direct_and_wrapped_waits() {
    let src = without_comments(
        r#"
// extern "C" __global__ void in_a_comment(int x) { atlas_pdl_enter(); }
/* __global__ void in_a_block(int x) {
   atlas_pdl_enter(); } */
extern "C" __global__ void declared(const float* a);
extern "C" __global__ void __launch_bounds__(W * 32) direct(const float* a) {
    // Entry order comment.
    atlas_pdl_enter();
    body<1>(a);
}
template <int NT>
__device__ __forceinline__ void wrapper(const float* a) {
    atlas_pdl_enter();
    body<NT>(a);
}
__device__ void not_a_wrapper(const float* a) { body<1>(a); atlas_pdl_enter(); }
extern "C" __global__ void wrapped(const float* a) { wrapper<2>(a); }
extern "C" __global__ void plain(const float* a) { not_a_wrapper(a); }
extern "C" __global__ void late(const float* a) {
    float x = a[0];
    atlas_pdl_enter();
}
extern "C" __global__ void other_name(const float* a) { rapper(a); }
extern "C" __global__ void touching(const float* a, const unsigned char* w) {
    atlas_pdl_enter_touch({w, 8u, 8u}, {w, 1u, 1u}, 4u, blockIdx.x, 2u);
    body<1>(a);
}
"#,
    );
    assert_eq!(
        kernels(&src),
        [
            ("direct", true),
            ("wrapped", true),
            ("plain", false),
            ("late", false),
            ("other_name", false),
            ("touching", true),
        ]
    );
}

/// A kernel is launched with PDL exactly when it is on [`PDL_KERNELS`], by
/// name, whichever file of the target defines it. One that is listed but does
/// not wait reads its predecessor's output early; one that waits but is not
/// listed pays the launch gap PDL exists to remove (the five-row KDA triple
/// did, at every width-5 verify step).
#[test]
fn pdl_target_kernels_wait_exactly_when_listed() {
    let gb10 = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/gb10");
    for target in PDL_TARGETS {
        for quant in std::fs::read_dir(gb10.join(target)).unwrap() {
            let quant = quant.unwrap().path();
            if !quant.is_dir() {
                continue;
            }
            let mut defined = Vec::new();
            for path in served_sources(&gb10.join("common"), &quant) {
                let src = without_comments(&std::fs::read_to_string(&path).unwrap());
                for (name, waits) in kernels(&src) {
                    assert_eq!(
                        waits,
                        PDL_KERNELS.contains(&name),
                        "{}: {name} waits on PDL entry = {waits}, but listed = {}",
                        path.display(),
                        !waits
                    );
                    defined.push(name.to_owned());
                }
            }
            for listed in PDL_KERNELS {
                assert!(
                    defined.iter().any(|name| name == listed),
                    "{listed} is listed, but {} defines no such kernel",
                    quant.display()
                );
            }
        }
    }
}
