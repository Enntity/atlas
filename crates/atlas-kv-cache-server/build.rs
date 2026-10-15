// SPDX-License-Identifier: AGPL-3.0-only
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("src/proto/kv_cache.proto")?;
    Ok(())
}
