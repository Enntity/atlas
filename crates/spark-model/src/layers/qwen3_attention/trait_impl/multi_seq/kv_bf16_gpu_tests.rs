// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::layers::ops::rows_gpu_tests::{Rng, backend, upload};
use crate::weight_map::test_checkpoint::read_checkpoint_tensor;

#[test]
#[ignore = "requires GB10 and ATLAS_FP4L_CKPT"]
fn qwen_kv_bf16_batchm_matches_scalar_checkpoint_rows() -> Result<()> {
    let dir = std::path::PathBuf::from(std::env::var("ATLAS_FP4L_CKPT")?);
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let scalar = gpu.kernel("gemv", "dense_gemv_bf16")?;
    let batch = gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?;
    let (kd, ks, kb) = read_checkpoint_tensor(&dir, "mtp.layers.0.self_attn.k_proj.weight")?;
    let (vd, vs, vb) = read_checkpoint_tensor(&dir, "mtp.layers.0.self_attn.v_proj.weight")?;
    assert_eq!((kd.as_str(), vd.as_str()), ("BF16", "BF16"));
    assert_eq!(ks, vs);
    assert_eq!(ks.len(), 2);
    assert_eq!(ks[0] % 2, 0);
    let hidden = ks[1];
    let mut rng = Rng(0x1357_2468_abcd_4321);
    // Both TP2 shards and TP1. Nonzero Q and padding sentinels must survive.
    for (parts, rank) in [(2, 0), (2, 1), (1, 0)] {
        let kv_dim = ks[0] / parts;
        let bytes = kv_dim * hidden * 2;
        let k = DenseWeight {
            weight: upload(gpu, &kb[rank * bytes..(rank + 1) * bytes])?,
        };
        let v = DenseWeight {
            weight: upload(gpu, &vb[rank * bytes..(rank + 1) * bytes])?,
        };
        let q_dim = 12288 / parts;
        let stride = q_dim + 2 * kv_dim + 13;
        for rows in [1usize, 2, 7, 8, 9, 16, 64, 257, 16385] {
            let acts: Vec<u8> = (0..rows * hidden)
                .flat_map(|_| {
                    let x = 2.0 * ((rng.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0);
                    ((x.to_bits() >> 16) as u16).to_le_bytes()
                })
                .collect();
            let input = upload(gpu, &acts)?;
            let poison = vec![0x5a; rows * stride * 2];
            let reference = upload(gpu, &poison)?;
            let actual = upload(gpu, &poison)?;
            for row in 0..rows {
                for (i, weight) in [&k, &v].into_iter().enumerate() {
                    ops::dense_gemv(
                        gpu,
                        scalar,
                        input.offset(row * hidden * 2),
                        weight,
                        reference.offset((row * stride + q_dim + i * kv_dim) * 2),
                        kv_dim as u32,
                        hidden as u32,
                        stream,
                    )?;
                }
            }
            project_rows(
                gpu,
                batch,
                input,
                [&k, &v],
                actual,
                rows,
                hidden,
                kv_dim,
                q_dim,
                stride,
                stream,
            )?;
            gpu.synchronize(stream)?;
            let mut expected = poison.clone();
            let mut observed = poison;
            gpu.copy_d2h(reference, &mut expected)?;
            gpu.copy_d2h(actual, &mut observed)?;
            assert!(
                expected == observed,
                "TP{parts} rank{rank}, rows={rows}: K/V or untouched Q/padding differ"
            );
            eprintln!("TP{parts} rank{rank}, rows={rows}: exact K/V and untouched Q/padding");
            gpu.free(input)?;
            gpu.free(reference)?;
            gpu.free(actual)?;
        }
        gpu.free(k.weight)?;
        gpu.free(v.weight)?;
    }
    Ok(())
}
