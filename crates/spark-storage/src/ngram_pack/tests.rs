// SPDX-License-Identifier: AGPL-3.0-only

//! The packer's offset math, on synthetic split-row tables whose shards are
//! scattered the way the FP4-Local checkpoint's are: several files, shards
//! out of order, other bytes in between, and the scale plane in a different
//! region from the data plane.

use super::*;

/// A deterministic, row- and plane-unique byte, so a row read from the wrong
/// shard, plane or offset can never compare equal by accident.
fn byte(plane: usize, row: u64, i: usize) -> u8 {
    (row.wrapping_mul(131) ^ (plane as u64 * 0x55) ^ (i as u64 * 7)) as u8
}

struct Fixture {
    dir: PathBuf,
    data: Vec<(PathBuf, u64)>,
    scale: Vec<(PathBuf, u64)>,
    rows: u64,
}

/// Three shards over two files. File A: junk, data shard 2, junk, data
/// shard 0, scale shard 1. File B: scale shard 0, junk, data shard 1, scale
/// shard 2. Data rows 5 B, scale rows 3 B, odd offsets throughout.
fn fixture(tag: &str, rows: u64) -> Fixture {
    let dir = std::env::temp_dir().join(format!("ngram_pack_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (a, b) = (dir.join("a.safetensors"), dir.join("b.safetensors"));
    let (dw, sw) = (5usize, 3usize);
    let mut fa = vec![0xAAu8; 13];
    let mut fb = Vec::new();
    let mut data = vec![(PathBuf::new(), 0u64); 3];
    let mut scale = vec![(PathBuf::new(), 0u64); 3];
    let put = |buf: &mut Vec<u8>, path: &Path, plane: usize, shard: usize, w: usize| {
        let off = buf.len() as u64;
        for r in 0..rows {
            let g = shard as u64 * rows + r;
            buf.extend((0..w).map(|i| byte(plane, g, i)));
        }
        (path.to_path_buf(), off)
    };
    data[2] = put(&mut fa, &a, 0, 2, dw);
    fa.extend([0xEEu8; 9]);
    data[0] = put(&mut fa, &a, 0, 0, dw);
    scale[1] = put(&mut fa, &a, 1, 1, sw);
    scale[0] = put(&mut fb, &b, 1, 0, sw);
    fb.extend([0xBBu8; 4097]);
    data[1] = put(&mut fb, &b, 0, 1, dw);
    scale[2] = put(&mut fb, &b, 1, 2, sw);
    std::fs::write(&a, &fa).unwrap();
    std::fs::write(&b, &fb).unwrap();
    Fixture {
        dir,
        data,
        scale,
        rows,
    }
}

impl Fixture {
    fn spec(&self) -> PackSpec<'_> {
        PackSpec {
            planes: vec![
                Plane {
                    shards: &self.data,
                    row_bytes: 5,
                },
                Plane {
                    shards: &self.scale,
                    row_bytes: 3,
                },
            ],
            rows_per_shard: self.rows,
            stride: 16,
            tag: "test",
        }
    }
}

/// Overwrite `path` in place (same inode) and restore its mtime.
fn rewrite_keeping_mtime(path: &Path, bytes: &[u8]) {
    let mtime = std::fs::metadata(path).unwrap().modified().unwrap();
    std::fs::write(path, bytes).unwrap();
    std::fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(mtime)
        .unwrap();
}

fn expected_row(g: u64) -> Vec<u8> {
    let mut v: Vec<u8> = (0..5).map(|i| byte(0, g, i)).collect();
    v.extend((0..3).map(|i| byte(1, g, i)));
    v.resize(16, 0);
    v
}

#[test]
fn interleave_concatenates_planes_and_zeroes_the_tail() {
    let a = [1u8, 2, 3, 4];
    let b = [9u8, 8];
    let mut out = [0xFFu8; 8];
    interleave(&[&a, &b], &[2, 1], 4, &mut out);
    assert_eq!(out, [1, 2, 9, 0, 3, 4, 8, 0]);
}

#[test]
fn source_loc_follows_each_shard_and_plane_base() {
    let f = fixture("loc", 10);
    let s = f.spec();
    // Row 23 = shard 2, local 3. Each plane has its own base AND stride.
    assert_eq!(
        s.source_loc(0, 23),
        (f.data[2].0.as_path(), f.data[2].1 + 3 * 5)
    );
    assert_eq!(
        s.source_loc(1, 23),
        (f.scale[2].0.as_path(), f.scale[2].1 + 3 * 3)
    );
    assert_eq!(s.source_loc(0, 10), (f.data[1].0.as_path(), f.data[1].1));
    assert_eq!(packed_offset(0, 96), HEADER);
    assert_eq!(packed_offset(320_001_535, 96), HEADER + 320_001_535 * 96);
    assert_eq!(HEADER % 4096, 0, "row 0 must start O_DIRECT-aligned");
    std::fs::remove_dir_all(&f.dir).unwrap();
}

/// Every packed row, read back at `HEADER + row * stride`, is the two source
/// rows it claims to be — across chunk boundaries (chunk 7 does not divide
/// the 10-row shards) and across files.
#[test]
fn pack_reassembles_every_row_from_scattered_shards() {
    let f = fixture("rows", 10);
    let spec = f.spec();
    let out = f.dir.join("packed.bin");
    pack(&spec, &out, 7, 4).unwrap();
    let bytes = std::fs::read(&out).unwrap();
    assert_eq!(bytes.len() as u64, packed_offset(30, 16));
    for g in 0..30u64 {
        let at = packed_offset(g, 16) as usize;
        assert_eq!(&bytes[at..at + 16], expected_row(g).as_slice(), "row {g}");
    }
    validate(&spec, &out, Verify::Full).unwrap();
    std::fs::remove_dir_all(&f.dir).unwrap();
}

#[test]
fn open_or_pack_reuses_a_valid_file_and_rebuilds_a_stale_one() {
    let f = fixture("reuse", 10);
    let spec = f.spec();
    let cache = f.dir.join("derived");
    let t = open_or_pack(&spec, &cache, Verify::Full).unwrap();
    assert_eq!((t.base_offset, t.rows_total, t.stride), (HEADER, 30, 16));
    assert!(t.path.starts_with(&cache));
    assert!(
        t.path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("ngram-test-")
    );

    // A second open must not rewrite it: stamp a byte in the header padding
    // (outside anything validated) and see it survive.
    let mut bytes = std::fs::read(&t.path).unwrap();
    bytes[4000] = 0x5A;
    std::fs::write(&t.path, &bytes).unwrap();
    let again = open_or_pack(&spec, &cache, Verify::Full).unwrap();
    assert_eq!(again.path, t.path);
    assert_eq!(std::fs::read(&t.path).unwrap()[4000], 0x5A);

    // A valid table on a read-only directory (packed at install time) opens
    // without the lock file it could not create.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::remove_file(cache.join(format!("{}.lock", spec.file_name().unwrap())));
        std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o555)).unwrap();
        let ro = open_or_pack(&spec, &cache, Verify::Full);
        std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(ro.unwrap().path, t.path);
    }

    // Same layout, new values: change a SAMPLED row (shard 0, row 0) in the
    // source and put its mtime back. Identity is unchanged (same names,
    // lengths, offsets, mtimes, inodes), so only the row spot-check can catch
    // it — and it must, then rebuild.
    let (a, off) = f.data[0].clone();
    let mut src = std::fs::read(&a).unwrap();
    src[off as usize] ^= 0xFF;
    rewrite_keeping_mtime(&a, &src);
    let e = validate(&spec, &t.path, Verify::Full).unwrap_err();
    assert!(
        format!("{e:#}").contains("does not match its sources"),
        "{e:#}"
    );
    let rebuilt = open_or_pack(&spec, &cache, Verify::Full).unwrap();
    let bytes = std::fs::read(&rebuilt.path).unwrap();
    assert_eq!(bytes[HEADER as usize], byte(0, 0, 0) ^ 0xFF);
    assert_eq!(bytes[4000], 0, "rebuilt from scratch");
    let leftovers: Vec<_> = std::fs::read_dir(&cache)
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().contains(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "temp files left behind");
    std::fs::remove_dir_all(&f.dir).unwrap();
}

#[test]
fn identity_names_sources_and_geometry() {
    let f = fixture("id", 10);
    let spec = f.spec();
    let id = spec.identity().unwrap();
    let name = spec.file_name().unwrap();
    // A longer source file is a different checkpoint: new identity, new name.
    let (b, _) = &f.data[1];
    let mut src = std::fs::read(b).unwrap();
    src.push(0);
    std::fs::write(b, &src).unwrap();
    assert_ne!(spec.identity().unwrap(), id);
    assert_ne!(spec.file_name().unwrap(), name);
    // Stride is geometry too.
    let wider = PackSpec {
        stride: 32,
        ..f.spec()
    };
    assert_ne!(wider.identity().unwrap(), spec.identity().unwrap());
    std::fs::remove_dir_all(&f.dir).unwrap();
}

#[test]
fn bad_specs_and_bad_files_are_refused() {
    let f = fixture("bad", 10);
    // Planes wider than the stride.
    let narrow = PackSpec {
        stride: 7,
        ..f.spec()
    };
    assert!(narrow.identity().is_err());
    // Mismatched shard counts between planes.
    let short = PackSpec {
        planes: vec![
            Plane {
                shards: &f.data,
                row_bytes: 5,
            },
            Plane {
                shards: &f.scale[..2],
                row_bytes: 3,
            },
        ],
        ..f.spec()
    };
    assert!(short.identity().is_err());
    // Truncated and wrong-magic files.
    let spec = f.spec();
    let out = f.dir.join("p.bin");
    pack(&spec, &out, 64, 4).unwrap();
    let full = std::fs::read(&out).unwrap();
    std::fs::write(&out, &full[..full.len() - 1]).unwrap();
    assert!(validate(&spec, &out, Verify::Full).is_err());
    let mut bad = full.clone();
    bad[0] = b'X';
    std::fs::write(&out, &bad).unwrap();
    assert!(validate(&spec, &out, Verify::Full).is_err());
    std::fs::remove_dir_all(&f.dir).unwrap();
}

#[test]
fn sample_rows_stay_in_range_and_cover_both_ends() {
    assert_eq!(sample_row(0, 2_500_012), 0);
    assert_eq!(sample_row(1, 2_500_012), 2_500_011);
    for s in 0..128 {
        assert!(sample_row(s, 2_500_012) < 2_500_012);
    }
}

/// The FP4-Local geometry: 128 shards per plane over 7 files with 10-digit
/// offsets. Its identity must fit the header, or the first real pack fails.
#[test]
fn identity_of_the_real_table_fits() {
    let dir = std::env::temp_dir().join(format!("ngram_pack_real_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let files: Vec<PathBuf> = (15..22)
        .map(|i| {
            let p = dir.join(format!("model-{i:05}-of-00021.safetensors"));
            std::fs::write(&p, b"x").unwrap();
            p
        })
        .collect();
    let plane = |base: u64| -> Vec<(PathBuf, u64)> {
        (0..128u64)
            .map(|s| (files[(s % 7) as usize].clone(), base + s * 200_000_960))
            .collect()
    };
    let (codes, scales) = (plane(3_488_889_896), plane(2_950_357_208));
    let spec = PackSpec {
        planes: vec![
            Plane {
                shards: &codes,
                row_bytes: 80,
            },
            Plane {
                shards: &scales,
                row_bytes: 10,
            },
        ],
        rows_per_shard: 2_500_012,
        stride: 96,
        tag: "nvfp4g16",
    };
    let id = spec.identity().unwrap();
    let chunks = spec.rows_total().div_ceil(spec.check_rows());
    println!("identity: {} bytes, {chunks} chunk checksums", id.len());
    assert!(id.len() > 4096, "the case the 64 KiB header exists for");
    assert_eq!(spec.check_rows(), CHUNK_ROWS, "one checksum per pack pass");
    assert_eq!(chunks, 1221);
    header_fits(id.len() as u64, chunks).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A corrupt or hostile header length must be refused, never trusted: on the
/// old `16 + len` check a length near `u64::MAX` wrapped past the range test
/// in a release build and `vec![0; len]` aborted the process, so a bad packed
/// file crash-looped the server instead of being rebuilt.
#[test]
fn a_corrupt_header_length_is_refused_and_rebuilt() {
    let f = fixture("hdrlen", 10);
    let spec = f.spec();
    let cache = f.dir.join("derived");
    let t = open_or_pack(&spec, &cache, Verify::Full).unwrap();
    let good = std::fs::read(&t.path).unwrap();
    for len in [u64::MAX, u64::MAX - 15, u64::MAX / 2, HEADER, HEADER - 15] {
        let mut bad = good.clone();
        bad[8..16].copy_from_slice(&len.to_le_bytes());
        std::fs::write(&t.path, &bad).unwrap();
        let e = validate(&spec, &t.path, Verify::Full).unwrap_err();
        assert!(
            format!("{e:#}").contains("out of range"),
            "len {len}: {e:#}"
        );
        // `open_or_pack` treats it as stale and rebuilds a valid file.
        let again = open_or_pack(&spec, &cache, Verify::Full).unwrap();
        assert_eq!(again.path, t.path);
        validate(&spec, &t.path, Verify::Full).unwrap();
    }
    std::fs::remove_dir_all(&f.dir).unwrap();
}

/// A packed file damaged after it was written, in a row the per-shard spot
/// check never reads, is caught by its chunk checksum: always under
/// `Verify::Full`, and under `Verify::Sample` whenever the sample includes the
/// chunk. `open_or_pack` then rebuilds it.
#[test]
fn a_damaged_unsampled_row_fails_its_chunk_checksum() {
    let f = fixture("chunks", 10);
    let spec = f.spec();
    let cache = f.dir.join("derived");
    let path = cache.join(spec.file_name().unwrap());
    std::fs::create_dir_all(&cache).unwrap();
    // 30 rows, 4 per chunk: 8 chunks, the last one short.
    pack(&spec, &path, 7, 4).unwrap();
    validate(&spec, &path, Verify::Full).unwrap();
    // Row 13 (chunk 3) is no shard's spot-check row (0, 19, and shard 2's).
    assert!((0..3).all(|s| s * 10 + sample_row(s, 10) != 13));
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[packed_offset(13, 16) as usize + 2] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();

    let e = validate(&spec, &path, Verify::Full).unwrap_err();
    assert!(format!("{e:#}").contains("chunk 3"), "{e:#}");
    let f2 = File::open(&path).unwrap();
    let hdr = read_header(&f2).unwrap();
    assert_eq!((hdr.check_rows, hdr.sums.len()), (4, 8));
    // The chunks the sample would have to draw: the damaged one fails, the
    // rest pass, so it is the checksum and not the spot check that caught it.
    assert!(verify_chunks(&f2, &hdr, 30, 16, &[3]).is_err());
    verify_chunks(&f2, &hdr, 30, 16, &[0, 1, 2, 4, 5, 6, 7]).unwrap();
    drop(f2);

    let t = open_or_pack(&spec, &cache, Verify::Full).unwrap();
    assert_eq!(t.path, path);
    let rebuilt = std::fs::read(&path).unwrap();
    let at = packed_offset(13, 16) as usize;
    assert_eq!(&rebuilt[at..at + 16], expected_row(13).as_slice());
    std::fs::remove_dir_all(&f.dir).unwrap();
}

/// A source rewritten in place with the same name and length is a new
/// identity through its mtime alone, and is rebuilt under the SAME file name
/// (the name depends on the layout only), leaving no stale copy behind.
#[test]
fn a_touched_source_is_a_new_identity_under_the_same_name() {
    let f = fixture("touch", 10);
    let spec = f.spec();
    let cache = f.dir.join("derived");
    let t = open_or_pack(&spec, &cache, Verify::Sample).unwrap();
    let (id, name) = (spec.identity().unwrap(), spec.file_name().unwrap());
    let (b, _) = &f.data[1];
    let later =
        std::fs::metadata(b).unwrap().modified().unwrap() + std::time::Duration::from_secs(5);
    File::options()
        .write(true)
        .open(b)
        .unwrap()
        .set_modified(later)
        .unwrap();
    assert_ne!(spec.identity().unwrap(), id);
    assert_eq!(spec.file_name().unwrap(), name);
    let e = validate(&spec, &t.path, Verify::Sample).unwrap_err();
    assert!(format!("{e:#}").contains("identity differs"), "{e:#}");
    assert_eq!(
        open_or_pack(&spec, &cache, Verify::Sample).unwrap().path,
        t.path
    );
    validate(&spec, &t.path, Verify::Full).unwrap();
    let bins = std::fs::read_dir(&cache)
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".bin"))
        .count();
    assert_eq!(bins, 1);
    std::fs::remove_dir_all(&f.dir).unwrap();
}

/// Corrupt checksum geometry in the header is refused, never trusted.
#[test]
fn corrupt_checksum_geometry_is_refused() {
    let f = fixture("geom", 10);
    let spec = f.spec();
    let out = f.dir.join("p.bin");
    pack(&spec, &out, 64, 4).unwrap();
    let good = std::fs::read(&out).unwrap();
    for (at, value) in [
        (24, u64::MAX),     // chunk count
        (24, u64::MAX / 4), // chunk count that wraps when multiplied
        (24, HEADER),       // more sums than the header holds
        (24, 7),            // fits, but not ceil(30 / 4)
        (16, 0),            // zero rows per chunk
        (16, 5),            // rows per chunk that does not give 8 chunks
    ] {
        let mut bad = good.clone();
        bad[at..at + 8].copy_from_slice(&value.to_le_bytes());
        std::fs::write(&out, &bad).unwrap();
        assert!(
            validate(&spec, &out, Verify::Full).is_err(),
            "word @{at} = {value} accepted"
        );
    }
    std::fs::remove_dir_all(&f.dir).unwrap();
}

#[test]
fn verify_mode_parses_leniently() {
    assert_eq!(Verify::parse(Some("full")), Verify::Full);
    for v in [None, Some(""), Some("1"), Some("FULL"), Some("bogus")] {
        assert_eq!(Verify::parse(v), Verify::Sample, "{v:?}");
    }
}

/// The cost of the default verify, at the NVFP4 table's real geometry
/// (96-byte rows, 24 MiB chunks), on whatever disk `$TMPDIR` is: packs a
/// 20-chunk (504 MB) single-plane table and times `Sample` (16 chunks) and
/// `Full`. Not a unit test; run by hand:
/// ```text
/// cargo test -p spark-storage --release --lib verify_cost -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn verify_cost() {
    let dir = std::env::temp_dir().join(format!("ngram_pack_cost_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("src.bin");
    let rows = 20 * (CHECK_BYTES / 96);
    let bytes: Vec<u8> = (0..rows * 90)
        .map(|i| ((i * 2654435761) >> 13) as u8)
        .collect();
    std::fs::write(&src, &bytes).unwrap();
    drop(bytes);
    let shards = vec![(src, 0u64)];
    let spec = PackSpec {
        planes: vec![Plane {
            shards: &shards,
            row_bytes: 90,
        }],
        rows_per_shard: rows,
        stride: 96,
        tag: "cost",
    };
    let t = std::time::Instant::now();
    let table = open_or_pack(&spec, &dir, Verify::Sample).unwrap();
    println!("pack + sample verify: {:.2} s", t.elapsed().as_secs_f64());
    for mode in [Verify::Sample, Verify::Full, Verify::Sample] {
        let t = std::time::Instant::now();
        validate(&spec, &table.path, mode).unwrap();
        println!(
            "{mode:?} verify of 20 chunks: {:.3} s",
            t.elapsed().as_secs_f64()
        );
    }
    let mut h = crc32fast::Hasher::new();
    let buf = vec![7u8; 256 << 20];
    let t = std::time::Instant::now();
    h.update(&buf);
    let _ = h.finalize();
    println!(
        "crc32: {:.2} GB/s",
        buf.len() as f64 / 1e9 / t.elapsed().as_secs_f64()
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
