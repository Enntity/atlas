// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// Qwen3.8's checkpoint bound (4096^2), which the encoder clamps to 2048^2.
const Q38: Option<usize> = Some(16_777_216);

fn qwen() -> VisionConfig {
    super::super::tests::ok_cfg()
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xEDB8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend((data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend(kind);
    out.extend(data);
    let crc = crc32(&out[start..]);
    out.extend(crc.to_be_bytes());
}

/// A ~60-byte PNG that DECLARES `w x h` RGB but carries a truncated IDAT: its
/// header reads fine, and any attempt to decode it fails.
fn header_only_png(w: u32, h: u32) -> String {
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = Vec::new();
    ihdr.extend(w.to_be_bytes());
    ihdr.extend(h.to_be_bytes());
    ihdr.extend([8, 2, 0, 0, 0]); // 8-bit RGB
    chunk(&mut png, b"IHDR", &ihdr);
    chunk(&mut png, b"IDAT", &[0x78, 0x9c]);
    chunk(&mut png, b"IEND", &[]);
    let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &png);
    format!("data:image/png;base64,{b64}")
}

fn real_png(w: u32, h: u32) -> String {
    let img = image::RgbImage::from_pixel(w, h, image::Rgb([200, 30, 90]));
    let mut buf = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut buf, image::ImageFormat::Png)
        .unwrap();
    let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, buf.get_ref());
    format!("data:image/png;base64,{b64}")
}

#[test]
fn a_huge_declared_image_is_sized_from_its_header_without_decoding() {
    // 16000x12000 declared in ~60 bytes: decoded and patched it would be
    // hundreds of MiB. The probe sizes it (clamped to the encoder's 2048^2
    // capacity) from the header alone; the decode, never reached by an
    // admission that refuses it, would fail on the truncated data.
    let uri = header_only_png(16_000, 12_000);
    let t = std::time::Instant::now();
    let probed = probe_image(&uri, &qwen(), Q38).unwrap();
    assert!(
        t.elapsed().as_millis() < 500,
        "no pixel work: {:?}",
        t.elapsed()
    );
    assert!(probed.patches() <= 16_384, "{probed:?}");
    assert_eq!(probed.pad_count(2), probed.patches() / 4);
    assert!(
        probed.preprocess(&qwen()).is_err(),
        "the pixels were never there"
    );
}

#[test]
fn a_header_past_the_decoder_limit_is_refused_at_the_probe() {
    assert!(probe_image(&header_only_png(20_000, 100), &qwen(), Q38).is_err());
}

#[test]
fn stills_past_one_encoder_batch_are_resized_into_it() {
    // A 2880x1800 Retina screenshot used to come out 2880x1792 = 20160
    // patches, and 2064x2032 rounded to 2080x2048 = 16640: both over the
    // encoder's 16384, failing at prefill. Both now fit, floored as HF does.
    for (w, h) in [(2880u32, 1800u32), (2064, 2032), (1800, 2880)] {
        let probed = probe_image(&header_only_png(w, h), &qwen(), Q38).unwrap();
        assert!(probed.patches() <= 16_384, "{w}x{h}: {probed:?}");
        assert!(
            probed.patches() > 15_000,
            "{w}x{h} kept its detail: {probed:?}"
        );
    }
    let (th, tw) = still_target(&qwen(), 2032, 2064, Q38).unwrap();
    assert_eq!((th, tw), (2016, 2048), "HF smart_resize's floor");
}

#[test]
fn a_full_decode_matches_its_probe() {
    let uri = real_png(2064, 2032);
    let probed = probe_image(&uri, &qwen(), Q38).unwrap();
    let (gh, gw) = (probed.grid_h, probed.grid_w);
    let (pixels, ph, pw) = probed.preprocess(&qwen()).unwrap();
    assert_eq!((ph, pw), (gh, gw));
    assert_eq!(pixels.len(), ph * pw * 3 * 2 * 16 * 16);
    assert_eq!(
        preprocess_image_with_max_pixels(&uri, &qwen(), Q38)
            .unwrap()
            .1,
        ph,
        "the one-call path is the probe plus the decode"
    );
}

#[test]
fn images_inside_the_bound_keep_their_size() {
    // Exactness: an image whose rounded size already fit is unchanged.
    for (w, h, want) in [(2048u32, 2048u32, (2048, 2048)), (1920, 1080, (1088, 1920))] {
        assert_eq!(still_target(&qwen(), h, w, Q38).unwrap(), want, "{w}x{h}");
    }
}
