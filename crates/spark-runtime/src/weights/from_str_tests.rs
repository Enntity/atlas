// SPDX-License-Identifier: AGPL-3.0-only

use super::WeightDtype;

#[test]
fn from_safetensors_str_matches_disk_mapping() {
    use WeightDtype::*;
    for (source, expected) in [
        ("F32", FP32),
        ("F16", FP16),
        ("BF16", BF16),
        ("U8", UInt8),
        ("I8", UInt8),
        ("F8_E4M3", FP8E4M3),
        ("F8_E8M0", FP8E8M0),
        ("I16", Int16),
        ("I32", Int32),
        ("I64", Int64),
    ] {
        assert_eq!(
            WeightDtype::from_safetensors_str(source).unwrap(),
            expected,
            "dtype {source}"
        );
    }
    assert!(WeightDtype::from_safetensors_str("bogus").is_err());
}

#[test]
fn exl3_raw_tensors_preserve_storage_dtypes() {
    use WeightDtype::*;
    assert_eq!(
        WeightDtype::from_safetensors_str_for_tensor(
            "F16",
            "model.layers.3.mlp.experts.0.gate_proj.suh"
        )
        .unwrap(),
        (FP16, false)
    );
    assert_eq!(
        WeightDtype::from_safetensors_str_for_tensor("I16", "x.trellis").unwrap(),
        (Int16, false)
    );
    assert_eq!(
        WeightDtype::from_safetensors_str_for_tensor("I32", "x.mcg").unwrap(),
        (Int32, false)
    );
    assert_eq!(
        WeightDtype::from_safetensors_str_for_tensor("F16", "model.layers.0.weight").unwrap(),
        (BF16, true)
    );
}

#[test]
fn f16_bytes_convert_to_bf16_via_f32() {
    use half::{bf16, f16};
    let values = [0.0f32, 1.0, -1.5, 0.1, 65504.0, -6.1035156e-5];
    let source = values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_le_bytes())
        .collect::<Vec<_>>();
    let output = super::f16_to_bf16_bytes(&source);
    assert_eq!(output.len(), source.len());
    for (index, value) in values.iter().enumerate() {
        let got = bf16::from_le_bytes([output[2 * index], output[2 * index + 1]]);
        let expected = bf16::from_f32(f16::from_f32(*value).to_f32());
        assert_eq!(got, expected, "value {value}");
    }
}
