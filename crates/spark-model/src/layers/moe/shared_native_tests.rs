// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn native(prefill: Fp8bPrefill) -> SharedNative {
    let w = |p: u64| Fp8BlockWeight {
        data: DevicePtr(p),
        scales: DevicePtr(p + 0x1_0000),
        tiled: false,
    };
    SharedNative {
        gate: w(0x100_0000),
        up: w(0x200_0000),
        down: w(0x300_0000),
        prefill,
    }
}

#[test]
fn each_ranks_half_is_its_rows_of_gate_up_and_columns_of_down() {
    // GLM: h 4096, inter 2048; rank r owns intermediate columns [1024 r, +1024).
    let (h, inter) = (4096usize, 2048usize);
    let n = native(Fp8bPrefill::W8a8);
    let r0 = n.split(0, h, inter);
    assert_eq!(r0.gate, n.gate);
    assert_eq!(r0.up, n.up);
    assert_eq!((r0.down_data, r0.down_scales), (n.down.data, n.down.scales));
    let r1 = n.split(1, h, inter);
    // 1024 rows of 4096 values; 8 grid rows of 32 f32.
    assert_eq!(r1.gate.data, DevicePtr(0x100_0000 + 1024 * 4096));
    assert_eq!(r1.gate.scales, DevicePtr(0x101_0000 + 8 * 32 * 4));
    assert_eq!(r1.up.data, DevicePtr(0x200_0000 + 1024 * 4096));
    // down [4096, 2048]: 1024 values and 8 grid entries into each row.
    assert_eq!(r1.down_data, DevicePtr(0x300_0000 + 1024));
    assert_eq!(r1.down_scales, DevicePtr(0x301_0000 + 8 * 4));
}

#[test]
fn prefill_gemm_is_w8a8_or_exact_w8a16() {
    assert_eq!(native(Fp8bPrefill::W8a8).gemm_mode(), Fp8bPrefill::W8a8);
    assert_eq!(native(Fp8bPrefill::W8a16).gemm_mode(), Fp8bPrefill::W8a16);
    // A shared expert has no BF16 copy: `bf16` means the exact W8A16 GEMM.
    assert_eq!(native(Fp8bPrefill::Bf16).gemm_mode(), Fp8bPrefill::W8a16);
}
