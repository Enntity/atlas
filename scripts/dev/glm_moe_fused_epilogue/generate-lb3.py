# SPDX-License-Identifier: AGPL-3.0-only
"""One derivative: constrain only the existing fused wrapper to three CTAs."""
import hashlib
from pathlib import Path

here = Path(__file__).resolve().parent
engine = here.parents[2]
expected = {
    here / "generated.cuh": "ea5dae666b4f811d09e148b0f5b059c50874c7a17b911d0f3fbccbadc286b50b",
    here / "bench.cu": "e5019bddf051d0f266925ec7495dee6627b9a0504d904689c4d1547a97b22039",
    engine / "kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu":
        "bad2afc59183a4b46fd094864ef88ae428517a23dee0d00f1d9e87de7503b33b",
}
for path, digest in expected.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path

original = (here / "generated.cuh").read_text()
old = 'extern "C" __global__ void atlas_dev_moe_fused_gate_up('
new = 'extern "C" __global__ __launch_bounds__(128,3) void atlas_dev_moe_fused_gate_up('
assert original.count(old) == 1 and "__launch_bounds__" not in original
candidate = original.replace(old, new)
assert candidate.replace(new, old) == original
(here / "generated-lb3.cuh").write_text(candidate)

bench = (here / "bench.cu").read_text()
bench = bench.replace('#include "generated.cuh"', '#include "generated-lb3.cuh"')
old_report = '''    std::printf("resources baseline_regs=%d candidate_regs=%d baseline_smem=%zu candidate_smem=%zu baseline_ctas=%d candidate_ctas=%d\\n",
        a.numRegs,b.numRegs,a.sharedSizeBytes,b.sharedSizeBytes,ac,bc);'''
new_report = '''    std::printf("resources baseline_regs=%d candidate_regs=%d baseline_smem=%zu candidate_smem=%zu baseline_ctas=%d candidate_ctas=%d baseline_local=%zu candidate_local=%zu\\n",
        a.numRegs,b.numRegs,a.sharedSizeBytes,b.sharedSizeBytes,ac,bc,a.localSizeBytes,b.localSizeBytes);
    require(b.localSizeBytes==0,"lb3 requires zero candidate local memory");
    require(bc>=3,"lb3 requires at least three candidate CTAs per SM");'''
assert bench.count(old_report) == 1
bench = bench.replace(old_report, new_report)
old_usage = '''    require(argc==2&&(!std::strcmp(argv[1],"--host-test")||!std::strcmp(argv[1],"--run")),"usage: bench --host-test|--run");'''
new_usage = '''    require(argc==2&&(!std::strcmp(argv[1],"--host-test")||!std::strcmp(argv[1],"--run")||!std::strcmp(argv[1],"--resources")),"usage: bench --host-test|--resources|--run");'''
assert bench.count(old_usage) == 1
bench = bench.replace(old_usage, new_usage)
marker = '''#else
    size_t free,total;CHECK(cudaMemGetInfo(&free,&total));'''
replacement = '''#else
    if(!std::strcmp(argv[1],"--resources")){
        resources();std::puts("PASS lb3 resource gate; no device buffers or kernel launches");return 0;
    }
    size_t free,total;CHECK(cudaMemGetInfo(&free,&total));'''
assert bench.count(marker) == 1
bench = bench.replace(marker, replacement)
assert 'if(gain<1.2)' in bench
# The actual fixture/timing/oracle body is exactly the retained source.
section = lambda text: text[text.index("static double run_case("):text.index("\n#endif\nint main(")]
assert section(bench) == section((here / "bench.cu").read_text())
(here / "bench-lb3.cu").write_text(bench)
for name in ("generated-lb3.cuh", "bench-lb3.cu"):
    print(name, hashlib.sha256((here / name).read_bytes()).hexdigest())
print("PASS candidate differs by one wrapper attribute; baseline and full run_case unchanged")
