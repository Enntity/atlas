# SPDX-License-Identifier: AGPL-3.0-only
"""Pin and extract the actual production CUTLASS collective/launch assembly."""
from pathlib import Path
import hashlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
source = ROOT / "crates/spark-runtime/cuda/cutlass_nvfp4_grouped_gemm.cu"
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "681f36a9ecd4069fbf04ead47965b3327fe06d304bdb4d77b06b017544f6f00a"
s = raw.decode()
header = s[:s.index("// FP4 (e2m1) quantization helper")]
struct = s[s.index("struct GroupedAPrep {"):s.index("// Gather+pack A once")]
launch = s[s.index("static int launch_projection("):s.index("#endif  // arch guard", s.index("static int launch_projection("))]
assert launch.count("cudaMemcpyAsync(") == 1
launch = launch.replace("int tag) {", "int tag, bool dry_run, size_t* required) {")
launch = launch.replace("    void* dst = ws + cursor;", "    if (cursor > workspace_size || bytes > workspace_size - cursor) { std::fprintf(stderr, \"metadata capacity exceeded\\n\"); std::abort(); }\n    void* dst = ws + cursor;")
launch = launch.replace("    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);", "    if (!dry_run) CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));")
launch = launch.replace("  if (cursor + need > workspace_size) {", "  if (required) *required = cursor + need;\n  if (cursor > workspace_size || need > workspace_size - cursor) {")
launch = launch.replace("  cutlass::Status st = gemm.initialize", "  if (dry_run) return 0;\n  cutlass::Status st = gemm.initialize")
assert "if (!dry_run) CHECK(cudaMemcpyAsync" in launch
assert launch.index("if (dry_run) return 0;") < launch.index("gemm.initialize")
default = header + "\n" + struct + launch + "\n#endif\n"
(HERE / "collective.cuh").write_text(default)
assert default.count("using TileShape = Shape<_128, _128, _128>;") == 1
assert default.count("cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>::CollectiveOp") == 1
cooperative = default.replace("using TileShape = Shape<_128, _128, _128>;", "using TileShape = Shape<_128, _128, _256>;")
cooperative = cooperative.replace("cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>::CollectiveOp", "cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::CollectiveOp")
(HERE / "collective_cooperative.cuh").write_text(cooperative)
print("PASS pinned production collective; dry planning cannot copy or launch")
