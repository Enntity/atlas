# ExLlamaV3 EXL3 CUDA core

These files are the minimal CUDA dependency closure used by Atlas's
GLM-5.3-Flash routed-expert kernel. They come from ExLlamaV3 commit
`c5d9c657966ffeeaa9353f0cc899f18629da4a13` (package version `0.0.43`) and
remain under the upstream MIT license in `LICENSE`.

`quant/exl3_moe_kernel.cuh` is specialized to the checkpoint's 4-bit MCG
codebook and 256-column tile, and exports the raw-pointer
`glm53_exl3_moe` entry point. All other vendored files are unmodified.

