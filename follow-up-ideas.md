- ATLAS_KV_PAGING seems to allow for cache sharing between nodes. does this effectively double usable kv cache size?
- nvme offload exists- add logic to allow for secondary offload to NFSv4 shares. So a process evicts from RAM to local nvme. local nvme evicts to nfs that is shared across all nodes
  - follow up on the USB-C Gadget idea to take advantage of DMA to allow for an external device ('ai-node') to act as a L1.5 KV cache. Assuming the end to end latency is lower 
    than NFS it might be worth to investigate. In theory the sparks can use DMA to read/write to a shared memory region on ai-node. 
- kv cache deduplication? 
- kv cache compression 
- kv spectulitive generation? 
- drafter draft caching? can drafted tokens be kept and re-used?
- revisit external optimizer for drafting / draft speculation?
  - incrementally improve the drafter by adding new training data live based on acceptence rates, optimizing acceptence rates at the end
- Atlas cpu profiling and thread pinning? 
- EXL3 GLM-5.3 launch/parity/perf follow-ups (incl. reference-policy KDA core, M128
  residual, perf areas) are tracked in `todo.md`. Note: new Rust source needs the
  AGPL SPDX header and <=500 LoC; new Python harnesses should match the SPDX
  header style of their neighbors in `tests/exl3_validation/`.
