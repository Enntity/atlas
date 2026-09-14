# SPDX-License-Identifier: AGPL-3.0-only
"""Spread the existing four QK tiles across eight warps; retain softmax/PV."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[2]/'kernels/gb10/deepseek-v4-flash/nvfp4/glm_sparse_prefill_kv_reuse.cu'
src=SOURCE.read_text()
sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
expected='b7675a6058e4eb9bcc011a9ed0ccedc02884fb68ede9195b1010f3ac0c485f39'
assert sha==expected,'production source changed; review before regeneration'
src=src.replace('glm_sparse_mla_prefill_bf16_head32_tc_kv_pad','atlas_dev_sparse_qk8')
src=src.replace('glm_kvp_','glm_qk8_').replace('GLM_KVP_STRIDE','GLM_QK8_STRIDE')
src=src.replace('shared69376 bytes','shared73472 bytes')
needle='smem_P + (unsigned int)BR_512 * (BC_512 + PAD_P_512));'
assert src.count(needle)==1
src=src.replace(needle,needle+'\n    float* smem_scores = smem_ml + 64; // 32 heads x32 keys FP32, 4096 bytes.')
a=src.index('        float acc_s[4][4];')
b=src.index('            unsigned int row0=qk_warp_m+group_id, row1=row0+8;',a)
old=src[a:b]
# Extract the original fragment loads and exact MMA, remove only the nt loop.
loop=old[old.index('            #pragma unroll\n            for (unsigned int ks'):]
loop=loop[:loop.rindex('\n            }')+len('\n            }')]
loop=loop.replace('                #pragma unroll\n                for (int nt=0; nt<4; nt++) {','                {')
loop=loop.replace('nc=nt*8+group_id','nc=qk_nt*8+group_id').replace('acc_s[nt]','qk_acc')
replacement='''        // Each warp owns one16-head x8-key score tile; its K chain is unchanged.
        {
            const unsigned qk_nt=warp_id>>1;
            float qk_acc[4]={0.f,0.f,0.f,0.f};
            const unsigned short* sQ=(const unsigned short*)smem_Q;
            const unsigned short* sK=(const unsigned short*)smem_K;
'''+loop+'''
            #pragma unroll
            for(int c=0;c<4;c++) smem_scores[warp_id*128+c*32+lane_id]=qk_acc[c];
        }
        __syncthreads();
        float acc_s[4][4];
        if (warp_id < 2) {
            #pragma unroll
            for(int nt=0;nt<4;nt++) {
                #pragma unroll
                for(int c=0;c<4;c++) acc_s[nt][c]=smem_scores[(nt*2+warp_id)*128+c*32+lane_id];
            }

'''
candidate=src[:a]+replacement+src[b:]
# No scaling, masking, softmax, PV, cache loading or final-store changes.
assert candidate[candidate.index('            unsigned int row0=qk_warp_m+group_id, row1=row0+8;'):]==src[b:]
# Every score has exactly one writer; warps0/1 reload the original fragment mapping.
written={}
for tid in range(256):
 warp,lane=divmod(tid,32);group,sub=divmod(lane,4)
 for rr in [0,8]:
  for cc in [0,1]:
   pos=((warp&1)*16+group+rr,(warp>>1)*8+sub*2+cc)
   assert pos not in written
   written[pos]=(warp,rr,cc)
assert len(written)==1024
read=set()
for tid in range(64):
 warp,lane=divmod(tid,32);group,sub=divmod(lane,4)
 for nt in range(4):
  for rr in [0,8]:
   for cc in [0,1]:
    pos=(warp*16+group+rr,nt*8+sub*2+cc)
    assert written[pos]==(nt*2+warp,rr,cc)
    read.add(pos)
assert len(read)==1024
for warp in range(8):
 for c in range(4):
  addresses=[warp*128+c*32+lane for lane in range(32)]
  assert len(set(a%32 for a in addresses))==32
  for lane,a in enumerate(addresses):
   consumer_warp=warp&1;nt=warp>>1
   assert a==(nt*2+consumer_warp)*128+c*32+lane
(HERE/'candidate.cuh').write_text('// SPDX-License-Identifier: AGPL-3.0-only\n// Generated from production SHA256 '+sha+'; do not edit.\n'+candidate.split('\n',1)[1])
print('PASS: unique1024-score exchange, original fragment reload mapping, unchanged scaling/softmax/PV suffix')
