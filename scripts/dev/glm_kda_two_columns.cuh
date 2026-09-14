// SPDX-License-Identifier: AGPL-3.0-only
// Standalone candidate: same FP32 recurrence/reduction per column, shared Q/K/D loads.
extern "C" __global__ void __launch_bounds__(128, 4)
atlas_dev_kda_two_columns(
    const __nv_bfloat16* qkv, const float* q_norm, const float* k_norm,
    const float* decay, const float* beta, float* state, __nv_bfloat16* output,
    unsigned tokens, unsigned heads, unsigned dim) {
    const unsigned head=blockIdx.x, warp=threadIdx.x>>5, lane=threadIdx.x&31;
    const unsigned col=blockIdx.y*8+warp*2;
    if(head>=heads || col+1>=dim || dim!=128)return;
    const unsigned r0=lane,r1=lane+32,r2=lane+64,r3=lane+96;
    float* H=state+(unsigned long long)head*dim*dim;
    float s0=H[(unsigned long long)r0*dim+col],s1=H[(unsigned long long)r1*dim+col];
    float s2=H[(unsigned long long)r2*dim+col],s3=H[(unsigned long long)r3*dim+col];
    float t0=H[(unsigned long long)r0*dim+col+1],t1=H[(unsigned long long)r1*dim+col+1];
    float t2=H[(unsigned long long)r2*dim+col+1],t3=H[(unsigned long long)r3*dim+col+1];
    for(unsigned pos=0;pos<tokens;++pos) {
        const unsigned long long base=(unsigned long long)pos*3*heads*dim;
        const unsigned long long off=((unsigned long long)pos*heads+head)*dim;
        const float q0=q_norm[off+r0],q1=q_norm[off+r1],q2=q_norm[off+r2],q3=q_norm[off+r3];
        const float k0=k_norm[off+r0],k1=k_norm[off+r1],k2=k_norm[off+r2],k3=k_norm[off+r3];
        const float d0=decay[off+r0],d1=decay[off+r1],d2=decay[off+r2],d3=decay[off+r3];
        float dk=s0*d0*k0+s1*d1*k1+s2*d2*k2+s3*d3*k3;
        float ek=t0*d0*k0+t1*d1*k1+t2*d2*k2+t3*d3*k3;
        #pragma unroll
        for(int shift=16;shift>0;shift>>=1) {
            dk+=__shfl_xor_sync(0xffffffffu,dk,shift);
            ek+=__shfl_xor_sync(0xffffffffu,ek,shift);
        }
        const unsigned long long voff=base+(unsigned long long)2*heads*dim+(unsigned long long)head*dim+col;
        const float b=beta[(unsigned long long)pos*heads+head];
        const float delta=((float)qkv[voff]-dk)*b;
        const float extra=((float)qkv[voff+1]-ek)*b;
        s0=s0*d0+delta*k0;s1=s1*d1+delta*k1;s2=s2*d2+delta*k2;s3=s3*d3+delta*k3;
        t0=t0*d0+extra*k0;t1=t1*d1+extra*k1;t2=t2*d2+extra*k2;t3=t3*d3+extra*k3;
        float dq=s0*q0+s1*q1+s2*q2+s3*q3;
        float eq=t0*q0+t1*q1+t2*q2+t3*q3;
        #pragma unroll
        for(int shift=16;shift>0;shift>>=1) {
            dq+=__shfl_xor_sync(0xffffffffu,dq,shift);
            eq+=__shfl_xor_sync(0xffffffffu,eq,shift);
        }
        if(lane==0) {
            output[off+col]=__float2bfloat16(dq);
            output[off+col+1]=__float2bfloat16(eq);
        }
    }
    H[(unsigned long long)r0*dim+col]=s0;H[(unsigned long long)r1*dim+col]=s1;
    H[(unsigned long long)r2*dim+col]=s2;H[(unsigned long long)r3*dim+col]=s3;
    H[(unsigned long long)r0*dim+col+1]=t0;H[(unsigned long long)r1*dim+col+1]=t1;
    H[(unsigned long long)r2*dim+col+1]=t2;H[(unsigned long long)r3*dim+col+1]=t3;
}
