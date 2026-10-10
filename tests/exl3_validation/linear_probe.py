# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit experimental MMA replay; never substitutes the production gate."""
import numpy as np
from differential_expert import digest
from native_ptx import load_module


def validate_profile(profile,k,n):
    if (len(profile)!=4 or tuple(profile[:2]) not in ((16,128),(32,128),(32,256),(16,512))
        or profile[2]<=0 or profile[3] not in (0,1) or k%profile[0] or n%profile[1]):
        raise ValueError('explicit compatible experimental MMA geometry required')


def replay(cp,common,source,funcs,trellis,svh,inputs,profile,guarded_payload,out_dir,
           production,logical_n,logical_offset,ptx_dir=None,ptx_binding=None):
    rows,k=inputs['native_xh'].shape
    n=svh.size
    if rows*n>np.iinfo(np.uint32).max:raise ValueError('F32 transport element count exceeds U32 ABI')
    validate_profile(profile,k,logical_n)
    if logical_offset%128 or logical_offset+n>logical_n:
        raise ValueError('experimental physical component exceeds logical reference geometry')
    if production and (profile[3]!=0 or not 1<=profile[2]<=48):
        raise ValueError('production generic profile requires unrounded 1..48 slices')
    if ptx_dir is not None:
        if not production:raise ValueError('embedded PTX requires production13 export')
        module=load_module(cp,common,'exl3_linear_mma',ptx_dir,ptx_binding)
    else:
        module=cp.RawModule(code=source.read_text()+'\n// '+digest(common/'exl3_mma.cuh'),
            options=('--std=c++17','--fmad=false','-I'+str(common)))
    mma=module.get_function('exl3_mul1_matmul_mma' if production else 'exl3_mma_probe')
    result={}
    transport_kernel=funcs['bf16_to_f32']
    for label,xh in inputs.items():
        xh=cp.asarray(xh,dtype=cp.float32)
        yh_storage=cp.full(rows*n+64,np.float32(713.5),cp.float32)
        bf_storage=cp.full(rows*n+128,np.uint16(0x7bcd),cp.uint16)
        f32_storage=cp.full(rows*n+64,np.float32(713.5),cp.float32)
        transport_storage=cp.full(rows*n+64,np.float32(713.5),cp.float32)
        yh,bf,f32=yh_storage[32:-32],bf_storage[64:-64],f32_storage[32:-32]
        arguments=(xh,trellis,yh,*map(np.uint32,(rows,n,k)),np.uint32(trellis.shape[-1]//16))
        if production:
            arguments+=tuple(map(np.uint32,(logical_n,logical_offset)))
        mma((n//(128 if production else 32),rows),(128,),arguments+tuple(map(np.uint32,profile)))
        funcs['output_hadamard']((n//128,rows),(128,),
            (yh,svh,bf,np.uint32(rows),np.uint32(n)))
        funcs['output_hadamard_f32']((n//128,rows),(128,),
            (yh,svh,f32,np.uint32(rows),np.uint32(n)))
        transport_kernel(((rows*n+255)//256,),(256,),
            (bf,transport_storage[32:-32],np.uint32(rows*n)))
        raw={'transport':cp.asnumpy(transport_storage),'yh':cp.asnumpy(yh_storage),'bf16':cp.asnumpy(bf_storage),'f32':cp.asnumpy(f32_storage)}
        np.savez(out_dir/f'experimental-{label}-guarded.npz',**raw)
        result[label+'_yh']=guarded_payload(raw['yh'],np.float32(713.5),(rows,n),32)
        bf16=guarded_payload(raw['bf16'],np.uint16(0x7bcd),(rows,n),64)
        result[label+'_bf16']=(bf16.astype(np.uint32)<<16).view(np.float32)
        result[label+'_transport']=guarded_payload(raw['transport'],np.float32(713.5),(rows,n),32)
        result[label+'_f32']=guarded_payload(raw['f32'],np.float32(713.5),(rows,n),32)
    return result


def transport_comparison(reference,transport):
    # Preserve raw BF16 producer identity through the actual GPU F32 conversion.
    from differential_linear import bit_comparison
    return bit_comparison(reference,transport)
