# SPDX-License-Identifier: AGPL-3.0-only
"""Real installed LinearEXL3 vs native packed projection, strict component gate.

Run in the exact reference image; this never qualifies whole-model parity.
All profiles and captured inputs are explicit; raw stage spans are saved before
guard and numeric checks, including failed component comparisons.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import re
from pathlib import Path
import sys
import numpy as np
from differential_expert import digest, validate_input, metrics
from linear_launch import launch_key, cache_record, integer_constant
from native_ptx import validate_binding, load_module


def bit_comparison(reference, candidate):
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.dtype != np.float32 or b.dtype != np.float32:
        raise ValueError('comparison requires exact F32 transport arrays')
    result = metrics(a, b, 0., 0.)
    result['bit_mismatches'] = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
    result['byte_exact'] = result['bit_mismatches'] == 0
    return result


def reference_rows(rows, minimum, pad):
    if not 1 <= rows <= 128 or not 1 <= minimum <= 128 or type(pad) is not bool:
        raise ValueError('explicit rows/minimum1..128 and padding flag required')
    return max(rows, minimum) if pad else rows


def guarded_payload(raw, sentinel, shape, guard_width):
    flat = np.asarray(raw).reshape(-1)
    if (type(guard_width) is not int or guard_width<=0 or
        len(flat) != int(np.prod(shape)) + 2*guard_width or
        np.any(flat[:guard_width] != sentinel) or np.any(flat[-guard_width:] != sentinel)):
        raise ValueError('native output guard corrupted or span length differs')
    return flat[guard_width:-guard_width].reshape(shape).copy()


def require_aligned_pointers(pointers):
    if not pointers or any(pointer<=0 or pointer%128 for pointer in pointers):
        raise ValueError('packed extension and native spans require 128-byte-aligned pointers')


def validate_mul1_marker(values, expected_unsigned):
    marker = np.asarray(values)
    if marker.dtype != np.dtype('<i4') or marker.size != 1:
        raise ValueError('EXL3 mul1 format marker must be an I32 scalar')
    if (int(marker.item()) & 0xffffffff) != expected_unsigned:
        raise ValueError('EXL3 mul1 format marker differs from native codebook multiplier')


def output_component_bounds(segments, index, rank):
    if not segments or any(type(width) is not int or width<=0 or width%256 for width in segments):
        raise ValueError('global output segments must each permit H128-aligned TP2 shards')
    if type(index) is not int or not 0<=index<len(segments) or rank not in (0,1):
        raise ValueError('explicit output component and TP2 rank required')
    local=segments[index]//2
    return sum(segments[:index])+rank*local,local


def fused_reference_layout(segments,index,local_n):
    output_component_bounds(segments,index,0)
    if len(segments)!=3 or len(set(segments))!=1 or local_n!=segments[index]//2:
        raise ValueError('reference operator requires equal QKV thirds and exact selected local width')
    return sum(segments)//2,sum(segments[:index])//2


def selected_geometry(tag, sources):
    if tag == 90:
        return {'path':'qtip_gemv','geometry':'not exposed by extension return value'}
    pattern = rf'^#define EXL3_GEMM_SHAPE_{tag}\s+([\d,\s]+)$'
    matches = [(path, re.search(pattern, text, re.MULTILINE)) for path,text in sources.items()]
    matches = [(path, match) for path,match in matches if match]
    if len(matches) != 1:
        raise ValueError('returned cooperative tag requires one authoritative kernel map')
    path, match = matches[0]
    values = [int(value) for value in match.group(1).replace(',',' ').split()]
    if len(values) != 5:
        raise ValueError('unexpected installed cooperative shape descriptor')
    return {'path':'cooperative_autotuner','source':path,
            **dict(zip(('tile_m','tile_k','tile_n','shared_stages','fragment_stages'),values)),
            'grid_binding':'selected grid is bound separately by the cooperative cache record'}


def load_parts(checkpoint, prefix, axis, rank, plugin, expected_marker, segments, component):
    from safetensors import safe_open
    index_path = checkpoint / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())['weight_map']
    parts, binding, global_parts = {}, {}, {}
    for suffix in ('trellis', 'suh', 'svh', 'mul1'):
        key = prefix + '.' + suffix
        shard = index[key]
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError('checkpoint shard must be a simple filename')
        path = checkpoint / shard
        with safe_open(path, framework='pt', device='cpu') as source:
            full = source.get_tensor(key)
        global_parts[suffix]=full
        binding[key] = {'sha256': hashlib.sha256(full.numpy().tobytes()).hexdigest(),
                        'shape': list(full.shape), 'dtype': str(full.dtype), 'shard': shard}
        if suffix == 'mul1':
            local = full.reshape(-1)[:1]
            validate_mul1_marker(full.numpy(), expected_marker)
        else:
            if axis=='output-component' and suffix in ('trellis','svh'):
                offset,width=output_component_bounds(segments,component,rank)
                dim,divisor=(1,16) if suffix=='trellis' else (0,1)
                if full.shape[dim]*divisor!=sum(segments):
                    raise ValueError('declared output segments differ from actual checkpoint shape')
                local=full.narrow(dim,offset//divisor,width//divisor)
            elif axis in ('replicated','output-component'):
                local=full
            else:
                shard_fn = plugin.shard_exl3_col if axis == 'output' else plugin.shard_exl3_row
                local = shard_fn(full, suffix, rank, 2)
        parts[suffix] = local.contiguous().cuda()
    reference_parts=parts
    if axis=='output-component':
        if len(segments)!=3 or len(set(segments))!=1:
            raise ValueError('installed GLM fused QKV helper requires three equal global segments')
        reference_parts=plugin.glm53_qkv_rank_slice(global_parts,rank,2,parts['suh'].device)
        reference_parts['mul1']=reference_parts.pop('marker')
    return parts, reference_parts, binding, index_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--axis', choices=('output', 'input','replicated','output-component'), required=True)
    parser.add_argument('--output-segments',type=int,nargs='+',
                        help='explicit global fused Q/K/V output widths in checkpoint order')
    parser.add_argument('--component-index',type=int)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--input', type=Path, required=True,
                        help='captured BF16-exact F32 npy [rows, localK]')
    parser.add_argument('--input-binding', type=Path, required=True,
                        help='capture adapter provenance JSON; must bind npy SHA')
    parser.add_argument('--reference-plugin', type=Path, required=True)
    parser.add_argument('--reference-plugin-sha256', required=True)
    parser.add_argument('--reference-kernel-source', type=Path, action='append', required=True)
    parser.add_argument('--reference-tune-cache', type=Path, required=True,
                        help='explicit private copied reference cache file; never modify original')
    parser.add_argument('--reference-min-rows', type=int, required=True)
    parser.add_argument('--reference-pad', choices=('true', 'false'), required=True)
    parser.add_argument('--native-common', type=Path, required=True)
    parser.add_argument('--native-ptx-dir',type=Path)
    parser.add_argument('--native-ptx-binding',type=Path)
    parser.add_argument('--experimental-mma-profile',type=int,nargs=4)
    parser.add_argument('--experimental-mma-source',type=Path)
    parser.add_argument('--experimental-mma-production',action='store_true',
                        help='execute scoped production13arg export, preserving scalar baseline gate')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if (args.experimental_mma_profile is None)!=(args.experimental_mma_source is None):
        raise ValueError('experimental MMA source and explicit profile must be supplied together')
    if args.experimental_mma_production and args.experimental_mma_profile is None:
        raise ValueError('production MMA replay requires explicit source/profile')
    if (args.native_ptx_dir is None)!=(args.native_ptx_binding is None):
        raise ValueError('native PTX directory and executable/source binding required together')
    ptx_binding=None
    if args.native_ptx_dir is not None:
        if not args.experimental_mma_production:
            raise ValueError('embedded PTX replay requires explicit production13 profile')
        ptx_binding=validate_binding(args.native_ptx_dir,args.native_ptx_binding,args.native_common)
        if digest(args.experimental_mma_source)!=ptx_binding['modules']['exl3_linear_mma']['sources']['exl3_linear_mma.cu']:
            raise ValueError('explicit MMA source differs from embedded PTX binding')
    if args.axis=='output-component':
        output_component_bounds(args.output_segments,args.component_index,args.rank)
    elif args.output_segments is not None or args.component_index is not None:
        raise ValueError('segmented parameters are only valid for output-component mode')
    x = validate_input(np.load(args.input, allow_pickle=False))
    input_binding = json.loads(args.input_binding.read_text())
    if input_binding.get('npy_sha256') != digest(args.input):
        raise ValueError('captured input npy differs from adapter binding')
    padded_rows = reference_rows(len(x), args.reference_min_rows, args.reference_pad == 'true')
    if digest(args.reference_plugin) != args.reference_plugin_sha256:
        raise ValueError('installed reference plugin source changed')
    source_hashes = {str(p): digest(p) for p in args.reference_kernel_source}
    source_text = {str(p):p.read_text() for p in args.reference_kernel_source}
    autotune_version = integer_constant(source_text,'COOP_AUTOTUNE_VERSION')
    blackwell_tag = integer_constant(source_text,'CC_BLACKWELL')
    if not args.reference_tune_cache.is_absolute() or args.reference_tune_cache.is_dir():
        raise ValueError('private reference tune cache must be an absolute file path')
    cache_before = digest(args.reference_tune_cache) if args.reference_tune_cache.exists() else None
    os.environ['EXLLAMAV3_TUNE_CACHE'] = str(args.reference_tune_cache)
    native_codebook = (args.native_common/'exl3_common.cuh').read_text()
    multiplier = re.search(r'product\s*=\s*state\s*\*\s*(0x[\da-fA-F]+)u',native_codebook)
    if multiplier is None:
        raise ValueError('native authoritative mul1 multiplier is missing')
    expected_marker = int(multiplier.group(1),16)
    import torch
    import cupy as cp
    import exllamav3_ext
    if not torch.cuda.is_available() or cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('exactly one visible real GPU required')
    spec = importlib.util.spec_from_file_location('exl3_linear_reference', args.reference_plugin)
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)
    if (plugin.MUL1_MARKER_SIGNED_INT32 & 0xffffffff) != expected_marker:
        raise ValueError('installed signed marker and native codebook multiplier differ')
    parts, reference_parts, tensors, index_path = load_parts(args.checkpoint, args.prefix, args.axis, args.rank, plugin,
                                                           expected_marker,args.output_segments,args.component_index)
    k, n = parts['suh'].numel(), parts['svh'].numel()
    reference_n=reference_parts['svh'].numel()
    component_offset=0
    if args.axis=='output-component':
        expected_reference_n,component_offset=fused_reference_layout(args.output_segments,args.component_index,n)
        if reference_n!=expected_reference_n:
            raise ValueError('installed fused QKV helper returned unexpected reference operator width')
    bits = parts['trellis'].shape[-1] // 16
    if x.shape[1] != k or k % 128 or n % 128 or not 1 <= bits <= 8:
        raise ValueError('captured input and packed H128 projection geometry differ')
    if args.prefix == 'lm_head' and args.reference_pad != 'false':
        raise ValueError('attested reference head explicitly does not pad')
    args.out.mkdir(parents=True, exist_ok=False)
    rows = len(x)
    dx = torch.from_numpy(x).cuda().bfloat16()
    xp = torch.nn.functional.pad(dx, (0, 0, 0, padded_rows - rows))
    # Installed packed loader creates default F16 handles and overrides forward
    # output to F32; external Glm53PackedLinearMethod casts that output to BF16.
    linear = plugin.make_linear_exl3(reference_parts['trellis'], reference_parts['suh'], reference_parts['svh'], reference_parts['mul1'],
                                     out_dtype=torch.float16)
    reference_full_f32 = linear.forward(xp.contiguous().half(), {}, out_dtype=torch.float32)[:rows]
    reference_f32=reference_full_f32[:,component_offset:component_offset+n]
    reference_bf16 = reference_f32.bfloat16().float()
    # Same unforced extension call as BC_LinearEXL3, rather than guessing a
    # shape from K/N. The returned tag identifies the actual autotuned launch.
    reference_input = xp.contiguous().half()
    direct_storage = torch.full((padded_rows*reference_n+64,), 713.5, dtype=torch.float32, device='cuda')
    direct_output = direct_storage[32:-32].view(padded_rows,reference_n)
    direct_hadamard = torch.empty_like(reference_input)
    require_aligned_pointers([tensor.data_ptr() for tensor in
                             (reference_input,direct_output,direct_hadamard,*reference_parts.values())])
    selected_tag = exllamav3_ext.exl3_gemm(reference_input, reference_parts['trellis'], direct_output,
        reference_parts['suh'], direct_hadamard, reference_parts['svh'], -1, False, True, 0)
    torch.cuda.synchronize()
    arrays = {'input': x, 'reference_f32_before_external_bf16': reference_f32.cpu().numpy(),
              'reference_bf16_transport': reference_bf16.cpu().numpy(),
              'reference_full_operator_f32':reference_full_f32.cpu().numpy(),
              'reference_direct_full_f32':direct_output[:rows].cpu().numpy(),
              'reference_direct_f32': direct_output[:rows,component_offset:component_offset+n].cpu().numpy(),
              'reference_direct_hadamard_f16': direct_hadamard.cpu().numpy()}
    np.savez(args.out/'reference-stages.npz',**arrays,
             guarded_direct_storage=direct_storage.cpu().numpy())
    guarded_payload(direct_storage.cpu().numpy(), np.float32(713.5), (padded_rows,reference_n),32)
    cache_binding={'path':str(args.reference_tune_cache),'before_sha256':cache_before,
                   'autotune_version':autotune_version,'architecture_enum':blackwell_tag}
    if args.reference_tune_cache.exists():
        payload=args.reference_tune_cache.read_bytes()
        (args.out/'reference-tune-cache-after.bin').write_bytes(payload)
        cache_binding['after_sha256']=hashlib.sha256(payload).hexdigest()
    if selected_tag!=90:
        if not args.reference_tune_cache.exists():
            raise ValueError('cooperative selection did not persist its required launch record')
        props=torch.cuda.get_device_properties(0)
        if props.major<10:
            raise ValueError('this component oracle requires attested Blackwell hardware')
        key=launch_key(padded_rows,k,reference_n,bits,0,blackwell_tag,props.multi_processor_count,autotune_version)
        cache_binding['actual_launch']=cache_record(payload,key,int(selected_tag))
    common = args.native_common
    options = ('--std=c++17', '--fmad=false', '-I' + str(common))
    native = load_module(cp,common,'exl3',args.native_ptx_dir,ptx_binding)
    residual = load_module(cp,common,'residual_add',args.native_ptx_dir,ptx_binding)
    funcs = {name: native.get_function('exl3_' + name) for name in
             ('input_hadamard_f16', 'input_hadamard', 'mul1_matmul',
              'output_hadamard', 'output_hadamard_f32')}
    funcs['bf16_to_f32']=residual.get_function('bf16_to_f32')
    guard = np.float32(713.5)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        cdx = cp.from_dlpack(dx.view(torch.uint16))
        suh, svh, trellis = (cp.from_dlpack(parts[name]) for name in ('suh', 'svh', 'trellis'))
        xh_storage = cp.full(rows*k+64, guard, cp.float32)
        yh_storage = cp.full(rows*n+64, guard, cp.float32)
        out_storage = cp.full(rows*n+128, np.uint16(0x7bcd), cp.uint16)
        transport_storage = cp.full(rows*n+64, guard, cp.float32)
        f32_storage = cp.full(rows*n+64, guard, cp.float32)
        xh, yh, out = xh_storage[32:-32], yh_storage[32:-32], out_storage[64:-64]
        require_aligned_pointers([tensor.data.ptr for tensor in
            (cdx,suh,svh,trellis,xh,yh,out,transport_storage[32:-32],f32_storage[32:-32])])
        funcs['input_hadamard_f16']((k//128, rows), (128,),
            (cdx, suh, xh, np.uint32(rows), np.uint32(k)))
        funcs['mul1_matmul']((n//128, rows), (128,),
            (xh, trellis, yh, *map(np.uint32, (rows, n, k, bits))))
        funcs['output_hadamard']((n//128, rows), (128,),
            (yh, svh, out, np.uint32(rows), np.uint32(n)))
        funcs['bf16_to_f32'](((rows*n+255)//256,), (256,),
            (out, transport_storage[32:-32], np.uint32(rows*n)))
        funcs['output_hadamard_f32']((n//128, rows), (128,),
            (yh, svh, f32_storage[32:-32], np.uint32(rows), np.uint32(n)))
        raw = {name: cp.asnumpy(storage) for name,storage in
               (('xh',xh_storage),('yh',yh_storage),('bf16',out_storage),
                ('transport',transport_storage),('f32',f32_storage))}
        np.savez(args.out/'native-guarded-storage.npz',**raw)
        arrays['native_xh_f32_storage'] = guarded_payload(raw['xh'], guard, (rows,k),32)
        arrays['native_yh_f32_storage'] = guarded_payload(raw['yh'], guard, (rows,n),32)
        native_bf16 = guarded_payload(raw['bf16'], np.uint16(0x7bcd), (rows,n),64)
        arrays['native_bf16_transport'] = (native_bf16.astype(np.uint32)<<16).view(np.float32)
        arrays['native_f32_transport'] = guarded_payload(raw['transport'], guard, (rows,n),32)
        arrays['native_f32_before_bf16_diagnostic'] = guarded_payload(raw['f32'], guard, (rows,n),32)
        if args.experimental_mma_profile is not None:
            if args.axis=='output-component' and not args.experimental_mma_production:
                raise ValueError('ordinary experimental probe cannot model fused logical partitions')
            from linear_probe import replay
            experimental=replay(cp,common,args.experimental_mma_source,funcs,trellis,svh,
                {'native_xh':arrays['native_xh_f32_storage'],
                 'reference_xh':arrays['reference_direct_hadamard_f16'][:rows].astype(np.float32)},
                args.experimental_mma_profile,guarded_payload,args.out,
                args.experimental_mma_production,reference_n,component_offset,
                args.native_ptx_dir,ptx_binding)
            arrays.update({'experimental_'+name:value for name,value in experimental.items()})
    np.savez(args.out/'stages.npz',**arrays)
    report = {'scope':'Actual packed linear component gate; not whole-model parity',
        'prefix':args.prefix,'axis':args.axis,'rank':args.rank,'tp':2,
        'output_components':{'global_widths':args.output_segments,'selected_index':args.component_index},
        'shape':{'rows':rows,'k':k,'n':n,'bits':bits},
        'input':{'npy_sha256':digest(args.input),'binding_sha256':digest(args.input_binding),
                 'binding':input_binding},
        'reference_profile':{'min_rows':args.reference_min_rows,'pad':args.reference_pad == 'true',
            'actual_padded_rows':padded_rows,'handle_default_output':'float16',
            'operator_n':reference_n,'selected_component_offset':component_offset,
            'forward_output':'float32','external_output':'bfloat16','reconstruct':False,
            'unforced_extension_selection':{'returned_tag':int(selected_tag),
                'force_shape_idx':-1,'mcg':False,'mul1':True,'force_num_sms':0,
                'selected_geometry':selected_geometry(int(selected_tag),
                    source_text),
                'cooperative_cache_binding':cache_binding,
                'bc_class':type(linear.bc).__name__,
                'handle':{'in_features':linear.in_features,'out_features':linear.out_features,
                          'K':linear.K,'mcg':linear.mcg,'mul1':linear.mul1,
                          'bias_present':linear.bias is not None}}},
        'native_compilation':({'compiler':'image-emitted NVCC PTX via CuPy driver load',
            'binding_sha256':digest(args.native_ptx_binding),'binding':ptx_binding,
            'qualification_limit':'Final-executable embedded PTX component replay; not full-model parity'}
            if ptx_binding is not None else {'compiler':'CuPy NVRTC','options':list(options),
            'qualification_limit':'Production CUDA source replay, not shipped NVCC PTX replay'}),
        'sources':{'plugin_sha256':digest(args.reference_plugin),'extension_sha256':digest(exllamav3_ext.__file__),
            'linear_class_sha256':digest(inspect.getfile(type(linear))),
            'installed_kernel_sha256':source_hashes,'native_sha256':{p.name:digest(p) for p in
                (common/'exl3.cu',common/'exl3_common.cuh',common/'residual_add.cu')},
            'harness_sha256':digest(__file__),'checkpoint_index_sha256':digest(index_path),
            'selected_tensors':tensors},
        'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),
            'multi_processor_count':torch.cuda.get_device_properties(0).multi_processor_count,
            'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction},
        'comparisons':{},'guard_checks':'all native stage spans preserve prefix/suffix sentinels'}
    try:
        for name,ref,candidate in (
            ('reference_full_direct_replay','reference_full_operator_f32','reference_direct_full_f32'),
            ('reference_direct_replay','reference_f32_before_external_bf16','reference_direct_f32'),
            ('bf16_producer','reference_bf16_transport','native_bf16_transport'),
            ('f32_transport','reference_bf16_transport','native_f32_transport'),
            ('f32_internal_diagnostic','reference_f32_before_external_bf16','native_f32_before_bf16_diagnostic')):
            report['comparisons'][name] = bit_comparison(arrays[ref],arrays[candidate])
        control = bit_comparison(arrays['reference_bf16_transport'],np.roll(arrays['reference_bf16_transport'],1,axis=1))
        if control['byte_exact']:
            raise ValueError('shuffled vocabulary/channel negative control is uninformative')
        report['negative_control_shuffled_channels'] = control
        report['pass'] = all(report['comparisons'][name]['byte_exact'] for name in
                            ('reference_full_direct_replay','reference_direct_replay','bf16_producer','f32_transport'))
    except ValueError as error:
        report['pass'] = False
        report['numeric_failure'] = str(error)
    if args.experimental_mma_profile is not None:
        report['experimental_mma']={'profile':args.experimental_mma_profile,
            'production_13arg_export':args.experimental_mma_production,
            'logical_n':reference_n,'component_offset':component_offset,
            'source_sha256':digest(args.experimental_mma_source),
            'helper_sha256':digest(common/'exl3_mma.cuh'),
            'driver_sha256':digest(Path(__file__).with_name('linear_probe.py')),
            'qualification_limit':'Experimental component diagnostic; production scalar result unchanged',
            'comparisons':{name:bit_comparison(arrays['reference_bf16_transport'],arrays['experimental_'+name+'_bf16'])
                           for name in ('native_xh','reference_xh')},
            'transport_comparisons':{name:bit_comparison(arrays['reference_bf16_transport'],arrays['experimental_'+name+'_transport'])
                for name in ('native_xh','reference_xh')}}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['comparisons'],indent=2))
    return 0 if report['pass'] else 1

if __name__=='__main__':sys.exit(main())
