# SPDX-License-Identifier: AGPL-3.0-only
"""Native single-expert staged precision runner, shared by GPU diagnostics."""
import numpy as np


def validate_mma_profiles(mma_module, mma_profile, fat_mma_profile):
    if (mma_module is None) != (mma_profile is None):
        raise ValueError('MMA module and explicit profile must be supplied together')
    if (mma_module is None) != (fat_mma_profile is None):
        raise ValueError('production MMA requires explicit thin and fat profiles together')
    if fat_mma_profile is not None and tuple(fat_mma_profile) != (16, 128, 1, 0):
        raise ValueError('explicit production fat MMA profile16/128/1/0 required')
    if mma_profile is not None and tuple(mma_profile) != (32, 256, 8, 1):
        raise ValueError('explicit thin MMA profile32/256/8/1 required')

def run_expert(module, dx, parts, weight, thin_cap, limit, mma_module=None, mma_profile=None,
               fat_probe_module=None, fat_probe_profile=None, fat_mma_profile=None):
    if thin_cap <= 0:
        raise ValueError('explicit positive thin capacity required')
    validate_mma_profiles(mma_module, mma_profile, fat_mma_profile)
    if (fat_probe_module is None) != (fat_probe_profile is None):
        raise ValueError('experimental fat module and profile must be supplied together')
    if fat_probe_profile is not None and tuple(fat_probe_profile) not in ((16, 128, 1, 0), (32, 128, 1, 0)):
        raise ValueError('explicit sequential-K16 or interleaved-K32 fat experiment required')
    import cupy as cp
    import torch
    rows, hidden = dx.shape
    intermediate = parts['up']['svh'].numel()
    if dx.dtype != torch.bfloat16 or rows <= 0 or hidden % 128 or intermediate % 128:
        raise ValueError('nonempty BF16 rows with H128 geometry required')
    for name, k, n in (('gate', hidden, intermediate), ('up', hidden, intermediate), ('down', intermediate, hidden)):
        part = parts[name]
        if part['suh'].numel() != k or part['svh'].numel() != n or tuple(part['trellis'].shape[:2]) != (k // 16, n // 16):
            raise ValueError('expert descriptor and packed dimensions differ')
        if part['trellis'].shape[2] % 16 or not 1 <= part['trellis'].shape[2] // 16 <= 8:
            raise ValueError('integer EXL3 bits1..8 required')
    ids = cp.zeros(rows, cp.int32)
    count_buffer = cp.full(3, np.uint32(0xffffffff), cp.uint32)
    counts = count_buffer[1:2]
    weights = cp.full(rows, np.float32(weight), cp.float32)
    def descriptor(part):
        arrays = [cp.from_dlpack(part[key]) for key in ('trellis', 'suh', 'svh')]
        return cp.asarray(np.array([[a.data.ptr for a in arrays]
                         + [part['trellis'].shape[2] // 16]], np.uint64)), arrays
    tables, keepalive = {}, []
    for name, part in parts.items():
        tables[name], arrays = descriptor(part)
        keepalive.extend(arrays)
    functions = {name: module.get_function('exl3_moe_' + name) for name in
        ('route_counts', 'input_hadamard_f16', 'mul1_matmul_precision',
         'output_hadamard_precision', 'activate_down_hadamard',
         'down_weighted_output', 'combine_f32')}
    matmul = (functions['mul1_matmul_precision'] if mma_module is None else
              mma_module.get_function('exl3_moe_mul1_matmul_precision_mma'))
    extra = () if mma_profile is None else tuple(map(np.uint32, (*mma_profile, *fat_mma_profile)))
    if mma_profile is not None and (hidden % 256 or intermediate % 256):
        raise ValueError('MMA output dimensions must be N256 aligned')
    result = {}
    def guarded_yh(shape):
        storage = cp.full(int(np.prod(shape)) + 2, np.float32(12345), cp.float32)
        return storage, storage[1:-1].reshape(shape)
    def check_guard(storage):
        np.testing.assert_array_equal(cp.asnumpy(storage[[0, -1]]), [12345, 12345])
    def packed_matmul(name, xh, yh, n, k):
        if fat_probe_module is not None and rows > thin_cap:
            part = parts[name]
            fat_probe_module.get_function('exl3_mma_probe')(((n + 31) // 32, rows), (128,),
                (xh, cp.from_dlpack(part['trellis']), yh,
                 *map(np.uint32, (rows, n, k, part['trellis'].shape[2] // 16, *fat_probe_profile))))
        else:
            matmul(((n + 127) // 128, rows), (128,),
                (xh, ids, tables[name], yh, counts,
                 *map(np.uint32, (rows, n, k, 1, 1, thin_cap)), *extra))
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        mixed_ids = cp.asarray([0, 1, 0, -1, 2, 1, 2, 2, 9], cp.int32)
        mixed_guard = cp.full(5, np.uint32(0xffffffff), cp.uint32)
        functions['route_counts']((1,), (128,),
            (mixed_ids, mixed_guard[1:-1], np.uint32(9), np.uint32(3)))
        np.testing.assert_array_equal(cp.asnumpy(mixed_guard), [0xffffffff, 2, 2, 3, 0xffffffff])
        # Actual BF16 Torch addition qualifies the two rounding boundaries.
        boundary = torch.zeros((3, 2, 128), dtype=torch.float32, device=dx.device)
        boundary[0, 0].fill_(1.00390625)
        boundary[1, 0].fill_(-1.00390625)
        boundary[2, 0].fill_(.125)
        shared_control = torch.full((3, 128), .00390625, dtype=torch.bfloat16, device=dx.device)
        shared_control[1].neg_()
        guard_output = cp.full(3 * 128 + 2, np.uint16(0x42f6), cp.uint16)
        for has_shared in (0, 1):
            functions['combine_f32']((2,), (256,),
                (cp.from_dlpack(boundary), cp.from_dlpack(shared_control.view(torch.uint16)),
                 guard_output[1:-1], *map(np.uint32, (3, 128, 2, has_shared))))
            reference_control = boundary.sum(1).bfloat16()
            if has_shared:
                reference_control = reference_control + shared_control
            np.testing.assert_array_equal(cp.asnumpy(guard_output[1:-1]).reshape(3, 128), reference_control.view(torch.uint16).cpu().numpy())
            np.testing.assert_array_equal(cp.asnumpy(guard_output)[[0, -1]], [0x42f6, 0x42f6])
        wrong_control = (boundary.sum(1) + shared_control.float()).bfloat16()
        if torch.equal(wrong_control, reference_control):
            raise AssertionError('BF16-before-shared negative control failed to detect old policy')
        result['combine_boundary_input'] = boundary.cpu().numpy()
        result['combine_boundary_shared'] = shared_control.float().cpu().numpy()
        result['combine_boundary_reference'] = reference_control.float().cpu().numpy()
        result['combine_boundary_native'] = (cp.asnumpy(guard_output[1:-1]).astype(np.uint32) << 16).view(np.float32).reshape(3, 128)
        functions['route_counts']((1,), (128,),
            (ids, counts, np.uint32(rows), np.uint32(1)))
        cinput = cp.from_dlpack(dx.view(torch.uint16))
        outputs = {}
        for name in ('gate', 'up'):
            xh = cp.empty((rows, hidden), cp.float32)
            yh_storage, yh = guarded_yh((rows, intermediate))
            output = cp.empty((rows, intermediate), cp.float32)
            functions['input_hadamard_f16']((hidden // 128, rows), (128,),
                (cinput, ids, tables[name], xh, *map(np.uint32, (rows, hidden, 1, 0, 1))))
            packed_matmul(name, xh, yh, intermediate, hidden)
            functions['output_hadamard_precision']((intermediate // 128, rows), (128,),
                (yh, ids, tables[name], output, counts, *map(np.uint32, (rows, intermediate, 1, 1, thin_cap))))
            outputs[name] = output
            check_guard(yh_storage)
            result[name + '_xh'], result[name + '_yh'] = cp.asnumpy(xh), cp.asnumpy(yh)
            result[name + '_output'] = cp.asnumpy(output)
        # Exercise the exact alias used by production to reuse gate workspace.
        down_xh = outputs['gate']
        functions['activate_down_hadamard']((intermediate // 128, rows), (128,),
            (outputs['gate'], outputs['up'], ids, tables['down'], down_xh, counts,
             *map(np.uint32, (rows, intermediate, 1, 1, thin_cap)), np.float32(limit), np.uint32(1)))
        down_storage, down_yh = guarded_yh((rows, hidden))
        weighted = cp.empty((rows, hidden), cp.float32)
        packed_matmul('down', down_xh, down_yh, hidden, intermediate)
        functions['down_weighted_output']((hidden // 128, rows), (128,),
            (down_yh, ids, tables['down'], weights, weighted, counts,
             *map(np.uint32, (rows, hidden, 1, 1, thin_cap))))
        combined = cp.empty((rows, hidden), cp.uint16)
        functions['combine_f32'](((rows * hidden + 255) // 256,), (256,),
            (weighted, np.uint64(0), combined, *map(np.uint32, (rows, hidden, 1, 0))))
        result['down_xh'], result['down_yh'] = cp.asnumpy(down_xh), cp.asnumpy(down_yh)
        result['weighted_f32'] = cp.asnumpy(weighted)
        check_guard(down_storage)
        result['weighted_bf16'] = (cp.asnumpy(combined).astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(cp.asnumpy(counts), [rows])
        np.testing.assert_array_equal(cp.asnumpy(count_buffer)[[0, -1]], [0xffffffff, 0xffffffff])
    return result
