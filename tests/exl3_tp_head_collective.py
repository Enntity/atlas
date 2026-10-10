# SPDX-License-Identifier: AGPL-3.0-only
"""Run on both TP ranks: actual NCCL byte gather and native F32 head reorder."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import hashlib
import ctypes
from pathlib import Path
import socket

import cupy as cp
import numpy as np
import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--master-addr', required=True)
    parser.add_argument('--master-port', type=int, required=True)
    parser.add_argument('--timeout', type=float, required=True)
    parser.add_argument('--expected-nccl-version')
    parser.add_argument('--expected-nccl-sha256')
    args = parser.parse_args()
    if not 0 < args.master_port < 65536 or args.timeout <= 0:
        parser.error('valid port and positive timeout required')
    if not torch.cuda.is_available():
        raise RuntimeError('actual CUDA device required')
    rank, world, rows, local_vocab, active_vocab = args.rank, 2, 3, 128, 250
    loaded_nccl = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                          if 'libnccl.so' in line})
    runtime_nccl = None
    if len(loaded_nccl) == 1:
        value = ctypes.c_int()
        library = ctypes.CDLL(loaded_nccl[0])
        if library.ncclGetVersion(ctypes.byref(value)) != 0:
            raise RuntimeError('ncclGetVersion failed')
        runtime_nccl = (value.value // 10000, value.value // 100 % 100, value.value % 100)
    if args.expected_nccl_version is not None or args.expected_nccl_sha256 is not None:
        if args.expected_nccl_version is None or args.expected_nccl_sha256 is None:
            parser.error('expected NCCL version and hash must be supplied together')
        actual = '.'.join(map(str, runtime_nccl)) if runtime_nccl is not None else 'ambiguous'
        if actual != args.expected_nccl_version or len(loaded_nccl) != 1:
            raise RuntimeError(f'NCCL identity mismatch: version={actual}, libraries={loaded_nccl}')
        digest = hashlib.sha256()
        with open(loaded_nccl[0], 'rb') as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != args.expected_nccl_sha256:
            raise RuntimeError('Loaded NCCL library SHA256 differs from candidate image')
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', rank=rank, world_size=world,
                            init_method=f'tcp://{args.master_addr}:{args.master_port}',
                            timeout=timedelta(seconds=args.timeout))
    try:
        values = (np.arange(local_vocab, dtype=np.float32)[None, :]
                  + np.arange(rows, dtype=np.float32)[:, None] * 1000
                  + rank * local_vocab)
        source = torch.from_numpy(values).cuda().contiguous().view(torch.uint8).reshape(-1)
        gathered = torch.empty(world * source.numel(), dtype=torch.uint8, device='cuda')
        # Atlas collectives preserve bytes and produce rank-major blocks.
        dist.all_gather_into_tensor(gathered, source)
        torch.cuda.synchronize()
        logits = cp.from_dlpack(gathered.view(torch.float32))
        output = cp.full(rows * active_vocab + 4, -999, dtype=cp.float32)
        root = Path(__file__).resolve().parents[1] / 'kernels/gb10/common'
        module = cp.RawModule(code=(root / 'exl3.cu').read_text(),
                              options=('--std=c++17', '-I' + str(root)))
        reorder = module.get_function('exl3_vocab_gather_reorder')
        reorder(((rows * active_vocab + 255) // 256,), (256,),
                (logits, output, np.uint32(rows), np.uint32(local_vocab), np.uint32(active_vocab)))
        expected = (np.arange(active_vocab, dtype=np.float32)[None, :]
                    + np.arange(rows, dtype=np.float32)[:, None] * 1000)
        np.testing.assert_array_equal(cp.asnumpy(output[:rows * active_vocab]).reshape(rows, active_vocab), expected)
        np.testing.assert_array_equal(cp.asnumpy(output[rows * active_vocab:]), np.full(4, -999, dtype=np.float32))
        print(json.dumps({'passed': True, 'rank': rank, 'world': world,
                          'hostname': socket.gethostname(), 'device': torch.cuda.get_device_name(0),
                          'rows': rows, 'local_vocab': local_vocab, 'active_vocab': active_vocab,
                          'backend': 'NCCL', 'transport_dtype': 'uint8', 'logit_dtype': 'f32',
                          'cuda': torch.version.cuda, 'nccl': runtime_nccl,
                          'torch_compiled_nccl': torch.cuda.nccl.version(),
                          'loaded_nccl_libraries': loaded_nccl,
                          'utc': datetime.now(timezone.utc).isoformat()}))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
