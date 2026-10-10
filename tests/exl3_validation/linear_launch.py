# SPDX-License-Identifier: AGPL-3.0-only
"""Decode the installed cooperative autotuner's source-bound launch record."""
import re
import struct


def launch_key(rows, k, n, bits, device, cc, sms, version):
    result = 1469598103934665603
    row_bucket = min(1 << (max(rows,2)-1).bit_length(),16)
    for value in (row_bucket,k,n,bits,1,device,cc,sms,2):
        result = ((result ^ value)*1099511628211) & 0xffffffffffffffff
    return ((result ^ version)*1099511628211) & 0xffffffffffffffff


def cache_record(payload, key, tag):
    if len(payload)<16:
        raise ValueError('truncated cooperative cache header')
    magic,version,size = struct.unpack_from('<8sII',payload)
    if (magic,version,size)!=(b'EX3ATUNE',1,32) or (len(payload)-16)%size:
        raise ValueError('unsupported or truncated cooperative cache format')
    matches=[]
    for offset in range(16,len(payload),size):
        record=struct.unpack_from('<QiiiiII',payload,offset)
        if record[0]==key:
            matches.append(record)
    if not matches:
        raise ValueError('actual cooperative launch key missing from captured cache')
    # Installed std::map assigns the last persisted record for each hash.
    _,actual_tag,block,sms,concurrency,r0,r1=matches[-1]
    if actual_tag!=tag or min(block,sms,concurrency)<=0 or r0 or r1:
        raise ValueError('cooperative cache record differs from actual returned launch')
    return {'hash':f'{key:016x}','tag':actual_tag,'block_dim':block,
            'grid':[sms,1,concurrency],'num_sms':sms,'concurrency':concurrency,
            'matching_records':len(matches)}


def integer_constant(sources, name):
    pattern=rf'(?:#define\s+{name}\s+|\b{name}\s*=\s*)(\d+)'
    matches=[int(m.group(1)) for text in sources.values() for m in re.finditer(pattern,text)]
    if not matches or len(set(matches))!=1:
        raise ValueError(f'authoritative installed integer constant {name} missing or ambiguous')
    return matches[0]
