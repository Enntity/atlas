# SPDX-License-Identifier: AGPL-3.0-only
"""Decode the installed cooperative autotuner's source-bound launch record.

All decoding here is diagnostics-only; it changes no arithmetic."""
import hashlib
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


def cache_manifest(payload):
    """Decode the full EX3ATUNE disk cache into its final per-hash record map.

    Diagnostics-only disk telemetry; no arithmetic changes. Canonical map
    encoding: take each hash's final record (the LAST occurrence in file
    order, matching the installed std::map overwrite), serialize it with
    ``struct.pack('<QiiiiII', *record)``, concatenate the serializations in
    ascending numeric hash order, and hash that concatenation. Record field
    values are deliberately NOT validated; only the attested aarch64 header
    sizes, format version and record truncation are rejected.
    """
    if len(payload) < 16:
        raise ValueError('truncated cooperative cache header')
    magic, version, size = struct.unpack_from('<8sII', payload)
    if magic != b'EX3ATUNE':
        raise ValueError('unknown cooperative cache magic')
    if version != 1:
        raise ValueError('unsupported cooperative cache version')
    if size != 32:
        raise ValueError('record size outside the attested aarch64 ABI')
    if (len(payload) - 16) % 32:
        raise ValueError('truncated cooperative cache records')
    final = {}
    record_count = 0
    for offset in range(16, len(payload), 32):
        record = struct.unpack_from('<QiiiiII', payload, offset)
        final[record[0]] = record  # Installed std::map keeps the last record per hash.
        record_count += 1
    canonical = b''.join(struct.pack('<QiiiiII', *final[hash_key])
                         for hash_key in sorted(final))
    return {'raw_sha256': hashlib.sha256(payload).hexdigest(),
            'record_map_sha256': hashlib.sha256(canonical).hexdigest(),
            'record_count': record_count,
            'unique_hash_count': len(final),
            'duplicate_overwrite_count': record_count - len(final)}


def integer_constant(sources, name):
    pattern=rf'(?:#define\s+{name}\s+|\b{name}\s*=\s*)(\d+)'
    matches=[int(m.group(1)) for text in sources.values() for m in re.finditer(pattern,text)]
    if not matches or len(set(matches))!=1:
        raise ValueError(f'authoritative installed integer constant {name} missing or ambiguous')
    return matches[0]
