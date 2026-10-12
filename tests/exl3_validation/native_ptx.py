# SPDX-License-Identifier: AGPL-3.0-only
"""Fail-closed whole-linear native PTX replay, or explicitly separate source mode."""
import hashlib
import json
from pathlib import Path
import re

NVRTC_OPTIONS = ('--std=c++17', '--fmad=false')

REQUIRED = {
    'exl3': {'entries': ('exl3_input_hadamard_f16','exl3_input_hadamard',
                        'exl3_mul1_matmul','exl3_output_hadamard','exl3_output_hadamard_f32'),
             'sources': ('exl3.cu','exl3_common.cuh')},
    'residual_add': {'entries': ('bf16_to_f32',),'sources': ('residual_add.cu',)},
    'exl3_linear_mma': {'entries': ('exl3_mul1_matmul_mma',),
                        'sources': ('exl3_linear_mma.cu','exl3_mma.cuh','exl3_common.cuh')},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_binding(directory,binding_path,common,required=REQUIRED):
    directory,common=Path(directory),Path(common)
    value=json.loads(Path(binding_path).read_text())
    for key in ('build_source_sha256','embedded_binary_sha256'):
        if not re.fullmatch(r'[0-9a-f]{64}',str(value.get(key,''))):
            raise ValueError('PTX binding requires exact '+key)
    if not isinstance(value.get('image'),str) or not value['image']:
        raise ValueError('PTX binding requires immutable image identity')
    for module,contract in required.items():
        item=value.get('modules',{}).get(module)
        if not isinstance(item,dict) or item.get('contained_in_binary') is not True:
            raise ValueError('PTX module requires final executable containment attestation: '+module)
        name=item.get('file')
        if not isinstance(name,str) or Path(name).name!=name or not name.endswith('.ptx'):
            raise ValueError('PTX file must be a local basename')
        path=directory/name
        if sha(path)!=item.get('sha256'):raise ValueError('embedded PTX hash differs: '+module)
        entries=set(re.findall(r'\.entry\s+([A-Za-z0-9_]+)\s*\(',path.read_text()))
        if not set(contract['entries']).issubset(entries):
            raise ValueError('embedded PTX is missing required exports: '+module)
        for entry,count in contract.get('argument_counts',{}).items():
            match=re.search(r'\.entry\s+'+re.escape(entry)+r'\s*\((.*?)\)',path.read_text(),re.S)
            if (not match or len(re.findall(r'\.param\b',match[1]))!=count or
                    item.get('entry_argument_counts',{}).get(entry)!=count):
                raise ValueError('embedded PTX argument ABI differs: '+entry)
        sources=item.get('sources',{})
        for name in contract['sources']:
            if sha(common/name)!=sources.get(name):
                raise ValueError('frozen CUDA source differs from embedded PTX binding: '+name)
    return value


def load_module(cp,common,module,ptx_dir=None,binding=None):
    if (ptx_dir is None)!=(binding is None):
        raise ValueError('PTX directory and validated binding required together')
    if ptx_dir is not None:
        return cp.RawModule(path=str(Path(ptx_dir)/binding['modules'][module]['file']))
    return cp.RawModule(code=(Path(common)/(module+'.cu')).read_text(),
                        options=NVRTC_OPTIONS + ('-I'+str(common),))
