# SPDX-License-Identifier: AGPL-3.0-only
"""Read-only reference GPU-worker cuBLAS attestation, scoped to one container."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import attest_runtime_maps as checker
from c2_remote_adapter import require


def select_worker(records,init_pid,cgroup,pattern):
    """Pure ownership/selection logic; reject multiple matching mapped workers."""
    parents={record['pid']:record['ppid'] for record in records}
    owned={init_pid}
    while True:
        expanded=owned|{pid for pid,parent in parents.items() if parent in owned}
        if expanded==owned:
            break
        owned=expanded
    matches=[]
    for record in records:
        if record['pid'] not in owned or record['cgroup']!=cgroup or not re.search(pattern,record['cmdline']):
            continue
        paths=[entry['path'] for entry in checker.parse_maps(record['maps']) if entry['path']]
        names={Path(path.removesuffix(' (deleted)')).name for path in paths}
        if {'libcublas.so.13','libcublasLt.so.13'} <= names and any(name.startswith('libcuda.so') for name in names):
            matches.append(record)
    require(len(matches)==1,'exactly one mapped GPU worker must match registered process pattern')
    return matches[0]


class ProcIO:
    def __init__(self,proc_root):
        self.root=Path(proc_root)
        require(self.root.is_absolute(),'explicit absolute proc root required')

    def command(self,args):
        return subprocess.run(args,check=True,capture_output=True).stdout

    def read(self,path):
        return Path(path).read_bytes()

    def process(self,pid):
        base=self.root/str(pid)
        status=(base/'status').read_text()
        parent=int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
        return {'pid':pid,'ppid':parent,'cmdline':self.read(base/'cmdline').replace(b'\0',b' ').decode().strip(),
                'cgroup':(base/'cgroup').read_text(),'maps':(base/'maps').read_text(),
                'start_time':(base/'stat').read_text().rsplit(')',1)[1].split()[19]}

    def library_hash(self,pid,path):
        require(Path(path).is_absolute(),'mapped library absolute path required')
        return hashlib.sha256(self.read(self.root/str(pid)/'root'/path.lstrip('/'))).hexdigest()


def attest(container,pins_path,pattern,io):
    require(re.fullmatch('[0-9a-f]{64}',container) is not None,'full container CID required')
    info=json.loads(io.command(['docker','inspect','--type','container',container]))
    require(len(info)==1 and info[0]['Id']==container and info[0]['State']['Running'] is True,
            'actual running container identity differs')
    init_pid=info[0]['State']['Pid']
    require(type(init_pid) is int and init_pid>0,'container init PID invalid')
    top=io.command(['docker','top',container,'-eo','pid,args']).decode().splitlines()[1:]
    pids=[int(line.split(None,1)[0]) for line in top if line.strip()]
    require(init_pid in pids and len(pids)==len(set(pids)),'container process inventory invalid')
    records=[io.process(pid) for pid in pids]
    root=next(record for record in records if record['pid']==init_pid)
    selected=select_worker(records,init_pid,root['cgroup'],pattern)
    paths=[entry['path'] for entry in checker.parse_maps(selected['maps']) if entry['path']]
    pins=checker.read_pins(pins_path)
    result=checker.verify(paths,pins,lambda path:io.library_hash(selected['pid'],path))
    require(result['ok'],'reference mapped providers differ from explicit pins')
    # Reject PID reuse, ownership changes or newly remapped providers during inspection.
    after=io.process(selected['pid'])
    require(after==selected,'selected worker changed during attestation')
    result.update(target=container,container=container,pid=selected['pid'],init_pid=init_pid,
        cmdline=selected['cmdline'],cmdline_pattern=pattern,cgroup=selected['cgroup'],
        maps_source=str(io.root/str(selected['pid'])/'maps'),fs_root=str(io.root/str(selected['pid'])/'root'),
        maps_sha256=hashlib.sha256(selected['maps'].encode()).hexdigest(),
        cmdline_sha256=hashlib.sha256(selected['cmdline'].encode()).hexdigest(),
        pins=str(pins_path),pins_sha256=hashlib.sha256(io.read(pins_path)).hexdigest(),
        owned_processes=pids)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('container','pins','cmdline-pattern','proc-root','checker','checker-sha256'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args(); io=ProcIO(args.proc_root)
    require(Path(args.checker).resolve()==Path(checker.__file__).resolve()
            and hashlib.sha256(io.read(args.checker)).hexdigest()==args.checker_sha256,
            'provider checker implementation differs from registered source')
    print(json.dumps(attest(args.container,args.pins,args.cmdline_pattern,io),indent=2))


if __name__=='__main__':
    main()
