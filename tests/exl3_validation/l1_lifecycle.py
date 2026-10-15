# SPDX-License-Identifier: AGPL-3.0-only
"""L1 generic two-engine lifecycle; shared frozen adapter owns transport receipts."""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import time

from c2_remote_adapter import RemoteAdapter, require
from attest_runtime_maps import read_pins
from l1_shutdown import validate_policy, validate_shutdown
from l1_binding import registered_requests


class L1Lifecycle(RemoteAdapter):
    def __init__(self, plan, engine, io, runner=subprocess.run):
        self.plan,self.engine,self.io=plan,engine,io
        super().__init__(plan['engines'][engine]['remote'],runner)
        self.settings=plan['engines'][engine]
        validate_policy(self.settings['shutdown'],engine)

    def verify_hash(self, rank, path, digest, container=None):
        command=['sha256sum',str(path)]
        if container:
            command=['docker','exec',container]+command
        tokens=self._run(rank,command).stdout.decode().split()
        require(tokens == [digest,str(path)],'live source fingerprint differs: '+str(path))

    def provider(self,rank,command,cid):
        args=[arg.replace('{container}',cid) for arg in command]
        report=json.loads(self._run(rank,args).stdout)
        require(report['target']==report['container']==cid and report['ok'] is True
                and report['issues']==[], 'provider actual target/verdict differs')
        require(type(report['pid']) is int and report['pid']>0
                and report['maps_source']==f"/proc/{report['pid']}/maps"
                and report['fs_root']==f"/proc/{report['pid']}/root", 'provider process evidence differs')
        require(report['pins']==args[args.index('--pins')+1], 'provider actual pins path differs')
        pins=read_pins(args[args.index('--pins')+1])
        libraries={item['name']:item for item in report['libraries']}
        require(len(report['libraries'])==2 and set(libraries)=={'libcublas.so.13','libcublasLt.so.13'}, 'both providers required')
        for name,item in libraries.items():
            require(item['path']==pins[name]['path'] and item['ok'] is True and item['sha256']==item['pinned_sha256']==pins[name]['sha256'],
                    'provider mapped library differs from registered pins')
        if self.engine=='reference':
            require(report['pins_sha256']==self.io.sha(args[args.index('--pins')+1])
                    and report['pid'] in report['owned_processes'] and report['cmdline']
                    and report['cmdline_pattern']==args[args.index('--cmdline-pattern')+1],
                    'reference worker identity/pin witness differs')
        return report

    def startup_sources(self,rank,root):
        code=("import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
              "print(json.dumps([json.loads(x.read_text()) for x in sorted(p.glob('hook-source-vllm.models.glm5next.nvidia.model-*.json'))]))")
        records=json.loads(self._run(rank,self.config['remote_python']+['-c',code,str(root/'diagnostic')]).stdout)
        source=self.plan['reference_source']
        require(bool(records) and all(record.get('module')=='vllm.models.glm5next.nvidia.model'
            and record.get('sha256')==source['model_sha256']
            and record.get('hook_sha256')==source['ffn_hook_sha256']
            and record.get('layers')==[0,1] for record in records), 'reference startup hook source/layers differ')

    def snapshot(self,rank,cid):
        code=("import json,subprocess,time,sys; "
              "d=json.loads(subprocess.check_output(['docker','inspect','--type','container',sys.argv[1]])); "
              "assert len(d)==1; d[0]['captured_epoch']=time.time(); print(json.dumps(d[0]))")
        return json.loads(self._run(rank,self.config['remote_python']+['-c',code,cid]).stdout)

    def shutdown(self,rank,cid,name,root):
        before=self.snapshot(rank,cid)
        self._run(rank,['docker','logs',cid])
        stop=self._run(rank,['docker','stop','--timeout',str(self.settings['shutdown']['stop_timeout_seconds']),cid])
        wait=self._run(rank,['docker','wait',cid])
        after=self.snapshot(rank,cid)
        events=[]
        if after['State']['ExitCode']!=0:
            event_result=self._run(rank,['docker','events','--since',str(before['captured_epoch']),
                '--until',str(after['captured_epoch']),'--filter','container='+cid,'--format','{{json .}}'])
            events=[json.loads(line) for line in event_result.stdout.decode().splitlines() if line]
        classification=validate_shutdown(policy=self.settings['shutdown'],engine=self.engine,rank=rank,cid=cid,
            name=name,before=before,after=after,stop_stdout=stop.stdout.decode(),wait_stdout=wait.stdout.decode(),
            events=events,collection_complete=self.collection_proof is not None,attested=self.attested=={'0','1'})
        self.io.write(root/('shutdown-rank'+rank+'.json'),{'classification':classification,'before':before,
            'after':after,'events':events,'collection_proof':self.collection_proof,'policy':self.settings['shutdown']})

    def validate_completed_reference(self,root):
        from l1_collect import build_binding
        for directory in ('collection','logits'):
            self._retrieve('0',root/directory,root/directory)
        manifest=self.io.artifact(root/'collection')[0]
        require(manifest['cases']==self.io.json(root/'cases.json')
            and manifest['request_ids']==self.config['request_ids'] and len(manifest['captures'])==8
            and manifest['provenance']==self.io.json(root/'provenance.json')
            and manifest['thresholds']==self.io.json(root/'thresholds.json')
            and manifest['model']==self.settings['model'] and manifest['endpoint']==self.settings['endpoint']
            and manifest['harness_sha256']=={name:self.config['remote_harness_sha256'][name]
                for name in ('collect_logits.py','parity_protocol.py')},'completed reference collection differs from registration')
        candidate=Path(self.plan['output'])/self.plan['attempt']/'candidate-observed'
        binding=build_binding(candidate,root,self.plan['reference_source'],self.io)
        path=root/'pre-stop-binding.json';self.io.write(path,binding)
        for rank in ('0','1'):
            destination=root/('pre-stop-rank'+rank)/'diagnostic';self.io.mkdir(destination)
            self._retrieve(rank,root/'diagnostic',destination)
            registered_requests(candidate/'collection',root/'collection',path,destination)
        self.collection_proof={'binding_sha256':self.io.sha(path),'rows':len(manifest['captures']),
            'file_sha256':{str(path.relative_to(root)):self.io.sha(path) for path in self.io.files(root)
                if path.is_relative_to(root/'collection') or path.is_relative_to(root/'logits')
                or path.is_relative_to(root/'pre-stop-rank0') or path.is_relative_to(root/'pre-stop-rank1')}}

    @contextmanager
    def window(self, window, root):
        root=Path(root)
        self.active={'run_root':root}
        self.collection_proof=None; self.attested=set()
        owned,created,started,errors={},{},{},[]
        staged=set()
        try:
            commands=window['commands']
            for rank in ('0','1'):
                require(not self._run(rank,['docker','ps','-q']).stdout.strip(),'host has active containers')
                require(not self._run(rank,['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).stdout.strip(),
                        'host has active GPU owner')
                image=self._run(rank,['docker','image','inspect','--format','{{.Id}}',self.settings['image_tag']]).stdout.decode().strip()
                require(image==self.settings['image_id'],'registered image differs')
                for path,digest in self.settings['remote_mount_sha256'][rank].items():
                    self.verify_hash(rank,path,digest)
            # Inputs are staged once per adapter, run roots once per window. No overwrite.
            for rank in ('0','1'):
                for path in (Path(self.plan['output'])/'tooling',Path(self.plan['output'])/'fixture',
                             Path(self.plan['output'])/'overlay'):
                    if (rank,path) not in self.staged:
                        self._stage(rank,path); self.staged.add((rank,path))
                self._stage(rank,root); staged.add(rank)
                for name,digest in self.config['remote_harness_sha256'].items():
                    self.verify_hash(rank,Path(self.config['remote_harness_dir'])/name,digest)
                for path,digest in self.plan['prepared_sha256'].items():
                    if path.startswith(('tooling/','overlay/','fixture/')):
                        self.verify_hash(rank,Path(self.plan['output'])/path,digest)
                args=commands[rank]
                name=args[args.index('--name')+1]
                result=self._run(rank,['docker','inspect','--format','{{.Id}}',name],check=False)
                require(result.returncode==1 and (b'no such object' in result.stderr.lower()
                        or b'no such container' in result.stderr.lower()),'fresh name absence unproven')
                owned[rank]=name
                cid=self._run(rank,args).stdout.decode().strip()
                require(len(cid)==64 and all(c in '0123456789abcdef' for c in cid),'full Docker ID required')
                created[rank]=cid
            for rank in ('1','0'):
                self._run(rank,['docker','start',created[rank]]); started[rank]=created[rank]
            deadline=time.monotonic()+self.config['readiness_timeout_seconds']
            while self._run('0',self.config['readiness_command'],check=False).returncode:
                require(time.monotonic()<deadline,'readiness timed out')
                time.sleep(min(1,max(0,deadline-time.monotonic())))
            for rank in ('0','1'):
                for path,digest in self.settings['container_source_sha256'].items():
                    self.verify_hash(rank,path,digest,created[rank])
                for command in self.config['provider_checks'][rank].values():
                    self.provider(rank,command,created[rank])
                if self.engine=='reference':
                    self.startup_sources(rank,root)
                self.attested.add(rank)
            if window['observed']:
                for rank in ('0','1'):
                    self._run(rank,['touch',str(root/'diagnostic'/'capture-enabled')])
            yield self
        finally:
            for rank,name in owned.items():
                cid=created.get(rank)
                if cid is None:
                    try:
                        result=self._run(rank,['docker','inspect','--format','{{.Id}}',name],check=False)
                        if result.returncode:
                            continue
                        cid=result.stdout.decode().strip()
                        require(len(cid)==64 and all(c in '0123456789abcdef' for c in cid),'owned container ID malformed')
                    except Exception as exc:
                        errors.append(str(exc)); continue
                try:
                    self.shutdown(rank,cid,name,root)
                except Exception as exc:
                    errors.append(str(exc))
                    # Snapshot/validation errors cannot strand an owned container.
                    for args in (['docker','stop','--timeout',str(self.settings['shutdown']['stop_timeout_seconds']),cid],
                                 ['docker','wait',cid]):
                        try:
                            self._run(rank,args)
                        except Exception as cleanup_exc:
                            errors.append(str(cleanup_exc))
            for rank in sorted(staged):
                for directory in ('diagnostic','records'):
                    destination=root/('rank'+rank)/directory
                    self.io.mkdir(destination)
                    try:
                        self._retrieve(rank,root/directory,destination)
                    except Exception as exc:
                        errors.append(str(exc))
                if rank=='0':
                    for directory in ('logits','collection'):
                        try:
                            self._retrieve(rank,root/directory,root/directory)
                        except Exception as exc:
                            errors.append(str(exc))
            receipts=[]
            for path in self.io.files(root):
                if path.name.startswith('remote-'):
                    receipts.append({'file':str(path.relative_to(root)), 'sha256':self.io.sha(path)})
            self.io.write(root/'lifecycle.json',{'engine':self.engine,'created':created,'owned_names':owned,
                    'started':started,'teardown_errors':errors,'commands':window['commands'],'command_receipts':receipts})
            self.active=None
            require(not errors,'L1 teardown/retrieval errors: '+'; '.join(errors))

    def collect(self, root):
        root=Path(root)
        command=self.config['remote_python']+[str(Path(self.config['remote_harness_dir'])/'collect_logits.py')]
        fields={'endpoint':self.settings['endpoint'],'model':self.settings['model'],
                'cases':root/'cases.json','capture-dir':root/'logits','out':root/'collection',
                'provenance':root/'provenance.json','thresholds':root/'thresholds.json',
                'request-ids':root/'request-ids.json','timeout':self.settings['timeout_seconds']}
        for key,value in fields.items():
            command.extend(['--'+key,str(value)])
        self._run('0',command)
        if self.engine=='reference':
            self.validate_completed_reference(root)
