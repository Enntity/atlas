# SPDX-License-Identifier: AGPL-3.0-only
import json
from pathlib import Path
import pytest

from l1_prepare import render_descriptor


def descriptor(engine):
    command = 'serve' if engine == 'candidate' else '/start.sh'
    return ['docker', 'create', '--name', 'old', '--env', 'NUMERIC_FLAG=keep',
            '--env', 'ATLAS_EXL3_FFN_DUMP_LAYERS=0,1,2,3', '--env',
            'ATLAS_EXL3_MLA_DUMP_DIR=/old', '--volume', '/old:/tmp/diagnostic',
            '--volume', '/old-logits:/tmp/capture', '--volume', '/old-ref:/tmp/reference-logits',
            '--volume', '/old-overlay:/opt/atlas-logits-patch:ro', 'tag', command]


@pytest.mark.parametrize('engine', ['candidate', 'reference'])
def test_render_only_changes_registered_observers_and_paths(tmp_path, engine):
    original = descriptor(engine)
    args = render_descriptor(original, engine=engine, image_tag='tag', image_id='sha256:'+'1'*64,
                             name='fresh-r0', root=tmp_path, fixture=tmp_path/'fixture',
                             overlay=tmp_path/'overlay', runner_sha256='2'*64)
    assert 'NUMERIC_FLAG=keep' in args
    assert 'ATLAS_EXL3_FFN_DUMP_LAYERS=0,1' in args
    assert not any('MLA_DUMP' in x for x in args)
    assert args[-2] == 'sha256:'+'1'*64
    assert original == descriptor(engine)
    if engine == 'candidate':
        assert 'ATLAS_EXL3_NORM_SUBSTITUTE_DIR='+str(tmp_path/'fixture') in args
    else:
        assert 'ATLAS_EXL3_CHUNK_CAPTURE=serial128' in args
        assert str(tmp_path/'overlay')+':/opt/atlas-logits-patch:ro' in args


def test_render_refuses_ambiguous_image_or_auto_removal(tmp_path):
    for bad in [descriptor('reference')+['tag'], ['docker','create','--rm']+descriptor('reference')[2:]]:
        with pytest.raises(ValueError):
            render_descriptor(bad, engine='reference', image_tag='tag', image_id='sha256:'+'1'*64,
                              name='new', root=tmp_path, fixture=tmp_path/'fixtures',
                              overlay=tmp_path/'overlay', runner_sha256='2'*64)


def test_numerical_exl3_flag_survives_render(tmp_path):
    original=descriptor('candidate'); original[4:4]=['--env','ATLAS_EXL3_GATE_PERTURBATION=registered']
    args=render_descriptor(original,engine='candidate',image_tag='tag',image_id='sha256:'+'1'*64,
        name='fresh',root=tmp_path,fixture=None,overlay=tmp_path/'overlay',runner_sha256='2'*64)
    assert 'ATLAS_EXL3_GATE_PERTURBATION=registered' in args
    assert not any('NORM_SUBSTITUTE_' in arg for arg in args)


def lifecycle_setup(tmp_path,engine='candidate'):
    import hashlib
    from l1_prepare import LocalIO
    from l1_lifecycle import L1Lifecycle
    io=LocalIO(); prep=tmp_path/'prep'; prep.mkdir()
    for directory in ('fixture','overlay','tooling'):(prep/directory).mkdir()
    hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ('collect_logits.py','parity_protocol.py','attest_runtime_maps.py')}
    pins=prep/'tooling'/'pins.sha256'; pins.write_text('a'*64+'  /lib/libcublas.so.13\n'+'b'*64+'  /lib/libcublasLt.so.13\n')
    command=['sudo','-n','python3',str(prep/'tooling'/'attest_runtime_maps.py'),'{container}','--pins',str(pins)]
    remote={'hosts':{'0':['spark0'],'1':['spark1']},'image_ids':{'0':'sha256:'+'1'*64,'1':'sha256:'+'1'*64},
        'remote_python':['python3'],'remote_harness_dir':str(prep/'tooling'),'request_ids':['r'],
        'readiness_command':['curl','health'],'readiness_timeout_seconds':1,'command_timeout_seconds':1,
        'provider_checks':{rank:{'cublas':command,'cublaslt':command} for rank in ('0','1')},
        'remote_harness_sha256':hashes}
    settings={'shutdown':{'stop_timeout_seconds':30,'worker_exit_policy':'forbid'},'remote':remote,'image_id':'sha256:'+'1'*64,'image_tag':'tag',
              'container_source_sha256':{'/bin/runtime':'c'*64},'remote_mount_sha256':{'0':{},'1':{}}}
    plan={'output':str(prep),'engines':{engine:settings},'prepared_sha256':{},
          'reference_source':{'model_sha256':'d'*64,'ffn_hook_sha256':'e'*64,'runner_sha256':'f'*64}}
    root=prep/'window';root.mkdir()
    for name in ('diagnostic','records','logits'):(root/name).mkdir()
    commands={rank:['docker','create','--name','own-r'+rank,'tag','serve'] for rank in ('0','1')}
    window={'commands':commands,'observed':True}
    return io,plan,root,window,L1Lifecycle


class Transport:
    def __init__(self,hashes,malformed_rank=None,provider_wrong=False):
        self.hashes=hashes;self.commands=[];self.names={};self.malformed_rank=malformed_rank
        self.provider_wrong=provider_wrong

    def __call__(self,args,**kwargs):
        import io,tarfile,shlex,subprocess
        rank=args[0][-1];command=shlex.split(args[-1]);self.commands.append((rank,command))
        data=b'';error=b'';status=0
        if command[:3]==['docker','image','inspect']:data=('sha256:'+'1'*64).encode()
        elif command[:2]==['docker','inspect']:
            name=command[-1]
            if name in self.names:data=self.names[name].encode()
            else:status=1;error=b'Error: No such object'
        elif command[:2]==['docker','create']:
            name=command[command.index('--name')+1];cid=('0' if rank=='0' else '1')*64
            self.names[name]=cid;data=b'malformed' if rank==self.malformed_rank else cid.encode()
        elif command[:2]==['docker','wait']:data=b'0'
        elif command[:2]==['python3','-c']:
            cid=command[-1];name=next(name for name,value in self.names.items() if value==cid)
            data=json.dumps({'Id':cid,'Name':'/'+name,'State':{'Running':True,'OOMKilled':False,'ExitCode':0},'captured_epoch':10}).encode()
            if any(c[:2]==['docker','stop'] and c[-1]==cid for _,c in self.commands):
                record=json.loads(data);record['State']['Running']=False;record['captured_epoch']=20;data=json.dumps(record).encode()
        elif command[:2]==['docker','stop']:data=command[-1].encode()
        elif command[0]=='sha256sum':
            name=Path(command[1]).name;data=(self.hashes[name]+'  '+command[1]).encode()
        elif command[:2]==['docker','exec']:data=('c'*64+'  '+command[-1]).encode()
        elif 'attest_runtime_maps.py' in ' '.join(command):
            cid=command[4];report={'target':cid,'container':cid,'ok':True,'issues':[],
                'pins':command[command.index('--pins')+1],'pid':123,'maps_source':'/proc/123/maps','fs_root':'/proc/123/root',
                'libraries':[{'name':name,'path':'/lib/'+name,'sha256':digest,'pinned_sha256':digest,'ok':True}
                             for name,digest in [('libcublas.so.13','a'*64),('libcublasLt.so.13','b'*64)]]}
            if self.provider_wrong:report['container']='wrong'
            data=json.dumps(report).encode()
        elif command[:2]==['tar','-cf']:
            buffer=io.BytesIO()
            with tarfile.open(fileobj=buffer,mode='w') as archive:
                content=('rank'+rank).encode();info=tarfile.TarInfo('rank-witness');info.size=len(content)
                archive.addfile(info,io.BytesIO(content))
            data=buffer.getvalue()
        return subprocess.CompletedProcess(args,status,data,error)


def test_lifecycle_retrieves_each_rank_and_tears_down_on_collector_failure(tmp_path):
    io,plan,root,window,cls=lifecycle_setup(tmp_path)
    transport=Transport(plan['engines']['candidate']['remote']['remote_harness_sha256'])
    adapter=cls(plan,'candidate',io,transport)
    with pytest.raises(RuntimeError,match='collection failed'):
        with adapter.window(window,root):raise RuntimeError('collection failed')
    assert (root/'rank0'/'diagnostic'/'rank-witness').read_text()=='rank0'
    assert (root/'rank1'/'diagnostic'/'rank-witness').read_text()=='rank1'
    assert sum(command[:2]==['docker','stop'] for _,command in transport.commands)==2
    assert json.loads((root/'lifecycle.json').read_text())['teardown_errors']==[]


def test_ambiguous_create_still_stops_generated_owned_name(tmp_path):
    io,plan,root,window,cls=lifecycle_setup(tmp_path)
    transport=Transport(plan['engines']['candidate']['remote']['remote_harness_sha256'],malformed_rank='1')
    with pytest.raises(ValueError,match='full Docker'):
        with cls(plan,'candidate',io,transport).window(window,root):pass
    assert sum(command[:2]==['docker','stop'] for _,command in transport.commands)==2


def test_provider_target_mismatch_prevents_marker_and_collect(tmp_path):
    io,plan,root,window,cls=lifecycle_setup(tmp_path)
    transport=Transport(plan['engines']['candidate']['remote']['remote_harness_sha256'],provider_wrong=True)
    with pytest.raises(ValueError,match='target'):
        with cls(plan,'candidate',io,transport).window(window,root):pytest.fail('collector must not run')
    assert not any(command[0]=='touch' for _,command in transport.commands)
