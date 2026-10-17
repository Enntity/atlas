# SPDX-License-Identifier: AGPL-3.0-only
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
from pathlib import Path
import threading

import numpy as np
import pytest

from collect_logits import collect
from l1_collect import run
from l1_binding import registered_requests
from test_l1_preparation import configuration
from l1_prepare import prepare


class CPUCollection:
    """Only the engine/lifecycle boundary is replaced; both collectors are real HTTP."""
    calls=[]
    changed_window=None
    fail_window=None
    changed_text_window=None
    changed_threshold_window=None

    def __init__(self,plan,engine,io):
        self.plan,self.engine,self.io=plan,engine,io

    @contextmanager
    def window(self,window,root):
        self.name=window['name'];self.root=root;self.calls.append(self.name)
        if self.engine=='reference':
            source=self.plan['reference_source']
            for rank in ('0','1'):
                destination=root/('rank'+rank)/'diagnostic';destination.mkdir(parents=True)
                self.io.write(destination/'hook-source-vllm.models.glm5next.nvidia.model-000.json',
                    {'module':'vllm.models.glm5next.nvidia.model','sha256':source['model_sha256'],
                     'hook_sha256':source['ffn_hook_sha256'],'layers':[0,1]})
        try:
            yield
        finally:
            self.io.write(root/'lifecycle.json',{'teardown_errors':[],'cpu_boundary_fixture':True})

    def collect(self,root):
        if self.name==self.fail_window:raise RuntimeError('preserved engine collection failure')
        adapter=self;counter=[0]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                index=counter[0];counter[0]+=1
                stem=f'logits-{index:06d}';array=np.arange(8,dtype='<f4')
                if adapter.name==adapter.changed_window and index==3:array[0]+=1
                (root/'logits'/(stem+'.f32')).write_bytes(array.tobytes())
                adapter.io.write(root/'logits'/(stem+'.json'),{'file':stem+'.f32','dtype':'f32-le','rows':1,'vocab':8})
                response_id=f'cmpl-{adapter.name}-{index}'
                if adapter.engine=='reference':
                    tokens=len(request['prompt']);context={'request_id':response_id+'-0-0123abcd',
                        'computed_before':0,'scheduled_tokens':tokens,'logical_after':tokens,'prefix_tokens':tokens,
                        'prefill_tokens':tokens,'phase':'terminal-prefill','discard':False,'sampleable_terminal_prefix':True,
                        'runner_source_sha256':adapter.plan['reference_source']['runner_sha256']}
                    adapter.io.write(root/'logits'/f'head-position-{index:06d}.json',
                                     {'head_capture':stem,'rows':1,'context':context})
                text='changed' if adapter.name==adapter.changed_text_window and index==0 else 'x'
                payload=json.dumps({'id':response_id,'choices':[{'text':text,'index':0}]}).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
            def log_message(self,*args):pass
        server=HTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            thresholds=self.io.json(root/'thresholds.json')
            if self.name==self.changed_threshold_window:thresholds['top_k']=3
            collect('http://127.0.0.1:'+str(server.server_port),'model',self.io.json(root/'cases.json'),
                root/'logits',root/'collection',self.io.json(root/'provenance.json'),10,
                thresholds,request_ids=self.io.json(root/'request-ids.json'))
        finally:
            server.shutdown();thread.join();server.server_close()


def configured(tmp_path):
    io,config=configuration(tmp_path);config['candidate_policy']='baseline'
    plan=prepare(config,io)
    CPUCollection.calls=[];CPUCollection.changed_window=None;CPUCollection.fail_window=None
    CPUCollection.changed_text_window=None;CPUCollection.changed_threshold_window=None
    return io,plan


def test_serial_four_windows_real_32_rows_and_exact_opaque_head_binding(tmp_path):
    io,plan=configured(tmp_path);receipt=run(plan,io,CPUCollection)
    assert CPUCollection.calls==['candidate-control','candidate-observed','reference-control','reference-observed']
    assert receipt['observer_passed'] is True
    assert sum(len(rows) for rows in receipt['observer_rows_equal'].values())==16
    attempt=Path(plan['output'])/plan['attempt'];reference=attempt/'reference-observed';candidate=attempt/'candidate-observed'
    requests,_,_=registered_requests(candidate/'collection',reference/'collection',attempt/'binding.json',reference/'rank1'/'diagnostic')
    assert len(requests['reference'])==8
    assert requests['reference'][0]['gpu_request_id']=='cmpl-reference-observed-0-0-0123abcd'
    assert receipt['attribution_authorized'] is False  # stage/policy completeness remains the bisect authority


def test_changed_actual_observer_row_preserves_receipt_and_withholds_binding(tmp_path):
    io,plan=configured(tmp_path);CPUCollection.changed_window='candidate-observed'
    with pytest.raises(ValueError,match='observer controls differ'):run(plan,io,CPUCollection)
    attempt=Path(plan['output'])/plan['attempt'];receipt=io.json(attempt/'receipt.json')
    assert receipt['observer_rows_equal']['candidate']==[True,True,True,False,True,True,True,True]
    assert receipt['status']=='failed' and not (attempt/'binding.json').exists()


def test_later_engine_failure_preserves_completed_candidate_windows(tmp_path):
    io,plan=configured(tmp_path);CPUCollection.fail_window='reference-control'
    with pytest.raises(RuntimeError,match='preserved engine'):run(plan,io,CPUCollection)
    attempt=Path(plan['output'])/plan['attempt'];receipt=io.json(attempt/'receipt.json')
    assert set(receipt['windows'])=={'candidate-control','candidate-observed'}
    assert (attempt/'reference-control'/'lifecycle.json').exists()
    assert not (attempt/'reference-observed').exists()


@pytest.mark.parametrize('field',['changed_text_window','changed_threshold_window'])
def test_observer_text_or_threshold_changes_cannot_pass_with_equal_logits(tmp_path,field):
    io,plan=configured(tmp_path);setattr(CPUCollection,field,'candidate-observed')
    with pytest.raises(ValueError,match='observer control'):run(plan,io,CPUCollection)
    assert not (Path(plan['output'])/plan['attempt']/'binding.json').exists()


def test_real_completed_reference_prerequisite_binds_all8_heads_and_both_sources(tmp_path):
    import io as bytesio
    import shlex,subprocess,tarfile
    from l1_lifecycle import L1Lifecycle
    io,plan=configured(tmp_path);run(plan,io,CPUCollection)
    root=Path(plan['output'])/plan['attempt']/'reference-observed'
    manifest=io.json(root/'collection'/'manifest.json')
    plan['engines']['reference']['endpoint']=manifest['endpoint']  # ephemeral CPU transport target
    def remote(args,**kwargs):
        rank=args[0][-1];command=shlex.split(args[-1])
        assert command[:2]==['tar','-cf']
        path=Path(command[command.index('-C')+1])
        if path.name=='diagnostic':path=root/('rank'+rank)/'diagnostic'
        buffer=bytesio.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w') as archive:
            for child in path.iterdir():archive.add(child,arcname=child.name)
        return subprocess.CompletedProcess(args,0,buffer.getvalue(),b'')
    adapter=L1Lifecycle(plan,'reference',io,remote);adapter.collection_proof=None
    adapter.validate_completed_reference(root)
    assert adapter.collection_proof['rows']==8
    assert any(key.startswith('pre-stop-rank1/') for key in adapter.collection_proof['file_sha256'])
    assert (root/'pre-stop-binding.json').exists()
    manifest['captures']=manifest['captures'][:7];io.write(root/'collection'/'manifest.json',manifest)
    with pytest.raises(ValueError):adapter.validate_completed_reference(root)
