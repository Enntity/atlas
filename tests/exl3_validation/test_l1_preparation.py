# SPDX-License-Identifier: AGPL-3.0-only
import copy
import hashlib
import json
from pathlib import Path

import pytest
from l1_prepare import LocalIO,prepare
from l1_collect import validate_plan
from test_l1_collection import descriptor,lifecycle_setup
from test_parity import provenance,thresholds


def configuration(tmp_path):
    class CPUCollectionIO(LocalIO):
        # A completed numerical campaign belongs to the existing analyzer suite;
        # this boundary fixture isolates preparation/source command behavior.
        def verify_c2_analysis(self,root,attempt,registered):
            assert registered['interpretation']['causal_interpretation_allowed'] is True
    io=CPUCollectionIO(); _,template,_,_,_=lifecycle_setup(tmp_path)
    remote=template['engines']['candidate']['remote']
    c2=tmp_path/'c2';c2.mkdir();fixture=c2/'reference-exact';fixture.mkdir()
    cases=[{'id':name,'prompt_ids':[1],'continuation_ids':[2,3,4]} for name in ('arithmetic','longer')]
    ids=['req-'+str(i) for i in range(8)]
    for name,value in [('cases.json',cases),('request-ids.json',ids),('thresholds.json',thresholds())]:io.write(c2/name,value)
    io.write(c2/'run-config.json',{'endpoint':'http://localhost:8011','model':'model','timeout_seconds':10})
    entries=[]
    for rank in (0,1):
        for index in range(8):
            path=fixture/f'L0-rank{rank}-{index:06d}-gated.bin';path.write_bytes(b'\0\0'*(index%4+1)*2)
            entries.append({'request_id':ids[index],'layer':0,'tp_rank':rank,'tp_world':2,'invocation':index,
                'stage':'gated_norm','tensor':'gated','dtype':'bf16-le','shape':[index%4+1,2],
                'file':path.name,'sha256':io.sha(path)})
    io.write(fixture/'manifest.json',{'schema':'atlas-norm-substitution-fixtures-v1','variant':'reference-exact',
        'source':{'capture_root':str(c2),'engine':'reference','schedule_sha256':'1'*64},'entries':entries})
    candidate=provenance('atlas');candidate['engine_build']['image']='sha256:'+'1'*64
    io.write(c2/'provenance.json',candidate)
    for rank in ('0','1'):io.write(c2/f'candidate-rank{rank}-create-command.json',descriptor('candidate'))
    proof=c2/'neutrality-proof.json';io.write(proof,{'schema':'atlas-c2-full-neutrality-proof-v1','passed':True,'rows_total':32,'runtime':candidate['engine_build']})
    registration={'preparation_root':str(c2),'variants':['reference-exact'],'runtime_image_id':candidate['engine_build']['image'],
        'runtime_neutrality_receipt':str(proof),'runtime_neutrality_sha256':io.sha(proof),
        'file_sha256':{str(p.relative_to(c2)):io.sha(p) for p in io.files(c2)}}
    io.write(c2/'execution-plan.json',registration)
    overlay=tmp_path/'overlay';overlay.mkdir()
    for name in ('reference_ffn_trace.py','reference_mhc_trace.py','reference_kda_trace.py','reference_chunk_trace.py','sitecustomize.py'):
        (overlay/name).write_text('# registered '+name)
    package={'schema':'atlas-upstream-hook-package-context-v5','package_file_sha256':{p.name:io.sha(p) for p in io.files(overlay)}}
    io.write(overlay/'package-binding.json',package)
    refprov=tmp_path/'ref-provenance.json';reference=provenance('vllm');reference['engine_build']['image']='sha256:'+'1'*64;io.write(refprov,reference)
    refdescriptors={}
    for rank in ('0','1'):
        path=tmp_path/f'ref-rank{rank}.json';args=descriptor('reference');args[4:4]=['--env','SERVED_MODEL_NAME=model','--env','PORT=8011'];io.write(path,args);refdescriptors[rank]=str(path)
    refremote=copy.deepcopy(remote)
    for checks in refremote['provider_checks'].values():
        for key,command in checks.items():
            checks[key]=['sudo','-n','python3',str(Path(remote['remote_harness_dir'])/'l1_reference_attest.py'),
                '--container','{container}','--pins',command[-1],'--checker',command[3],
                '--checker-sha256',remote['remote_harness_sha256']['attest_runtime_maps.py'],
                '--cmdline-pattern','worker','--proc-root','/proc']
    settings={}
    for engine,descriptors,prov,r in [('candidate',{rank:str(c2/f'candidate-rank{rank}-create-command.json') for rank in ('0','1')},str(c2/'provenance.json'),remote),
                                    ('reference',refdescriptors,str(refprov),refremote)]:
        r=copy.deepcopy(r);r['request_ids']=ids;r['readiness_command']=['curl','http://localhost:8011/v1/models']
        sources={'/bin/runtime':candidate['engine_build']['binary_sha256']} if engine=='candidate' else {'/model.py':'d'*64,'/runner.py':'f'*64}
        settings[engine]={'shutdown':{'stop_timeout_seconds':30,'worker_exit_policy':'forbid'},'descriptors':descriptors,'image_tag':'tag','image_id':'sha256:'+'1'*64,'provenance':prov,
            'remote':r,'endpoint':'http://localhost:8011','model':'model','timeout_seconds':10,
            'container_source_sha256':sources,'remote_mount_sha256':{'0':{},'1':{}}}
    bound=[refprov,c2/'provenance.json',*[Path(p) for p in refdescriptors.values()],
           *[c2/f'candidate-rank{rank}-create-command.json' for rank in ('0','1')],Path(remote['provider_checks']['0']['cublas'][-1])]
    config={'schema':'atlas-l1-preparation-config-v1','output':str(tmp_path/'output'),'attempt':'l1-explicit01',
        'c2_preparation':str(c2),'c2_plan_sha256':io.sha(c2/'execution-plan.json'),'variant':'reference-exact',
        'candidate_policy':'selected-treatment','preserved_attempts':[],'reference_overlay':str(overlay),'reference_overlay_sha256':io.sha(overlay/'package-binding.json'),
        'reference_source':{'model_sha256':'d'*64,'ffn_hook_sha256':io.sha(overlay/'reference_ffn_trace.py'),'runner_sha256':'f'*64},
        'engines':settings,'input_sha256':{str(p):io.sha(p) for p in bound}}
    attempt=c2/'completed-campaign';attempt.mkdir()
    io.write(attempt/'receipt.json',{'variant':'reference-exact','repeats_per_arm':2,'arms':{'a':[{},{}],'b':[{},{}]}})
    analysis=tmp_path/'c2-analysis.json'
    io.write(analysis,{'schema':'atlas-c2-execution-analysis-v1','execution_binding':registration,
        'execution_binding_sha256':config['c2_plan_sha256'],'interpretation':{'status':'diagnostic-intervention-bound',
        'causal_interpretation_allowed':True},'scoring_skipped_reason':None,'attempt':str(attempt),
        'receipt_sha256':io.sha(attempt/'receipt.json')})
    config.update(evidence_root=str(tmp_path),c2_analysis=str(analysis),c2_analysis_sha256=io.sha(analysis))
    return io,config


def test_prepare_binds_four_windows_and_revalidates_before_execution(tmp_path):
    io,config=configuration(tmp_path);plan=prepare(config,io)
    validate_plan(plan,io)
    assert len(plan['windows'])==4
    for window in plan['windows']:
        for args in window['commands'].values():assert 'ATLAS_EXL3_FFN_DUMP_LAYERS=0,1' in args
    changed=copy.deepcopy(plan);changed['windows'][0]['commands']['0'].append('--invented')
    with pytest.raises(ValueError,match='live command'):validate_plan(changed,io)
    module=next(iter(plan['module_sha256']));changed=copy.deepcopy(plan);changed['module_sha256'][module]='0'*64
    with pytest.raises(ValueError,match='collector source'):validate_plan(changed,io)


def test_fixture_order_or_changed_overlay_rejected_before_output_creation(tmp_path):
    io,config=configuration(tmp_path)
    path=Path(config['c2_preparation'])/'request-ids.json';io.write(path,list(reversed(io.json(path))))
    with pytest.raises(ValueError,match='C2 input'):prepare(config,io)
    assert not Path(config['output']).exists()


def test_candidate_descriptor_must_match_registered_c2_profile(tmp_path):
    io,config=configuration(tmp_path)
    path=Path(config['engines']['candidate']['descriptors']['0']);args=io.json(path);args.insert(4,'--invented');io.write(path,args)
    config['input_sha256'][str(path)]=io.sha(path)
    with pytest.raises(ValueError,match='C2 profile'):prepare(config,io)
    assert not Path(config['output']).exists()


def test_incomplete_or_failed_first_control_campaign_is_not_a_fixture_authorization(tmp_path):
    io,config=configuration(tmp_path)
    path=Path(config['c2_analysis']);analysis=io.json(path)
    analysis['interpretation']['causal_interpretation_allowed']=False
    io.write(path,analysis);config['c2_analysis_sha256']=io.sha(path)
    with pytest.raises(ValueError,match='first controls'):prepare(config,io)
    assert not Path(config['output']).exists()


def test_configured_baseline_explicitly_unsets_substitution_and_preserves_selected_binding(tmp_path):
    io,config=configuration(tmp_path);config['candidate_policy']='baseline'
    plan=prepare(config,io)
    for window in plan['windows'][:2]:
        assert not any('NORM_SUBSTITUTE' in arg for args in window['commands'].values() for arg in args)
    validate_plan(plan,io)
    assert plan['variant']=='reference-exact' and plan['candidate_policy']=='baseline'


def test_reference_package_change_is_rejected_before_new_preparation(tmp_path):
    io,config=configuration(tmp_path)
    (Path(config['reference_overlay'])/'reference_mhc_trace.py').write_text('changed hook')
    with pytest.raises(ValueError,match='overlay package'):prepare(config,io)
    assert not Path(config['output']).exists()


@pytest.mark.parametrize('engine,field,value',[('candidate','endpoint','http://foreign.invalid:8011'),
    ('candidate','model','wrong-model'),('candidate','timeout_seconds',11),
    ('reference','endpoint','http://foreign.invalid:8011'),('reference','model','wrong-model')])
def test_collector_endpoint_model_timeout_bind_actual_local_engine_profile(tmp_path,engine,field,value):
    io,config=configuration(tmp_path);config['engines'][engine][field]=value
    with pytest.raises(ValueError,match='collection|endpoint|model'):prepare(config,io)
    assert not Path(config['output']).exists()


def test_c2_recomputation_allows_only_two_ssot_evaluation_timestamps():
    from l1_prepare import analysis_equivalent
    report={'execution_binding':{'runtime_image_id':'source'},'first_control':True,
            'repeat_pair_scores':[{'evaluated_utc':'2026-10-14T00:00:00+00:00','metric':0.25},
                                  {'evaluated_utc':'2026-10-14T00:00:00+00:00','metric':0.5}],
            'other':{'evaluated_utc':'frozen-other-timestamp'}}
    recomputed=copy.deepcopy(report)
    for pair in recomputed['repeat_pair_scores']:pair['evaluated_utc']='2026-10-14T01:00:00+00:00'
    assert analysis_equivalent(report,recomputed)
    for field in ('execution_binding','first_control','metric','other'):
        changed=copy.deepcopy(recomputed)
        if field=='metric':changed['repeat_pair_scores'][0]['metric']=0.26
        elif field=='other':changed['other']['evaluated_utc']='changed'
        else:changed[field]='changed'
        assert not analysis_equivalent(report,changed)
    assert report['repeat_pair_scores'][0]['evaluated_utc']=='2026-10-14T00:00:00+00:00'
