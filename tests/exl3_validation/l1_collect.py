# SPDX-License-Identifier: AGPL-3.0-only
"""Execute a registered L1 plan; observer mismatches withhold L1 input binding."""
import argparse
import json
import hashlib
import copy
from pathlib import Path

from c2_ab_driver import locate_substitution_records
from c2_remote_adapter import require
from l1_binding import registered_requests, source_binding
from l1_lifecycle import L1Lifecycle
from l1_prepare import LocalIO, render_descriptor, OBSERVER_KEYS, injected_provenance


def validate_plan(plan,io):
    require(plan['schema']=='atlas-l1-collection-plan-v1','L1 plan schema differs')
    root=Path(plan['output'])
    for path,digest in plan['input_sha256'].items():
        require(io.sha(path)==digest,'registered external input changed: '+path)
    for relative,digest in plan['prepared_sha256'].items():
        require(io.sha(root/relative)==digest,'prepared source/input differs: '+relative)
    for path,digest in plan['module_sha256'].items():
        require(io.sha(path)==digest,'registered collector source changed: '+path)
    names=['candidate-control','candidate-observed','reference-control','reference-observed']
    require([w['name'] for w in plan['windows']]==names,'serial observer schedule differs')
    for window in plan['windows']:
        require(window['engine']==window['name'].split('-')[0]
                and window['observed']==window['name'].endswith('observed'),'observer registration differs')
        settings=plan['engines'][window['engine']]
        for rank,path in settings['descriptors'].items():
            expected=render_descriptor(io.json(path),engine=window['engine'],image_tag=settings['image_tag'],
                image_id=settings['image_id'],name='atlas-l1-'+plan['attempt']+'-'+window['name']+'-r'+rank,
                root=root/plan['attempt']/window['name'],
                fixture=(root/'fixture' if plan['candidate_policy']=='selected-treatment' else None),
                overlay=root/'overlay',runner_sha256=plan['reference_source']['runner_sha256'])
            require(window['commands'][rank]==expected,'registered live command differs')
        require(window['provenance']==injected_provenance(io.json(settings['provenance']),window['commands']),
                'registered live provenance differs')


def observer_equal(control,observed,io):
    left,right=io.artifact(control)[0],io.artifact(observed)[0]
    for key in ('cases','tokens_sha256','vocab','request_policy','request_ids','generated_text','thresholds'):
        require(left[key]==right[key],'observer control profile/schedule differs: '+key)
    profiles=[]
    for manifest in (left,right):
        profile=copy.deepcopy(manifest['provenance'])
        for environment in [profile['environment'],*profile['rank_environments'].values()]:
            for key in OBSERVER_KEYS:
                environment.pop(key,None)
        profiles.append(profile)
    require(profiles[0]==profiles[1], 'observer numerical/source profile differs')
    require(len(left['captures'])==len(right['captures'])==8,'observer control must cover eight rows')
    actual=[]
    for root,manifest in ((Path(control),left),(Path(observed),right)):
        hashes=[]
        for capture in manifest['captures']:
            path=Path(capture['capture'])
            require(path.parent==root.parent/'logits', 'observer raw capture root differs')
            row,metadata=io.raw(path)
            require(metadata==capture['capture_metadata']
                and io.sha(path.parent/metadata['file'])==capture['raw_sha256'], 'observer raw capture differs')
            digest=hashlib.sha256(row.astype('<f4').tobytes()).hexdigest()
            require(digest==capture['row_sha256'],'observer row fingerprint differs')
            hashes.append(digest)
        actual.append(hashes)
    return [a==b for a,b in zip(*actual)]


def build_binding(candidate,reference,source,io):
    manifest=io.json(Path(reference)/'collection'/'manifest.json')
    heads=[]
    for capture in manifest['captures']:
        sidecar=Path(capture['capture'])
        require(sidecar.parent==Path(reference)/'logits','head capture root differs from registered run')
        head=sidecar.with_name('head-position-'+sidecar.stem.removeprefix('logits-')+'.json')
        response=Path(reference)/'collection'/'responses'/capture['response']
        heads.append({'head_position_file':str(head),'head_position_sha256':io.sha(head),
            'raw_sidecar_file':str(sidecar),'raw_sidecar_sha256':io.sha(sidecar),
            'response_sha256':io.sha(response)})
    return {'schema':'atlas-l1-bisect-inputs-v1','reference_source':source,
        'candidate_manifest_sha256':io.sha(Path(candidate)/'collection'/'manifest.json'),
        'reference_manifest_sha256':io.sha(Path(reference)/'collection'/'manifest.json'),
        'reference_head_provenance':heads}


def run(plan,io,lifecycle_factory=L1Lifecycle):
    validate_plan(plan,io)
    attempt=Path(plan['output'])/plan['attempt']; io.mkdir(attempt)
    results={}; shared_staged=set()
    receipt={'schema':'atlas-l1-collection-receipt-v1','scope':'diagnostic only; no qualification verdict',
             'plan_sha256':io.sha(Path(plan['output'])/'plan.json'),'windows':results,
             'observer_passed':False,'attribution_authorized':False}
    try:
        for engine in ('candidate','reference'):
            lifecycle=lifecycle_factory(plan,engine,io); lifecycle.staged=shared_staged
            for window in [w for w in plan['windows'] if w['engine']==engine]:
                root=attempt/window['name']; io.mkdir(root)
                for directory in ('diagnostic','records','logits'):
                    io.mkdir(root/directory)
                for name in ('cases.json','request-ids.json','thresholds.json'):
                    io.copy(Path(plan['output'])/name,root/name)
                io.write(root/'provenance.json',window['provenance'])
                with lifecycle.window(window,root):
                    lifecycle.collect(root)
                result={'lifecycle_sha256':io.sha(root/'lifecycle.json'),
                        'manifest_sha256':io.sha(root/'collection'/'manifest.json')}
                if engine=='candidate' and plan['candidate_policy']=='selected-treatment':
                    entries=io.json(Path(plan['output'])/'fixture'/'manifest.json')['entries']
                    # Records stay rank-local. The production verifier requires a
                    # complete rank/invocation set, so build a separate immutable view.
                    merged=root/'record-accounting'; io.mkdir(merged)
                    for rank in ('0','1'):
                        for path in io.files(root/('rank'+rank)/'records'):
                            target=merged/path.name
                            require(not target.exists(),'substitution record rank collision')
                            io.copy(path,target)
                    result['substitution_records']=locate_substitution_records(merged,entries,plan['variant'])
                if engine=='reference':
                    result['source_receipts']={rank:source_binding(root/('rank'+rank)/'diagnostic',plan['reference_source'])
                                               for rank in ('0','1')}
                results[window['name']]=result
        comparisons={engine:observer_equal(attempt/(engine+'-control')/'collection',
                                           attempt/(engine+'-observed')/'collection',io)
                     for engine in ('candidate','reference')}
        receipt['observer_rows_equal']=comparisons
        receipt['observer_passed']=all(all(rows) for rows in comparisons.values())
        require(receipt['observer_passed'],'L1 observer controls differ; attribution withheld')
        candidate,reference=attempt/'candidate-observed',attempt/'reference-observed'
        binding=build_binding(candidate,reference,plan['reference_source'],io)
        binding_path=attempt/'binding.json'; io.write(binding_path,binding)
        for rank in ('0','1'):
            registered_requests(candidate/'collection',reference/'collection',binding_path,
                                reference/('rank'+rank)/'diagnostic')
        receipt['binding_sha256']=io.sha(binding_path)
        # L1 stage/policy completeness is checked by l1_bisect, never inferred here.
        receipt['status']='collected-and-request-bound; stage/policy bisect checks still required'
    except Exception as exc:
        receipt['status']='failed'; receipt['error']=str(exc)
        raise
    finally:
        receipt['file_sha256']={str(p.relative_to(attempt)):io.sha(p) for p in io.files(attempt)}
        io.write(attempt/'receipt.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan',required=True)
    parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args(); io=LocalIO()
    require(io.sha(args.plan)==args.plan_sha256,'explicit execution plan fingerprint differs')
    run(io.json(args.plan),io)


if __name__=='__main__':
    main()
