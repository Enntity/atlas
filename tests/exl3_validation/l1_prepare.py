# SPDX-License-Identifier: AGPL-3.0-only
"""Offline source-bound L1 preparation. Every external input needs an explicit hash."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

from c2_remote_adapter import RemoteAdapter, require
from collect_logits import validate_provenance
from l1_shutdown import validate_policy
from make_norm_fixtures import validate_manifest
from parity_protocol import validate_cases, read_raw, load_artifact


OBSERVER_KEYS = {'ATLAS_EXL3_NORM_SUBSTITUTE_DIR','ATLAS_EXL3_NORM_SUBSTITUTE_RECORD_DIR',
    'ATLAS_EXL3_LOGITS_DIR','ATLAS_EXL3_FFN_DUMP_DIR','ATLAS_EXL3_FFN_DUMP_LAYERS',
    'ATLAS_EXL3_UPSTREAM_DUMP_DIR','ATLAS_EXL3_UPSTREAM_DUMP_LAYERS',
    'ATLAS_EXL3_MHC_DUMP_LAYERS','ATLAS_EXL3_CHUNK_CAPTURE','ATLAS_EXL3_CHUNK_RUNNER_SHA256',
    'ATLAS_EXL3_ATTN_DUMP_DIR','ATLAS_EXL3_ATTN_DUMP_LAYERS',
    'ATLAS_EXL3_MLA_DUMP_DIR','ATLAS_EXL3_MLA_DUMP_LAYERS'}


def analysis_equivalent(registered,recomputed):
    values=[]
    for report in (registered,recomputed):
        value=copy.deepcopy(report)
        pairs=value['repeat_pair_scores']
        require(isinstance(pairs,list) and len(pairs)==2, 'two C2 repeat-pair evaluations required')
        for pair in pairs:
            require(isinstance(pair['evaluated_utc'],str) and pair['evaluated_utc'],
                    'explicit scorer evaluation timestamp required')
            del pair['evaluated_utc']
        values.append(value)
    return values[0]==values[1]


class LocalIO:
    """Local evidence boundary; preparation never invokes remote commands."""
    def read(self, path):
        path = Path(path)
        require(path.is_file() and not path.is_symlink(), 'regular source file required: '+str(path))
        return path.read_bytes()

    def json(self, path):
        return json.loads(self.read(path))

    def sha(self, path):
        return hashlib.sha256(self.read(path)).hexdigest()

    def write(self, path, value):
        Path(path).write_text(json.dumps(value, indent=2)+'\n')

    def copy(self, source, target):
        shutil.copyfile(source, target)

    def copy_tree(self, source, target):
        shutil.copytree(source, target)

    def mkdir(self, path):
        Path(path).mkdir(parents=True, exist_ok=False)

    def artifact(self, path):
        return load_artifact(path)

    def raw(self, path):
        return read_raw(path)

    def verify_c2_analysis(self, root, attempt, registered):
        analyzer = Path(__file__).parents[2]/'docs/design/evidence/c2-execution-20261014/analyze.py'
        require(self.sha(analyzer)==registered['analyzer_sha256'], 'registered C2 analyzer source changed')
        with tempfile.TemporaryDirectory(prefix='atlas-l1-c2-recheck-') as directory:
            report=Path(directory)/'analysis.json'
            subprocess.run([sys.executable,str(analyzer),'--root',str(root),'--attempt',str(attempt),
                            '--out',str(report)],check=True,capture_output=True)
            require(analysis_equivalent(registered,self.json(report)), 'completed C2 analysis does not recompute')

    def files(self, path):
        return sorted(p for p in Path(path).rglob('*') if p.is_file())


def render_descriptor(original, *, engine, image_tag, image_id, name, root, fixture, overlay,
                      runner_sha256):
    """Pure transformation: numerical env/command preserved; observers re-registered."""
    require(engine in ('candidate', 'reference'), 'unknown engine')
    require(original[:2] == ['docker', 'create'] and '--rm' not in original,
            'explicit durable docker create required')
    require(original.count(image_tag) == 1, 'descriptor image must occur exactly once')
    index = original.index(image_tag)
    require(original[index+1] == ('serve' if engine == 'candidate' else '/start.sh'),
            'registered engine entry command differs')
    require(original.count('--name') == 1, 'exactly one container name required')
    root = Path(root)
    env = {'ATLAS_EXL3_LOGITS_DIR': str(root/'logits'),
           'ATLAS_EXL3_UPSTREAM_DUMP_DIR': str(root/'diagnostic'),
           'ATLAS_EXL3_UPSTREAM_DUMP_LAYERS': '0,1',
           'ATLAS_EXL3_FFN_DUMP_DIR': str(root/'diagnostic'),
           'ATLAS_EXL3_FFN_DUMP_LAYERS': '0,1'}
    if engine == 'candidate':
        if fixture is not None:
            env.update(ATLAS_EXL3_NORM_SUBSTITUTE_DIR=str(fixture),
                   ATLAS_EXL3_NORM_SUBSTITUTE_RECORD_DIR=str(root/'records'))
    else:
        env.update(ATLAS_EXL3_MHC_DUMP_LAYERS='0,1', ATLAS_EXL3_CHUNK_CAPTURE='serial128',
                   ATLAS_EXL3_CHUNK_RUNNER_SHA256=runner_sha256,
                   PYTHONPATH='/opt/atlas-logits-patch')
    observer_keys = OBSERVER_KEYS
    args, pos = [], 0
    removed_mounts = {'/tmp/diagnostic','/tmp/capture','/tmp/reference-logits','/opt/atlas-logits-patch'}
    while pos < index:
        flag = original[pos]
        if flag in ('--env','-e','--volume','-v'):
            value = original[pos+1]
            if flag in ('--env','-e'):
                key = value.split('=',1)[0]
                if key in observer_keys or key == 'PYTHONPATH' and engine == 'reference':
                    pos += 2
                    continue
            elif value.split(':')[1] in removed_mounts:
                pos += 2
                continue
            args.extend([flag,value]); pos += 2
        elif flag == '--name':
            args.extend([flag,name]); pos += 2
        else:
            args.append(flag); pos += 1
    for key,value in sorted(env.items()):
        args.extend(['--env',key+'='+value])
    args.extend(['--volume',str(root)+':'+str(root)])
    if engine == 'candidate' and fixture is not None:
        args.extend(['--volume',str(fixture)+':'+str(fixture)+':ro'])
    elif engine == 'reference':
        args.extend(['--volume',str(overlay)+':/opt/atlas-logits-patch:ro'])
    return args+[image_id]+original[index+1:]


def injected_provenance(original,commands):
    provenance=copy.deepcopy(original)
    rank_env={}
    for rank,args in commands.items():
        rank_env[rank]={args[index+1].split('=',1)[0]:args[index+1].split('=',1)[1]
                        for index,arg in enumerate(args) if arg in ('--env','-e')}
    provenance['environment']={key:value for key,value in provenance['environment'].items() if key not in OBSERVER_KEYS}
    provenance['environment'].update(rank_env['0'])
    provenance['rank_environments']=rank_env
    return provenance


def prepare(config, io):
    keys = {'schema','output','attempt','c2_preparation','c2_plan_sha256','variant',
            'reference_overlay','reference_overlay_sha256','reference_source','engines','input_sha256','candidate_policy',
            'evidence_root','c2_analysis','c2_analysis_sha256','preserved_attempts'}
    require(set(config) == keys and config['schema'] == 'atlas-l1-preparation-config-v1',
            'explicit L1 configuration keys required')
    output = Path(config['output'])
    require(output.is_absolute() and re.fullmatch('[a-z0-9][a-z0-9-]{0,55}',config['attempt']),
            'absolute fresh root and Docker-safe attempt name required')
    for path,digest in config['input_sha256'].items():
        require(Path(path).is_absolute() and io.sha(path) == digest, 'input binding differs: '+path)
    require(isinstance(config['preserved_attempts'],list), 'explicit preserved attempt list required')
    for prior in config['preserved_attempts']:
        require(set(prior)=={'attempt','file_sha256'} and Path(prior['attempt']).is_absolute(),
                'preserved failed attempt binding required')
        require(io.json(Path(prior['attempt'])/'receipt.json')['status']=='failed', 'prior failed status must remain failed')
        for relative,digest in prior['file_sha256'].items():
            require(io.sha(Path(prior['attempt'])/relative)==digest, 'preserved failed attempt changed')
    c2root = Path(config['c2_preparation'])
    registration = io.json(c2root/'execution-plan.json')
    require(io.sha(c2root/'execution-plan.json') == config['c2_plan_sha256']
            and registration['preparation_root'] == str(c2root.resolve()), 'C2 source registration differs')
    require(config['candidate_policy'] in ('baseline','selected-treatment'), 'explicit candidate policy required')
    require(config['variant'] in registration['variants'], 'selected C2 variant not registered')
    used = ['cases.json','request-ids.json','thresholds.json','provenance.json','run-config.json',
            config['variant']+'/manifest.json']
    for name in used:
        require(io.sha(c2root/name) == registration['file_sha256'][name], 'registered C2 input differs: '+name)
    analysis=io.json(config['c2_analysis'])
    require(io.sha(config['c2_analysis'])==config['c2_analysis_sha256']
            and analysis['schema']=='atlas-c2-execution-analysis-v1'
            and analysis['execution_binding']==registration
            and analysis['execution_binding_sha256']==config['c2_plan_sha256']
            and analysis['interpretation']['causal_interpretation_allowed'] is True
            and analysis['interpretation']['status']=='diagnostic-intervention-bound'
            and analysis['scoring_skipped_reason'] is None, 'completed C2 analysis/first controls required')
    c2attempt=Path(analysis['attempt'])
    require(c2attempt.parent==c2root and io.sha(c2attempt/'receipt.json')==analysis['receipt_sha256'],
            'completed C2 receipt binding differs')
    c2receipt=io.json(c2attempt/'receipt.json')
    require(c2receipt['variant']==config['variant'] and c2receipt['repeats_per_arm']==2
            and set(c2receipt['arms'])=={'a','b'} and all(len(runs)==2 for runs in c2receipt['arms'].values()),
            'selected variant must have completed analyzed C2 campaign')
    io.verify_c2_analysis(config['evidence_root'],c2attempt,analysis)
    proof = io.json(registration['runtime_neutrality_receipt'])
    require(io.sha(registration['runtime_neutrality_receipt']) == registration['runtime_neutrality_sha256']
            and proof['schema']=='atlas-c2-full-neutrality-proof-v1'
            and proof['passed'] is True and proof['rows_total']==32
            and proof['runtime']==io.json(c2root/'provenance.json')['engine_build'],
            'registered runtime neutrality proof differs')
    fixture = validate_manifest(c2root/config['variant'])
    cases = validate_cases(io.json(c2root/'cases.json'))
    ids = io.json(c2root/'request-ids.json')
    rows = [(case['id'],step,len(case['prompt_ids'])+step) for case in cases
            for step in range(len(case['continuation_ids'])+1)]
    require(len(rows)==len(ids)==8 and all(n<=128 for _,_,n in rows), 'exact eight cold serial128 rows required')
    for rank in (0,1):
        entries = sorted((e for e in fixture['entries'] if e['tp_rank']==rank), key=lambda e:e['invocation'])
        require([e['request_id'] for e in entries] == ids, 'fixture request order differs')
        for entry in entries:
            relative = config['variant']+'/'+entry['file']
            require(io.sha(c2root/relative) == registration['file_sha256'][relative], 'fixture registration differs')
    overlay = Path(config['reference_overlay'])
    package = io.json(overlay/'package-binding.json')
    require(io.sha(overlay/'package-binding.json') == config['reference_overlay_sha256']
            and package['schema'] == 'atlas-upstream-hook-package-context-v5', 'v5 overlay binding required')
    require({'reference_ffn_trace.py','reference_mhc_trace.py','reference_kda_trace.py',
             'reference_chunk_trace.py','sitecustomize.py'} <= set(package['package_file_sha256']),
            'complete v5 producer package required')
    for name,digest in package['package_file_sha256'].items():
        require(io.sha(overlay/name)==digest, 'overlay package differs: '+name)
    source = config['reference_source']
    require(all(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest) for digest in source.values()),
            'full reference source hashes required')
    require(set(source)=={'model_sha256','ffn_hook_sha256','runner_sha256'}
            and source['ffn_hook_sha256']==package['package_file_sha256']['reference_ffn_trace.py'],
            'reference source binding differs')
    require(set(config['engines'])=={'candidate','reference'}, 'both engines required')
    for engine,settings in config['engines'].items():
        required = {'descriptors','image_tag','image_id','provenance','remote','endpoint','model',
                    'timeout_seconds','container_source_sha256','remote_mount_sha256','shutdown'}
        validate_policy(settings['shutdown'],engine)
        if settings['shutdown']['worker_exit_policy']=='controlled-headless137':
            paths=settings['container_source_sha256']
            require(any(path.endswith('/entrypoints/cli/serve.py') for path in paths)
                    and any(path.endswith('/v1/executor/multiproc_executor.py') for path in paths),
                    'headless CLI/executor source guards required for worker shutdown policy')
        require(set(settings)==required, 'explicit engine configuration required: '+engine)
        require(set(settings['descriptors'])==set(settings['remote_mount_sha256'])=={'0','1'}, 'both ranks required')
        provenance = io.json(settings['provenance']); validate_provenance(provenance)
        require(provenance['engine']==('atlas' if engine=='candidate' else 'vllm')
                and provenance['engine_build']['image']==settings['image_id'], 'engine provenance/image differs')
        require(settings['remote']['request_ids']==ids
                and all(v==settings['image_id'] for v in settings['remote']['image_ids'].values()), 'remote schedule/image differs')
        for checks in settings['remote']['provider_checks'].values():
            for command in checks.values():
                require('--pins' in command, 'explicit provider pins required')
                require(command[command.index('--pins')+1] in config['input_sha256'], 'provider pins source unbound')
                if engine=='reference':
                    require('--checker-sha256' in command
                        and command[command.index('--checker-sha256')+1]==settings['remote']['remote_harness_sha256']['attest_runtime_maps.py'],
                        'reference provider checker source differs')
        RemoteAdapter(settings['remote'])  # validates command/schema/source hashes without I/O
        endpoint=urlsplit(settings['endpoint'])
        require(endpoint.scheme=='http' and endpoint.hostname in ('localhost','127.0.0.1','::1')
                and endpoint.port is not None and endpoint.username is None and endpoint.password is None
                and endpoint.path in ('','/') and not endpoint.query and not endpoint.fragment,
                'collection endpoint must be explicit local HTTP host/port')
        require(settings['endpoint'].rstrip('/')+'/v1/models' in settings['remote']['readiness_command'],
                'readiness endpoint must match collection endpoint')
        require(settings['timeout_seconds']>0, 'collection timeout required')
        if engine=='candidate':
            registered_config=io.json(c2root/'run-config.json')
            require(all(settings[key]==registered_config[key] for key in ('endpoint','model','timeout_seconds')),
                    'candidate collection differs from registered C2 profile')
            require(provenance['engine_build']==io.json(c2root/'provenance.json')['engine_build']
                    and settings['image_id']==registration['runtime_image_id'], 'candidate runtime must match selected C2')
            require(provenance['engine_build']['binary_sha256'] in settings['container_source_sha256'].values(),
                    'candidate actual binary check required')
        else:
            reference_args=io.json(settings['descriptors']['0'])
            environment={reference_args[index+1].split('=',1)[0]:reference_args[index+1].split('=',1)[1]
                         for index,arg in enumerate(reference_args) if arg in ('--env','-e')}
            require(settings['model']==environment['SERVED_MODEL_NAME'] and endpoint.port==int(environment['PORT']),
                    'reference collection model/port differs from registered startup profile')
            require(source['model_sha256'] in settings['container_source_sha256'].values()
                    and source['runner_sha256'] in settings['container_source_sha256'].values(),
                    'reference installed model/runner checks required')
        for path in [settings['provenance'], *settings['descriptors'].values()]:
            require(path in config['input_sha256'], 'unbound engine input: '+path)
        for rank,path in settings['descriptors'].items():
            if engine=='candidate':
                require(io.sha(path)==registration['file_sha256']['candidate-rank'+rank+'-create-command.json'],
                        'candidate descriptor differs from selected C2 profile')
            args=io.json(path)
            for index,arg in enumerate(args):
                if arg in ('--volume','-v'):
                    host=args[index+1].split(':')[0]
                    if host.endswith(('.py','.sh','.jinja')):
                        require(host in settings['remote_mount_sha256'][rank], 'mounted executable source unbound: '+host)
    io.mkdir(output)
    io.copy_tree(overlay,output/'overlay'); io.copy_tree(c2root/config['variant'],output/'fixture')
    io.mkdir(output/'tooling')
    for name in ('collect_logits.py','parity_protocol.py','attest_runtime_maps.py',
                 'l1_reference_attest.py','c2_remote_adapter.py'):
        io.copy(Path(__file__).parent/name,output/'tooling'/name)
    for name in ('cases.json','request-ids.json','thresholds.json'):
        io.copy(c2root/name,output/name)
    plan=copy.deepcopy(config); plan['schema']='atlas-l1-collection-plan-v1'; plan['windows']=[]
    for engine,settings in plan['engines'].items():
        io.copy(settings['provenance'],output/(engine+'-provenance.json'))
        old_harness = Path(settings['remote']['remote_harness_dir'])
        settings['remote']['remote_harness_dir'] = str(output/'tooling')
        pins_files=set()
        for checks in settings['remote']['provider_checks'].values():
            for command in checks.values():
                require('--pins' in command, 'explicit provider pins required')
                pins_files.add(command[command.index('--pins')+1])
                if engine=='reference':
                    require(str(old_harness/'l1_reference_attest.py') in command
                            and '--checker-sha256' in command and '--cmdline-pattern' in command
                            and '--proc-root' in command, 'reference worker attester with explicit selection required')
        require(len(pins_files)==1, 'one explicit provider pin policy per engine required')
        pins_path=pins_files.pop()
        require(pins_path in config['input_sha256'], 'provider pins source unbound')
        new_pins=output/'tooling'/(engine+'-pins.sha256'); io.copy(pins_path,new_pins)
        for checks in settings['remote']['provider_checks'].values():
            for provider,command in checks.items():
                checks[provider]=[str(output/'tooling'/Path(arg).name) if arg in
                    [str(old_harness/'attest_runtime_maps.py'),str(old_harness/'l1_reference_attest.py')]
                    else str(new_pins) if arg==pins_path else arg for arg in command]
        for observed in (False,True):
            window=engine+('-observed' if observed else '-control')
            commands={}
            for rank,path in settings['descriptors'].items():
                commands[rank]=render_descriptor(io.json(path),engine=engine,image_tag=settings['image_tag'],
                    image_id=settings['image_id'],name='atlas-l1-'+config['attempt']+'-'+window+'-r'+rank,
                    root=output/config['attempt']/window,fixture=(output/'fixture' if config['candidate_policy']=='selected-treatment' else None),overlay=output/'overlay',
                    runner_sha256=source['runner_sha256'])
            provenance=injected_provenance(io.json(settings['provenance']),commands)
            plan['windows'].append({'name':window,'engine':engine,'observed':observed,'commands':commands,'provenance':provenance})
    plan['prepared_sha256']={str(p.relative_to(output)):io.sha(p) for p in io.files(output)}
    plan['module_sha256']={str(Path(__file__).parent/name):io.sha(Path(__file__).parent/name)
                           for name in ('l1_prepare.py','l1_lifecycle.py','l1_collect.py','l1_binding.py','c2_remote_adapter.py','l1_reference_attest.py','l1_shutdown.py')}
    io.write(output/'plan.json',plan)
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',required=True)
    args=parser.parse_args(); io=LocalIO(); prepare(io.json(args.config),io)


if __name__=='__main__':
    main()
