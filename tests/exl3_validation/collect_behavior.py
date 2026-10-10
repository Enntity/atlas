# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded native TP2 HTTP behavior gates, without performance measurements."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import threading
from behavior_io import BehaviorIORouter
from behavior_protocol import validate_body,validate_tool
from parity_protocol import validate_provenance

MODEL_PROMPT='What is the capital of Japan? Reply with only Tokyo and no punctuation.'
MATH_PROMPTS=('Calculate 37 * 14 - 86. Reply only with the integer answer.',
              'Calculate 91 - 38. Reply only with the integer answer.',
              'Calculate 12 * 13. Reply only with the integer answer.',
              'Calculate 144 / 12. Reply only with the integer answer.')
MATH_ANSWERS=('432','53','156','12')
TOOL_NAME='lookup_inventory';TOOL_ARGUMENTS={'sku':'BLUE-17','warehouse':'north'}
TOOL={'type':'function','function':{'name':TOOL_NAME,'description':'Look up inventory.',
      'parameters':{'type':'object','properties':{'sku':{'type':'string'},'warehouse':{'type':'string'}},
                    'required':['sku','warehouse'],'additionalProperties':False}}}


def flag_value(flags,name):
    values=[flags[i+1] for i,flag in enumerate(flags[:-1]) if flag==name]
    values += [flag.split('=',1)[1] for flag in flags if flag.startswith(name+'=')]
    if len(values)!=1:raise ValueError('one explicit server flag required: '+name)
    return values[0]


def collect(base_url,model,provenance,out,context_limit,concurrency,timeout):
    validate_provenance(provenance)
    if provenance['engine']!='atlas':raise ValueError('behavior collector targets Atlas; no reference parity inferred')
    if context_limit!=4096 or type(concurrency) is not int or not 1<=concurrency<=4:
        raise ValueError('explicit context4096 and simultaneous HTTP width1..4 required')
    for name,expected in [('--max-seq-len','4096'),('--max-prefill-tokens','128'),
                          ('--max-batch-size','1'),('--max-num-seqs','1')]:
        if flag_value(provenance['serve_flags'],name)!=expected:
            raise ValueError('behavior profile requires '+name+' '+expected)
    io=BehaviorIORouter(base_url,out,timeout)
    scope='simultaneous HTTP; serial C1 GPU admission'
    graph={'status':'SKIP_UNQUALIFIED','reason':'This profile explicitly runs eager. TP2 graph opt-in is not qualified; configured FFN diagnostics reject graph capture. No graph execution or equivalence is claimed.'}
    manifest={'provenance':provenance,'context_limit':context_limit,'concurrency':concurrency,
              'concurrency_scope':scope,'graph':graph,'source_sha256':io.sources(),'requests':[],
              'scope':'Bounded HTTP behavior, not throughput, model quality certification, or vLLM parity.'}
    io.write('plan.json',{**manifest,'request_policy':{'temperature':0,'seed':42,'max_tokens':256,
             'thinking_budget':32,'no_external_tools':True},'thinking_off_contract':
             'GLM template intentionally retains native low-effort reasoning without tools; no empty-think override or requirement that off suppress all internal reasoning.'})

    def payload(prompt,thinking):
        value={'model':model,'messages':[{'role':'user','content':prompt}],
               'temperature':0,'seed':42,'max_tokens':256,'stream':False}
        if thinking is not None:
            value['chat_template_kwargs']={'enable_thinking':thinking}
        if thinking is True:value['thinking_token_budget']=32
        return value

    def run(name,task,answer=None,finish='stop',barrier=None):
        if barrier is not None:barrier.wait(timeout=10)
        receipt,body=io.request(name,task)
        if body is not None:
            try:
                summary=validate_body(body,finish,task['max_tokens'],context_limit,
                                      task.get('thinking_token_budget'))
                if answer is not None and summary['content'].strip()!=answer:
                    raise ValueError('wrong visible answer')
                if task.get('chat_template_kwargs',{}).get('enable_thinking') is True and not summary['reasoning']:
                    raise ValueError('thinking-on request did not exercise reasoning channel')
                receipt['summary']=summary;receipt['passed']=True
            except Exception as error:receipt['error']=type(error).__name__+': '+str(error)
        return receipt,body

    ordinary,_=run('ordinary-chat',payload(MODEL_PROMPT,None),'Tokyo');manifest['requests'].append(ordinary)
    off_task=payload(MODEL_PROMPT,False)
    off,_=run('thinking-off',off_task,'Tokyo');manifest['requests'].append(off)
    on,_=run('thinking-on',payload(MATH_PROMPTS[0],True),'432');manifest['requests'].append(on)
    sse_task={**off_task,'stream':True,'stream_options':{'include_usage':True}}
    sse,_=run('stream-chat',sse_task,'Tokyo')
    if sse['passed'] and off['passed']:
        sse['blocking_identity']={key:sse['summary'][key]==off['summary'][key] for key in
                                 ('content','reasoning','reasoning_tokens','finish_reason','accounting')}
        if not all(sse['blocking_identity'].values()):
            sse['passed']=False;sse['error']='isolated blocking/SSE identity mismatch; retained as behavior failure, not normalized away'
    else:sse['passed']=False;sse['error']=sse['error'] or 'blocking comparison prerequisite failed'
    manifest['requests'].append(sse)

    tool_task=payload('Use lookup_inventory exactly once for SKU BLUE-17 in warehouse north. Wait for the tool result; then reply only with the integer available count.',False)
    tool_task.update({'tools':[TOOL],'tool_choice':{'type':'function','function':{'name':TOOL_NAME}}})
    tool,tool_body=run('tool-call',tool_task,finish='tool_calls')
    tool_id=None
    if tool['passed']:
        try:tool_id=validate_tool(tool_body,TOOL_NAME,TOOL_ARGUMENTS)
        except Exception as error:tool['passed']=False;tool['error']=str(error)
    manifest['requests'].append(tool)
    automatic,automatic_body=run('auto-tool-call',{**tool_task,'tool_choice':'auto'},finish='tool_calls')
    if automatic['passed']:
        try:validate_tool(automatic_body,TOOL_NAME,TOOL_ARGUMENTS)
        except Exception as error:automatic['passed']=False;automatic['error']=str(error)
    manifest['requests'].append(automatic)
    stream_tool={**tool_task,'stream':True,'stream_options':{'include_usage':True}}
    streamed,streamed_body=run('stream-tool',stream_tool,finish='tool_calls')
    if streamed['passed']:
        try:validate_tool(streamed_body,TOOL_NAME,TOOL_ARGUMENTS)
        except Exception as error:streamed['passed']=False;streamed['error']=str(error)
    manifest['requests'].append(streamed)
    if tool_id is not None:
        followup={**tool_task,'tool_choice':'none','messages':tool_task['messages']+[
            tool_body['choices'][0]['message'],{'role':'tool','tool_call_id':tool_id,
            'content':json.dumps({**TOOL_ARGUMENTS,'available':19})}]}
        final,_=run('tool-followup',followup,'19');manifest['requests'].append(final)
        if final['passed'] and final['response']['choices'][0]['message'].get('tool_calls'):
            final['passed']=False;final['error']='unexpected tool call after tool_choice none'
    else:manifest['requests'].append({'id':'tool-followup','passed':False,'error':'valid initial tool call missing; no fabricated tool ID'})

    barrier=threading.Barrier(concurrency)
    def queued(index):
        return run('queued-'+str(index),payload(MATH_PROMPTS[index],True),MATH_ANSWERS[index],barrier=barrier)[0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        manifest['requests'].extend(pool.map(queued,range(concurrency)))
    manifest['passed']=all(row['passed'] for row in manifest['requests'])
    io.write('manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('base-url','model','provenance','out','context-limit','concurrency','timeout'):
        parser.add_argument('--'+flag,required=True)
    args=parser.parse_args()
    result=collect(args.base_url,args.model,json.loads(Path(args.provenance).read_text()),
                   Path(args.out),int(args.context_limit),int(args.concurrency),float(args.timeout))
    print(json.dumps({'passed':result['passed'],'requests':len(result['requests']),
                     'graph':result['graph'],'concurrency_scope':result['concurrency_scope']}))
    raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
