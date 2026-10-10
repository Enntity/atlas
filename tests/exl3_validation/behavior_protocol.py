# SPDX-License-Identifier: AGPL-3.0-only
"""Strict saved OpenAI behavior envelopes; no generated code or tools executed."""
import json
import math


def unique_object(pairs):
    result=dict(pairs)
    if len(result)!=len(pairs):raise ValueError('duplicate JSON keys')
    return result


def finite(value):
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('nonfinite response')
    if isinstance(value,dict):
        for item in value.values():finite(item)
    elif isinstance(value,list):
        for item in value:finite(item)


def validate_body(body,finish,max_tokens,context_limit,thinking_budget=None):
    finite(body)
    if not isinstance(body,dict) or 'error' in body:raise ValueError('server error or invalid envelope')
    if not all(isinstance(body.get(k),str) and body[k] for k in ('id','model')):
        raise ValueError('missing response identity')
    choices=body.get('choices')
    if not isinstance(choices,list) or len(choices)!=1:raise ValueError('exactly one choice required')
    choice=choices[0]
    if choice.get('index')!=0 or choice.get('finish_reason')!=finish:raise ValueError('incomplete or wrong finish')
    message=choice.get('message',{})
    if message.get('role')!='assistant':raise ValueError('assistant envelope required')
    content=message.get('content')
    if content is not None and not isinstance(content,str):raise ValueError('content must be text or null')
    if finish=='stop' and (not content or not content.strip()):raise ValueError('visible answer missing')
    reasoning=message.get('reasoning_content',message.get('reasoning')) or ''
    if not isinstance(reasoning,str):raise ValueError('invalid reasoning field')
    if any(marker in (content or '') or marker in reasoning for marker in ('<think>','</think>','<tool_call>')):
        raise ValueError('raw reasoning/tool envelope leaked into text')
    usage=body.get('usage',{})
    for key in ('prompt_tokens','completion_tokens','total_tokens'):
        if type(usage.get(key)) is not int or usage[key]<=0:raise ValueError('positive integer usage required')
    if usage['total_tokens']!=usage['prompt_tokens']+usage['completion_tokens']:
        raise ValueError('usage total inconsistent')
    if usage['completion_tokens']>max_tokens or usage['prompt_tokens']+max_tokens>context_limit:
        raise ValueError('output or context budget exceeded')
    count=usage.get('completion_tokens_details',{}).get('reasoning_tokens')
    if type(count) is not int or not 0<=count<=usage['completion_tokens']:
        raise ValueError('reasoning usage missing or invalid')
    if reasoning and count==0:raise ValueError('reasoning field contradicts accounting')
    if thinking_budget is not None and count>thinking_budget:raise ValueError('thinking budget exceeded')
    return {'content':content,'reasoning':reasoning,'reasoning_tokens':count,
            'finish_reason':finish,'usage':usage,'accounting':{
                key:usage[key] for key in ('prompt_tokens','completion_tokens','total_tokens')}}


def validate_tool(body,name,arguments):
    choice=body['choices'][0]
    if choice.get('finish_reason')!='tool_calls':raise ValueError('tool completion required')
    calls=choice['message'].get('tool_calls')
    if not isinstance(calls,list) or len(calls)!=1:raise ValueError('exactly one tool call required')
    call=calls[0];function=call.get('function',{})
    if call.get('type')!='function' or not isinstance(call.get('id'),str) or not call['id']:
        raise ValueError('tool call identity missing')
    if function.get('name')!=name or not isinstance(function.get('arguments'),str):
        raise ValueError('wrong tool or invalid arguments type')
    parsed=json.loads(function['arguments'],object_pairs_hook=unique_object)
    if parsed!=arguments:raise ValueError('tool arguments crossed or changed')
    return call['id']


def decode_sse(raw):
    events=[];done=False;data=[]
    for line in raw.splitlines()+['']:
        if line.startswith('data:'):
            if done:raise ValueError('event after DONE')
            data.append(line[5:].lstrip(' '))
        elif not line and data:
            value='\n'.join(data);data=[]
            if value=='[DONE]':done=True;continue
            event=json.loads(value,object_pairs_hook=unique_object);finite(event)
            if not isinstance(event,dict) or 'error' in event:raise ValueError('SSE server error')
            events.append(event)
    if not done or not events:raise ValueError('incomplete SSE lifecycle')
    identity={k:events[0].get(k) for k in ('id','model')}
    if not all(isinstance(v,str) and v for v in identity.values()):raise ValueError('SSE identity missing')
    content=[];reasoning=[];calls={};finishes=[];usages=[];role=None
    for event in events:
        if any(event.get(k)!=v for k,v in identity.items()):raise ValueError('SSE identity changed')
        if event.get('usage'):usages.append(event['usage'])
        for choice in event.get('choices',[]):
            if choice.get('index')!=0:raise ValueError('unexpected SSE choice')
            delta=choice.get('delta') or {}
            if delta.get('role'):
                if role not in (None,delta['role']):raise ValueError('SSE role changed')
                role=delta['role']
            if delta.get('content') is not None:
                if not isinstance(delta['content'],str):raise ValueError('invalid SSE text')
                content.append(delta['content'])
            reason=delta.get('reasoning_content',delta.get('reasoning'))
            if reason is not None:
                if not isinstance(reason,str):raise ValueError('invalid SSE reasoning')
                reasoning.append(reason)
            for fragment in delta.get('tool_calls') or []:
                index=fragment.get('index')
                if type(index) is not int or index<0:raise ValueError('invalid streamed tool index')
                call=calls.setdefault(index,{'id':'','type':'function','function':{'name':'','arguments':''}})
                if fragment.get('id'):
                    if call['id'] and call['id']!=fragment['id']:raise ValueError('streamed tool ID changed')
                    call['id']=fragment['id']
                if fragment.get('type') not in (None,'function'):raise ValueError('invalid streamed tool type')
                for key in ('name','arguments'):
                    part=(fragment.get('function') or {}).get(key)
                    if part is not None:
                        if not isinstance(part,str):raise ValueError('invalid streamed function field')
                        call['function'][key]+=part
            if choice.get('finish_reason'):finishes.append(choice['finish_reason'])
    if role!='assistant' or len(finishes)!=1 or len(usages)!=1:raise ValueError('SSE role/finish/usage lifecycle incomplete')
    message={'role':role,'content':''.join(content) if content else (None if calls else '')}
    if reasoning:message['reasoning_content']=''.join(reasoning)
    if calls:
        if sorted(calls)!=list(range(len(calls))):raise ValueError('streamed tool index gap')
        message['tool_calls']=[calls[i] for i in sorted(calls)]
    return {**identity,'choices':[{'index':0,'finish_reason':finishes[0],'message':message}],
            'usage':usages[0]},events
