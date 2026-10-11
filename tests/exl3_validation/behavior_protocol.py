# SPDX-License-Identifier: AGPL-3.0-only
"""Strict saved OpenAI behavior envelopes; no generated code or tools executed."""
import json
import math
import re

# Special-token and tool-envelope prefixes that must never surface in a
# visible channel. Scanned as substrings: '<think' also catches '<thinking>'.
# Sources: api/strip.rs MARKERS (think/analysis tags), poolside_v1 tool
# envelope (tool_call/arg_key/arg_value), and the GLM template specials.
LEAK_MARKERS = ('<think', '</think', '<analysis', '</analysis',
                '<tool_call', '</tool_call', '<arg_key', '</arg_key',
                '<arg_value', '</arg_value', '<tool_response', '</tool_response',
                '<function', '<parameter', '[gMASK]', '<sop>')
LEAK_SPECIAL = re.compile(r'<\|[^|]*\|>')
# The budget only ARMS the forced </think>; scheduler/confidence.rs
# should_inject_think_end then defers it up to MAX_SENTENCE_DEFER_TOKENS (64)
# for a sentence boundary, and </think> itself is counted. Budget + 65 is the
# largest reasoning count the server can legitimately report outside a code
# fence (the math prompts never open one).
THINKING_OVERSHOOT = 64 + 1


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


def assert_no_special_tokens(text,channel):
    if not isinstance(text,str):raise ValueError(channel+' must be text')
    for marker in LEAK_MARKERS:
        if marker in text:raise ValueError('special token leaked into '+channel+': '+repr(marker))
    if LEAK_SPECIAL.search(text):raise ValueError('special token leaked into '+channel+': <|...|>')


def check_usage(body,max_tokens,context_limit):
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
    return usage,count


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
    assert_no_special_tokens(content or '','content')
    assert_no_special_tokens(reasoning,'reasoning')
    usage,count=check_usage(body,max_tokens,context_limit)
    if reasoning and count==0:raise ValueError('reasoning field contradicts accounting')
    if thinking_budget is not None and count>thinking_budget+THINKING_OVERSHOOT:
        raise ValueError('thinking budget exceeded beyond the sentence-boundary deferral')
    return {'content':content,'reasoning':reasoning,'reasoning_tokens':count,
            'finish_reason':finish,'usage':usage,'accounting':{
                key:usage[key] for key in ('prompt_tokens','completion_tokens','total_tokens')}}


def validate_completion(body,finish,max_tokens,context_limit):
    """Legacy /v1/completions envelope: text channel plus the same accounting
    and special-token gates as the chat surface."""
    finite(body)
    if not isinstance(body,dict) or 'error' in body:raise ValueError('server error or invalid envelope')
    if not all(isinstance(body.get(k),str) and body[k] for k in ('id','model')):
        raise ValueError('missing response identity')
    if body.get('object')!='text_completion':raise ValueError('legacy completion object required')
    choices=body.get('choices')
    if not isinstance(choices,list) or len(choices)!=1:raise ValueError('exactly one choice required')
    choice=choices[0]
    if choice.get('index')!=0 or choice.get('finish_reason')!=finish:raise ValueError('incomplete or wrong finish')
    text=choice.get('text')
    if not isinstance(text,str):raise ValueError('completion text must be text')
    if finish=='stop' and not text.strip():raise ValueError('visible answer missing')
    assert_no_special_tokens(text,'completion text')
    usage,count=check_usage(body,max_tokens,context_limit)
    return {'text':text,'finish_reason':finish,'usage':usage,'reasoning_tokens':count,
            'accounting':{key:usage[key] for key in ('prompt_tokens','completion_tokens','total_tokens')}}


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


def parse_sse_events(raw):
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
    return events


def decode_sse(raw):
    events=parse_sse_events(raw)
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


def decode_completions_sse(raw):
    """Rebuild one legacy /v1/completions body from its SSE stream.

    Chunks carry `choices[0].text` deltas and the terminal framing places the
    usage either on the finish chunk or on a separate `choices: []` chunk
    (stream_options.include_usage); exactly one of each is required."""
    events=parse_sse_events(raw)
    identity={k:events[0].get(k) for k in ('id','model')}
    if not all(isinstance(v,str) and v for v in identity.values()):raise ValueError('SSE identity missing')
    text=[];finishes=[];usages=[]
    for event in events:
        if any(event.get(k)!=v for k,v in identity.items()):raise ValueError('SSE identity changed')
        if event.get('object')!='text_completion':raise ValueError('SSE completion object missing')
        if event.get('usage'):usages.append(event['usage'])
        for choice in event.get('choices',[]):
            if choice.get('index')!=0:raise ValueError('unexpected SSE choice')
            if choice.get('text') is not None:
                if not isinstance(choice['text'],str):raise ValueError('invalid SSE text')
                text.append(choice['text'])
            if choice.get('finish_reason'):finishes.append(choice['finish_reason'])
    if len(finishes)!=1 or len(usages)!=1:raise ValueError('SSE finish/usage lifecycle incomplete')
    return {**identity,'object':'text_completion',
            'choices':[{'index':0,'text':''.join(text),'finish_reason':finishes[0]}],
            'usage':usages[0]},events


def sse_shape(events):
    """Compact per-event kinds of an OpenAI-style stream: 'usage' for a
    choices:[] usage chunk, 'finish' for a chunk carrying finish_reason,
    'delta' otherwise. Used to gate include_usage ordering separately from
    content validation."""
    shape=[]
    for event in events:
        choices=event.get('choices') or []
        if not choices and event.get('usage'):shape.append('usage')
        elif any(choice.get('finish_reason') for choice in choices):shape.append('finish')
        else:shape.append('delta')
    return shape


def usage_follows_finish(shape):
    """OpenAI include_usage framing: the usage-only chunk is the last data
    event, after the single finish chunk."""
    return shape.count('usage')==1 and shape.count('finish')==1 and shape[-1]=='usage'
