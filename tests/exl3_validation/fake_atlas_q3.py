# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only fake of the Atlas Q3 surfaces for the behavior unit tests.

Never a qualification target: it mirrors the server contract the behavior
gates assert (spec-correct by default) and injects one failure per mode:
  fenced          structured output wrapped in a code fence
  bad-args        parallel get_weather calls carry an out-of-enum unit
  tool-reasoning  tool turns report reasoning despite thinking_in_tools=false
  anthropic-drift streamed stop-sequence hit reported as end_turn/null
                  (what anthropic/translator.rs does today)
  leaky           the first request after an interruption answers wrongly
  gauge           /metrics reports atlas_requests_active 1
  not-ready       /health reports {"status":"loading"} with 503
"""
import json
import time

from behavior_messages import STOP_MARKER as MSG_STOP
from behavior_robustness import LONG_PROMPT as INTERRUPT_PROMPT
from behavior_tools import CITY_PROMPT, CITY_VALUE, NO_TOOL_PROMPT, OBJECT_PROMPT, PARALLEL_PROMPT

SLOW_SECONDS = 1.5


def _json(status, value):
    return status, 'application/json', json.dumps(value).encode()


def _sse(events, mode):
    raw = ''.join('data: ' + json.dumps(e) + '\n\n' for e in events)
    if mode != 'broken-stream':
        raw += 'data: [DONE]\n\n'
    return 200, 'text/event-stream', raw.encode()


def q3_get(path, mode, calls):
    if path == '/health':
        if mode == 'not-ready':
            return _json(503, {'status': 'loading'})
        return _json(200, {'status': 'ready', 'model': 'test'})
    if path == '/metrics':
        value = 1 if mode == 'gauge' else 0
        text = '# TYPE atlas_requests_active gauge\natlas_requests_active ' + str(value) + '\n'
        return 200, 'text/plain', text.encode()
    return _json(404, {'error': 'not found'})


def _usage(completion, reasoning=0):
    return {'prompt_tokens': 7, 'completion_tokens': completion, 'total_tokens': 7 + completion,
            'completion_tokens_details': {'reasoning_tokens': reasoning}}


def _chat_body(model, message, finish, usage):
    return {'id': 'chat-q3', 'object': 'chat.completion', 'model': model,
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish}], 'usage': usage}


def _chat_stream(model, message, finish, usage, mode):
    def chunk(delta, fin=None):
        return {'id': 'chat-q3', 'model': model,
                'choices': [{'index': 0, 'delta': delta, 'finish_reason': fin}]}
    events = [chunk({'role': 'assistant'})]
    if message.get('reasoning_content'):
        events.append(chunk({'reasoning_content': message['reasoning_content']}))
    if message.get('content'):
        text = message['content']
        events += [chunk({'content': text[:2]}), chunk({'content': text[2:]})]
    for index, call in enumerate(message.get('tool_calls') or []):
        arguments = call['function']['arguments']
        events.append(chunk({'tool_calls': [{'index': index, 'id': call['id'], 'type': 'function',
                                             'function': {'name': call['function']['name'],
                                                          'arguments': ''}}]}))
        for part in (arguments[:5], arguments[5:]):  # split delivery must reassemble
            events.append(chunk({'tool_calls': [{'index': index, 'function': {'arguments': part}}]}))
    events.append(chunk({}, finish))
    events.append({'id': 'chat-q3', 'model': model, 'choices': [], 'usage': usage})
    return _sse(events, mode)


def _chat_q3(payload, mode):
    """Tools / structured-output chat requests the base fake does not model."""
    last = payload['messages'][-1]
    prompt = last.get('content') if isinstance(last.get('content'), str) else ''
    choice = payload.get('tool_choice')
    tools = payload.get('tools') and choice != 'none'
    model = payload['model']
    reasoning = 'Tool turn.' if mode == 'tool-reasoning' and tools else ''
    if isinstance(choice, str) and choice not in ('auto', 'none', 'required'):
        return _json(400, {'error': {'message': "Invalid tool_choice value: '" + choice + "'. Must be "
                                                "'auto', 'none', 'required', or a function object.",
                                     'type': 'invalid_request_error', 'param': None, 'code': None}})
    message, finish = None, 'stop'
    if payload.get('response_format'):
        value = CITY_VALUE if CITY_PROMPT in prompt else {'city': 'Tokyo', 'country': 'Japan'}
        text = json.dumps(value)
        message = {'role': 'assistant', 'content': '```json\n' + text + '\n```' if mode == 'fenced' else text}
    elif tools and NO_TOOL_PROMPT in prompt:
        message = {'role': 'assistant', 'content': '4'}
    elif tools and PARALLEL_PROMPT in prompt:
        unit = 'kelvin' if mode == 'bad-args' else 'celsius'
        message = {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'call_%016x' % index, 'type': 'function', 'function': {
                'name': 'get_weather', 'arguments': json.dumps({'city': city, 'unit': unit})}}
            for index, city in enumerate(('Paris', 'Tokyo'))]}
        finish = 'tool_calls'
    elif any(m.get('role') == 'tool' and 'temperature' in m.get('content', '') for m in payload['messages']):
        results = [json.loads(m['content']) for m in payload['messages'] if m.get('role') == 'tool']
        message = {'role': 'assistant', 'content': max(results, key=lambda r: r['temperature'])['city']}
    elif tools and (choice == 'required' or isinstance(choice, dict)):
        message = {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'call-owned', 'type': 'function', 'function': {
                'name': 'lookup_inventory', 'arguments': '{"sku": "BLUE-17", "warehouse": "north"}'}}]}
        finish = 'tool_calls'
    if message is None:
        return None
    if reasoning:
        message['reasoning_content'] = reasoning
    usage = _usage(3 + len(reasoning.split()), len(reasoning.split()))
    if payload.get('stream'):
        return _chat_stream(model, message, finish, usage, mode)
    return _json(200, _chat_body(model, message, finish, usage))


def _to_chat(payload):
    """Anthropic request -> the equivalent chat payload for the base fake."""
    messages = []
    for message in payload['messages']:
        content = message['content']
        if isinstance(content, str):
            messages.append({'role': message['role'], 'content': content})
            continue
        for block in content:
            if block['type'] == 'tool_result':
                messages.append({'role': 'tool', 'tool_call_id': block['tool_use_id'],
                                 'content': block['content']})
            elif block['type'] == 'text':
                messages.append({'role': message['role'], 'content': block['text']})
            elif block['type'] == 'tool_use':
                messages.append({'role': 'assistant', 'content': None})
    chat = {'model': payload['model'], 'messages': messages, 'max_tokens': payload['max_tokens'],
            'temperature': 0, 'stream': False}
    if payload.get('stop_sequences'):
        chat['stop'] = payload['stop_sequences']
    thinking = (payload.get('thinking') or {}).get('type')
    if thinking is not None:
        chat['chat_template_kwargs'] = {'enable_thinking': thinking != 'disabled'}
    if payload.get('tools'):
        chat['tools'] = payload['tools']
        chat['tool_choice'] = 'none' if (payload.get('tool_choice') or {}).get('type') == 'none' else 'auto'
    return chat


def _anthropic(payload, mode, base_reply, calls):
    _, raw = base_reply(_to_chat(payload), 'ok', calls)
    body = json.loads(raw)
    choice, usage = body['choices'][0], body['usage']
    message = choice['message']
    content = []
    if message.get('reasoning_content'):
        content.append({'type': 'thinking', 'thinking': message['reasoning_content']})
    if message.get('content'):
        content.append({'type': 'text', 'text': message['content']})
    for call in message.get('tool_calls') or []:
        content.append({'type': 'tool_use', 'id': call['id'], 'name': call['function']['name'],
                        'input': json.loads(call['function']['arguments'])})
    matched = MSG_STOP if MSG_STOP in payload.get('stop_sequences', []) else None
    stop_reason = {'stop': 'stop_sequence' if matched else 'end_turn', 'length': 'max_tokens',
                   'tool_calls': 'tool_use'}[choice['finish_reason']]
    result = {'id': 'msg_q3', 'type': 'message', 'role': 'assistant', 'content': content,
              'model': payload['model'], 'stop_reason': stop_reason, 'stop_sequence': matched,
              'usage': {'input_tokens': usage['prompt_tokens'], 'output_tokens': usage['completion_tokens'],
                        'cache_read_input_tokens': 0}}
    if not payload.get('stream'):
        return _json(200, result)
    from test_behavior_anthropic_protocol import frames
    if mode == 'anthropic-drift' and stop_reason == 'stop_sequence':
        result = {**result, 'stop_reason': 'end_turn', 'stop_sequence': None}
    raw = frames(result).encode()
    return 200, 'text/event-stream', raw


def q3_reply(path, payload, mode, calls, base_reply):
    """Return (status, content_type, raw) or None to defer to the base fake."""
    if INTERRUPT_PROMPT in json.dumps(payload):
        calls['interrupted'] += 1
        if not payload.get('stream'):
            time.sleep(SLOW_SECONDS)  # outlasts the client timeout
        long_text = ' '.join('word%d' % i for i in range(40))
        if path == '/v1/messages':
            from test_behavior_anthropic_protocol import frames
            body = {'id': 'msg_long', 'type': 'message', 'role': 'assistant', 'model': payload['model'],
                    'content': [{'type': 'text', 'text': long_text}], 'stop_reason': 'end_turn',
                    'stop_sequence': None, 'usage': {'input_tokens': 7, 'output_tokens': 40}}
            return 200, 'text/event-stream', frames(body, split_json=False).encode()
        message = {'role': 'assistant', 'content': long_text}
        if payload.get('stream'):
            return _chat_stream(payload['model'], message, 'stop', _usage(40), mode)
        return _json(200, _chat_body(payload['model'], message, 'stop', _usage(40)))
    if mode == 'leaky' and calls['interrupted'] and not calls.get('leaked'):
        calls['leaked'] = True  # first request after an interruption answers wrongly
        payload = {**payload, 'messages': [{'role': 'user', 'content': 'unmatched'}]}
        if path == '/v1/chat/completions':
            return (200,) + base_reply(payload, 'ok', calls)
    if path == '/v1/messages':
        return _anthropic(payload, mode, base_reply, calls)
    if path == '/v1/chat/completions':
        return _chat_q3(payload, mode)
    return None
