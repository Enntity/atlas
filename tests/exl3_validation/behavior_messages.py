# SPDX-License-Identifier: AGPL-3.0-only
"""Anthropic /v1/messages behavior cases and OpenAI-path parity.

Server facts used (crates/spark-server/src/anthropic): `max_tokens` is
required; there is no `seed` field (IR seed = None), so every parity case is
greedy; `thinking: {type:'enabled', budget_tokens}` / `{type:'disabled'}`
map to the same ThinkingDirective as the OpenAI path; tool_choice `tool`
(named) / `auto` / `any` / `none`; tool ids are `call_*` (not `toolu_*`),
so only non-empty + unique is gated. `output_tokens` includes thinking
tokens, so it must equal OpenAI `completion_tokens` for the same request.

Gated even though current code disagrees (protocol drift, expected red until
fixed): a streamed stop-sequence hit must report stop_reason
'stop_sequence' + the matched string like the blocking path does
(translator.rs hard-codes stop_sequence null and maps 'stop' to end_turn).
Not gated: message_start usage (Atlas sends input_tokens 0 there; the real
counts arrive in message_delta, which is what the stream summary uses), and
the thinking budget (Anthropic usage has no reasoning-token split).
"""
import json

from behavior_anthropic_protocol import validate_message
from behavior_protocol import validate_body
from behavior_schema import schema_errors

ANSWER = 'Tokyo'
TOKYO_PROMPT = 'What is the capital of Japan? Reply only with Tokyo and no punctuation.'
CALC_PROMPT = 'Calculate 37 * 14 - 86. Reply only with the integer answer.'
CALC_ANSWER = '432'
LONG_PROMPT = 'Count from 1 to 30, separated by single spaces.'
STOP_MARKER = 'STOPNOW'
STOP_PROMPT = 'Write exactly the phrase alpha STOPNOW omega and nothing else.'
MAX_TOKENS = 256
LENGTH_TOKENS = 8
THINKING_BUDGET = 32
TOOL_NAME = 'lookup_inventory'
TOOL_ARGUMENTS = {'sku': 'BLUE-17', 'warehouse': 'north'}
TOOL_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['sku', 'warehouse'],
               'properties': {'sku': {'type': 'string'}, 'warehouse': {'type': 'string'}}}
TOOL_PROMPT = ('Use lookup_inventory exactly once for SKU BLUE-17 in warehouse north. '
               'Wait for the tool result; then reply only with the integer available count.')
TOOL_RESULT = {**TOOL_ARGUMENTS, 'available': 19}


def collect_messages(io, model, context_limit):
    rows = {}
    tool = {'name': TOOL_NAME, 'description': 'Look up inventory.', 'input_schema': TOOL_SCHEMA}

    def payload(prompt=None, stream=False, max_tokens=MAX_TOKENS, thinking=None, stop=None,
                messages=None, tools=False, choice=None):
        value = {'model': model, 'max_tokens': max_tokens, 'temperature': 0, 'stream': stream,
                 'messages': messages or [{'role': 'user', 'content': prompt}]}
        if thinking is not None:
            value['thinking'] = thinking
        if stop:
            value['stop_sequences'] = stop
        if tools:
            value['tools'] = [tool]
        if choice is not None:
            value['tool_choice'] = choice
        return value

    def fail(receipt, message):
        receipt['passed'] = False
        receipt['error'] = message

    def run(name, task, stop_reason, stop_sequence=None, check=None):
        receipt, body = io.request(name, task, 'messages')
        if body is not None:
            try:
                summary = validate_message(body, stop_reason, task['max_tokens'], context_limit,
                                           stop_sequence)
                if check is not None:
                    check(summary)
                receipt['summary'] = summary
                receipt['passed'] = True
            except Exception as error:
                receipt['error'] = type(error).__name__ + ': ' + str(error)
        rows[name] = receipt
        return receipt

    def exact(word):
        def check(summary):
            if summary['text'].strip() != word:
                raise ValueError('wrong visible answer')
        return check

    def stop_cut(summary):
        if STOP_MARKER in summary['text'] or 'omega' in summary['text'] or 'alpha' not in summary['text']:
            raise ValueError('stop sequence not honored at the match point')

    def length_used(summary):
        if summary['usage']['output_tokens'] != LENGTH_TOKENS:
            raise ValueError('max_tokens stop must consume exactly max_tokens')

    def thought(word):
        def check(summary):
            exact(word)(summary)
            if not summary['thinking'].strip():
                raise ValueError('thinking enabled but no thinking block')
            if summary['thinking'].strip() == summary['text'].strip():
                raise ValueError('thinking and text are not separated')
        return check

    def called(summary):
        if len(summary['tool_uses']) != 1 or summary['tool_uses'][0]['name'] != TOOL_NAME:
            raise ValueError('exactly one lookup_inventory tool_use required')
        arguments = summary['tool_uses'][0]['input']
        errors = schema_errors(arguments, TOOL_SCHEMA)
        if errors:
            raise ValueError('tool input violates input_schema: ' + '; '.join(errors))
        if arguments != TOOL_ARGUMENTS:
            raise ValueError('tool input crossed or changed')

    def pair(a, b, key, fields):
        if not (rows[a].get('passed') and rows[b].get('passed')):
            return
        identity = {field: rows[a]['summary'][field] == rows[b]['summary'][field] for field in fields}
        rows[b][key] = identity
        if not all(identity.values()):
            fail(rows[b], key + ' mismatch: ' + repr(identity))

    blocking_fields = ('text', 'thinking', 'blocks', 'stop_reason', 'stop_sequence', 'usage', 'tool_uses')
    for suffix, stream in (('', False), ('-stream', True)):
        run('msg' + suffix + '-stop', payload(TOKYO_PROMPT, stream), 'end_turn', check=exact(ANSWER))
        run('msg' + suffix + '-length', payload(LONG_PROMPT, stream, LENGTH_TOKENS), 'max_tokens',
            check=length_used)
        run('msg' + suffix + '-stop-seq', payload(STOP_PROMPT, stream, stop=[STOP_MARKER]),
            'stop_sequence', STOP_MARKER, stop_cut)
        run('msg' + suffix + '-think-on',
            payload(CALC_PROMPT, stream, thinking={'type': 'enabled', 'budget_tokens': THINKING_BUDGET}),
            'end_turn', check=thought(CALC_ANSWER))
        run('msg' + suffix + '-think-off', payload(CALC_PROMPT, stream, thinking={'type': 'disabled'}),
            'end_turn', check=exact(CALC_ANSWER))
        run('msg' + suffix + '-tool-use',
            payload(TOOL_PROMPT, stream, thinking={'type': 'disabled'}, tools=True,
                    choice={'type': 'tool', 'name': TOOL_NAME}), 'tool_use', check=called)
    for name in ('msg-stop', 'msg-length', 'msg-stop-seq', 'msg-think-on', 'msg-think-off'):
        pair(name, name.replace('msg-', 'msg-stream-'), 'blocking_identity', blocking_fields)

    # tool_use -> tool_result round trip (dependent follow-up).
    first = rows['msg-tool-use']
    if first.get('passed'):
        call = first['summary']['tool_uses'][0]
        assistant = first['response']['content']
        followup = payload(messages=[
            {'role': 'user', 'content': TOOL_PROMPT},
            {'role': 'assistant', 'content': assistant},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': call['id'],
                                          'content': json.dumps(TOOL_RESULT)}]}],
            thinking={'type': 'disabled'}, tools=True, choice={'type': 'none'})
        run('msg-tool-result', followup, 'end_turn', check=exact('19'))
    else:
        rows['msg-tool-result'] = {'id': 'msg-tool-result', 'passed': False,
                                   'error': 'valid initial tool_use missing; no fabricated tool id'}

    # Parity with the OpenAI path: same greedy prompt, same thinking default.
    chat, body = io.request('msg-parity-chat', {
        'model': model, 'messages': [{'role': 'user', 'content': TOKYO_PROMPT}],
        'temperature': 0, 'max_tokens': MAX_TOKENS, 'stream': False}, 'chat')
    rows['msg-parity-chat'] = chat
    if body is not None:
        try:
            summary = validate_body(body, 'stop', MAX_TOKENS, context_limit)
            chat['summary'] = summary
            chat['passed'] = True
        except Exception as error:
            chat['error'] = type(error).__name__ + ': ' + str(error)
    if chat.get('passed') and rows['msg-stop'].get('passed'):
        ours, theirs = rows['msg-stop']['summary'], chat['summary']
        identity = {'text': ours['text'] == theirs['content'],
                    'thinking': ours['thinking'] == theirs['reasoning'],
                    'input_tokens': ours['usage']['input_tokens'] == theirs['usage']['prompt_tokens'],
                    'output_tokens': ours['usage']['output_tokens'] == theirs['usage']['completion_tokens']}
        chat['openai_parity'] = identity
        if not all(identity.values()):
            fail(chat, 'openai_parity mismatch: ' + repr(identity))
    return list(rows.values())
