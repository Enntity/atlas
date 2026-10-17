# SPDX-License-Identifier: AGPL-3.0-only
"""Client cancellation / client timeout, then health and a next request.

Each interruption is followed by: GET /health (200, {"status":"ready",
"model":...} per api/misc_handlers.rs readiness), then the exact baseline
greedy request, which must pass the full envelope gate AND reproduce the
pre-interruption baseline byte-for-byte (text, reasoning, accounting) — a
leaked KV/scheduler state would perturb it. Finally /metrics must report
`atlas_requests_active 0` (metrics.rs ActiveRequestGuard; a dropped handler
future that pinned the gauge is the Atlas#368 leak class).

Unverified from code (recorded, not gated): whether a dropped BLOCKING
request also stops its scheduler work; only the streaming path's
cancellation is visible (scheduler retire_if_cancelled on a closed sink).
"""
import json
import re

from behavior_anthropic_protocol import validate_message
from behavior_protocol import validate_body

BASE_PROMPT = 'What is the capital of Japan? Reply only with Tokyo and no punctuation.'
BASE_ANSWER = 'Tokyo'
LONG_PROMPT = ('Write a detailed 600-word explanation of how TCP congestion control '
               'works, aimed at a systems engineer.')
LONG_TOKENS = 512
CANCEL_FRAMES = 3
CLIENT_TIMEOUT_S = 1.0
ACTIVE_GAUGE = re.compile(r'^atlas_requests_active\s+(\S+)\s*$', re.M)


def collect_robustness(io, model, context_limit):
    rows = []

    def chat(prompt, max_tokens, stream=False):
        value = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                 'temperature': 0, 'seed': 42, 'max_tokens': max_tokens, 'stream': stream}
        if stream:
            value['stream_options'] = {'include_usage': True}
        return value

    def messages(prompt, max_tokens, stream=False):
        return {'model': model, 'max_tokens': max_tokens, 'temperature': 0, 'stream': stream,
                'messages': [{'role': 'user', 'content': prompt}]}

    def answer(name, payload, surface):
        receipt, body = io.request(name, payload, surface)
        if body is not None:
            try:
                if surface == 'messages':
                    summary = validate_message(body, 'end_turn', payload['max_tokens'], context_limit)
                    key = (summary['text'], summary['thinking'], summary['usage'])
                    text = summary['text']
                else:
                    summary = validate_body(body, 'stop', payload['max_tokens'], context_limit)
                    key = (summary['content'], summary['reasoning'], summary['accounting'])
                    text = summary['content']
                if text.strip() != BASE_ANSWER:
                    raise ValueError('wrong visible answer')
                receipt['summary'] = summary
                receipt['identity_key'] = repr(key)
                receipt['passed'] = True
            except Exception as error:
                receipt['error'] = type(error).__name__ + ': ' + str(error)
        rows.append(receipt)
        return receipt

    def health(name):
        receipt, text = io.probe(name, 'health')
        try:
            body = json.loads(text)
            if receipt.get('http_status') != 200 or body.get('status') != 'ready':
                raise ValueError('server not ready after interruption: ' + text[:200])
            if not isinstance(body.get('model'), str) or not body['model']:
                raise ValueError('ready probe lacks model identity')
            receipt['passed'] = True
        except Exception as error:
            receipt['error'] = receipt['error'] or type(error).__name__ + ': ' + str(error)
        rows.append(receipt)

    def interrupted(name, payload, surface, mode, limit):
        receipt, _ = io.interrupt(name, payload, surface, mode, limit)
        try:
            if receipt.get('http_status') is not None and receipt['http_status'] != 200:
                raise ValueError('interrupted request was rejected: HTTP ' + str(receipt['http_status']))
            if mode == 'cancel':
                if receipt.get('content_type') != 'text/event-stream':
                    raise ValueError('SSE media type required before cancelling')
                if receipt.get('completed_before_cancel') or receipt.get('frames_before_cancel') != limit:
                    raise ValueError('stream ended before the client cancelled; cancel not exercised')
            elif receipt.get('completed_before_timeout') is not False:
                raise ValueError('response completed inside the client timeout; timeout not exercised')
            receipt['passed'] = True
        except Exception as error:
            receipt['error'] = type(error).__name__ + ': ' + str(error)
        rows.append(receipt)

    def same(before, after):
        if before.get('passed') and after.get('passed') and before['identity_key'] != after['identity_key']:
            after['passed'] = False
            after['error'] = 'post-interruption output differs from baseline (leaked state)'

    base_chat = answer('robust-baseline-chat', chat(BASE_PROMPT, 256), 'chat')
    base_msg = answer('robust-baseline-messages', messages(BASE_PROMPT, 256), 'messages')

    steps = [('robust-cancel-chat', chat(LONG_PROMPT, LONG_TOKENS, True), 'chat', 'cancel', CANCEL_FRAMES),
             ('robust-cancel-messages', messages(LONG_PROMPT, LONG_TOKENS, True), 'messages',
              'cancel', CANCEL_FRAMES),
             ('robust-timeout-chat', chat(LONG_PROMPT, LONG_TOKENS), 'chat', 'timeout', CLIENT_TIMEOUT_S),
             ('robust-timeout-messages', messages(LONG_PROMPT, LONG_TOKENS), 'messages',
              'timeout', CLIENT_TIMEOUT_S)]
    for name, payload, surface, mode, limit in steps:
        interrupted(name, payload, surface, mode, limit)
        health(name + '-health')
        if surface == 'chat':
            same(base_chat, answer(name + '-next', chat(BASE_PROMPT, 256), 'chat'))
        else:
            same(base_msg, answer(name + '-next', messages(BASE_PROMPT, 256), 'messages'))

    receipt, text = io.probe('robust-active-gauge', 'metrics')
    values = ACTIVE_GAUGE.findall(text)
    receipt['atlas_requests_active'] = values
    if receipt.get('http_status') == 200 and len(values) == 1 and float(values[0]) == 0:
        receipt['passed'] = True
    else:
        receipt['error'] = receipt['error'] or 'atlas_requests_active must be exactly 0 after all requests'
    rows.append(receipt)
    return rows
