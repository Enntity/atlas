# SPDX-License-Identifier: AGPL-3.0-only
"""Strict Anthropic /v1/messages envelopes (blocking and SSE), stdlib only.

Contract taken from crates/spark-server/src/anthropic (types.rs, translate.rs,
translator.rs, helpers.rs):
- blocking: {id:'msg_*', type:'message', role:'assistant', content:[...],
  model, stop_reason, stop_sequence, usage:{input_tokens, output_tokens,
  cache_read_input_tokens}}; blocks are thinking -> text -> tool_use.
- streaming: `event: <name>` + `data: <json>` with data.type == name, order
  message_start, (content_block_start, content_block_delta*,
  content_block_stop)*, message_delta, message_stop; no `[DONE]`; one open
  block at a time with contiguous indices from 0; axum keep-alive comment
  lines (': ...') are tolerated.
Atlas emits no thinking `signature` / `signature_delta` / `redacted_thinking`,
so none is required (and none is rejected either).
"""
import json

from behavior_protocol import assert_no_special_tokens, finite, unique_object
from behavior_schema import parse_strict_json

STOP_REASONS = ('end_turn', 'max_tokens', 'stop_sequence', 'tool_use', 'refusal')
BLOCK_DELTA = {'text': ('text_delta', 'text'), 'thinking': ('thinking_delta', 'thinking'),
               'tool_use': ('input_json_delta', 'partial_json')}


def check_message_usage(usage, max_tokens, context_limit):
    if not isinstance(usage, dict):
        raise ValueError('usage object required')
    for key in ('input_tokens', 'output_tokens'):
        if type(usage.get(key)) is not int or usage[key] <= 0:
            raise ValueError('positive integer usage required: ' + key)
    cached = usage.get('cache_read_input_tokens', 0)
    if type(cached) is not int or not 0 <= cached <= usage['input_tokens']:
        raise ValueError('cache_read_input_tokens invalid')
    if usage['output_tokens'] > max_tokens or usage['input_tokens'] + max_tokens > context_limit:
        raise ValueError('output or context budget exceeded')
    return {'input_tokens': usage['input_tokens'], 'output_tokens': usage['output_tokens'],
            'cache_read_input_tokens': cached}


def validate_message(body, stop_reason, max_tokens, context_limit, stop_sequence=None):
    """Validate one blocking (or SSE-reconstructed) message; return a summary."""
    finite(body)
    if not isinstance(body, dict) or body.get('type') == 'error' or 'error' in body:
        raise ValueError('server error or invalid envelope')
    if not isinstance(body.get('id'), str) or not body['id'].startswith('msg_'):
        raise ValueError('message id must be msg_*')
    if body.get('type') != 'message' or body.get('role') != 'assistant':
        raise ValueError('assistant message envelope required')
    if not isinstance(body.get('model'), str) or not body['model']:
        raise ValueError('model identity missing')
    if body.get('stop_reason') not in STOP_REASONS:
        raise ValueError('unknown stop_reason: ' + repr(body.get('stop_reason')))
    if body['stop_reason'] != stop_reason:
        raise ValueError('wrong stop_reason: ' + repr(body['stop_reason']))
    if body.get('stop_sequence') != stop_sequence:
        raise ValueError('wrong stop_sequence: ' + repr(body.get('stop_sequence')))
    content = body.get('content')
    if not isinstance(content, list):
        raise ValueError('content block list required')
    text, thinking, tools, kinds = [], [], [], []
    for block in content:
        kind = block.get('type') if isinstance(block, dict) else None
        if kind == 'text':
            if not isinstance(block.get('text'), str):
                raise ValueError('text block requires text')
            assert_no_special_tokens(block['text'], 'text block')
            text.append(block['text'])
        elif kind == 'thinking':
            if not isinstance(block.get('thinking'), str):
                raise ValueError('thinking block requires thinking text')
            assert_no_special_tokens(block['thinking'], 'thinking block')
            thinking.append(block['thinking'])
        elif kind == 'tool_use':
            if not isinstance(block.get('id'), str) or not block['id']:
                raise ValueError('tool_use id missing')
            if not isinstance(block.get('name'), str) or not block['name']:
                raise ValueError('tool_use name missing')
            if not isinstance(block.get('input'), dict):
                raise ValueError('tool_use input must be an object')
            tools.append({'id': block['id'], 'name': block['name'], 'input': block['input']})
        else:
            raise ValueError('unexpected content block: ' + repr(kind))
        kinds.append(kind)
    if kinds != sorted(kinds, key=('thinking', 'text', 'tool_use').index):
        raise ValueError('content blocks out of thinking/text/tool_use order')
    if len({tool['id'] for tool in tools}) != len(tools):
        raise ValueError('duplicate tool_use ids')
    if (stop_reason == 'tool_use') != bool(tools):
        raise ValueError('tool_use stop_reason and tool_use blocks disagree')
    visible = ''.join(text)
    if stop_reason in ('end_turn', 'stop_sequence') and not visible.strip():
        raise ValueError('visible answer missing')
    usage = check_message_usage(body.get('usage'), max_tokens, context_limit)
    return {'text': visible, 'thinking': ''.join(thinking), 'tool_uses': tools,
            'blocks': kinds, 'stop_reason': body['stop_reason'],
            'stop_sequence': body.get('stop_sequence'), 'usage': usage}


def parse_named_sse(raw):
    """Parse `event:`/`data:` frames; every frame must be named and typed."""
    events, name, data = [], None, []
    for line in raw.splitlines() + ['']:
        if line.startswith(':'):
            continue
        if line.startswith('event:'):
            if name is not None:
                raise ValueError('two event names in one frame')
            name = line[6:].strip()
        elif line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
        elif not line:
            if name is None and not data:
                continue
            if name is None or not data:
                raise ValueError('SSE frame missing event name or data')
            value = '\n'.join(data)
            if value == '[DONE]':
                raise ValueError('[DONE] is not part of the Anthropic stream')
            event = json.loads(value, object_pairs_hook=unique_object)
            finite(event)
            if not isinstance(event, dict) or event.get('type') != name:
                raise ValueError('event name and data.type disagree: ' + repr(name))
            events.append(event)
            name, data = None, []
        else:
            raise ValueError('unexpected SSE line: ' + repr(line[:40]))
    return events


def decode_messages_sse(raw):
    """Rebuild a blocking-shaped message from the stream, enforcing order."""
    events = parse_named_sse(raw)
    if not events or events[0]['type'] != 'message_start':
        raise ValueError('stream must open with message_start')
    if events[-1]['type'] != 'message_stop' or len(events) < 3:
        raise ValueError('stream must close with message_stop')
    if events[-2]['type'] != 'message_delta':
        raise ValueError('message_delta must precede message_stop')
    start = events[0].get('message')
    if not isinstance(start, dict) or start.get('content') != [] or start.get('stop_reason') is not None:
        raise ValueError('message_start must carry an empty, unfinished message')
    blocks, open_index = [], None
    for event in events[1:-2]:
        kind = event['type']
        if kind == 'content_block_start':
            if open_index is not None:
                raise ValueError('content block opened while another is open')
            if event.get('index') != len(blocks):
                raise ValueError('content block indices must be contiguous from 0')
            block = event.get('content_block')
            if not isinstance(block, dict) or block.get('type') not in BLOCK_DELTA:
                raise ValueError('unknown content block start')
            blocks.append({'block': dict(block), 'parts': []})
            open_index = event['index']
        elif kind == 'content_block_delta':
            if event.get('index') != open_index or open_index is None:
                raise ValueError('delta for a block that is not open')
            entry = blocks[open_index]
            delta_type, field = BLOCK_DELTA[entry['block']['type']]
            delta = event.get('delta') or {}
            if delta.get('type') != delta_type or not isinstance(delta.get(field), str):
                raise ValueError('delta type does not match its block')
            entry['parts'].append(delta[field])
        elif kind == 'content_block_stop':
            if event.get('index') != open_index or open_index is None:
                raise ValueError('stop for a block that is not open')
            open_index = None
        elif kind != 'ping':
            raise ValueError('unexpected event inside the message: ' + kind)
    if open_index is not None:
        raise ValueError('content block left open')
    delta = events[-2].get('delta') or {}
    content = []
    for entry in blocks:
        block, joined = entry['block'], ''.join(entry['parts'])
        if block['type'] == 'text':
            content.append({'type': 'text', 'text': block.get('text', '') + joined})
        elif block['type'] == 'thinking':
            content.append({'type': 'thinking', 'thinking': block.get('thinking', '') + joined})
        else:
            if block.get('input') not in ({}, None):
                raise ValueError('streamed tool_use must start with empty input')
            content.append({'type': 'tool_use', 'id': block.get('id'), 'name': block.get('name'),
                            'input': parse_strict_json(joined)})
    usage = events[-2].get('usage')
    body = {'id': start.get('id'), 'type': 'message', 'role': start.get('role'),
            'model': start.get('model'), 'content': content,
            'stop_reason': delta.get('stop_reason'), 'stop_sequence': delta.get('stop_sequence'),
            'usage': usage}
    return body, events
