# SPDX-License-Identifier: AGPL-3.0-only
import json
import unittest

from behavior_anthropic_protocol import decode_messages_sse, validate_message


def message(text='Tokyo', thinking=None, tools=None, stop='end_turn', stop_sequence=None):
    content = []
    if thinking is not None:
        content.append({'type': 'thinking', 'thinking': thinking})
    if text:
        content.append({'type': 'text', 'text': text})
    for tool in tools or []:
        content.append({'type': 'tool_use', **tool})
    return {'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'content': content,
            'model': 'test', 'stop_reason': stop, 'stop_sequence': stop_sequence,
            'usage': {'input_tokens': 9, 'output_tokens': 4, 'cache_read_input_tokens': 0}}


def frames(body, split_json=True, keepalive=True):
    """Encode `body` the way anthropic/translator.rs streams it."""
    out = []

    def emit(name, data):
        out.append('event: ' + name + '\ndata: ' + json.dumps({'type': name, **data}) + '\n\n')

    emit('message_start', {'message': {**{k: body[k] for k in ('id', 'type', 'role', 'model')},
                                       'content': [], 'stop_reason': None, 'stop_sequence': None,
                                       'usage': {'input_tokens': 0, 'output_tokens': 0}}})
    if keepalive:
        out.append(':\n\n')
    for index, block in enumerate(body['content']):
        if block['type'] == 'tool_use':
            emit('content_block_start', {'index': index, 'content_block': {
                'type': 'tool_use', 'id': block['id'], 'name': block['name'], 'input': {}}})
            raw = json.dumps(block['input'])
            parts = [raw[:3], raw[3:]] if split_json else [raw]
            for part in parts:
                emit('content_block_delta', {'index': index, 'delta': {
                    'type': 'input_json_delta', 'partial_json': part}})
        else:
            field = block['type']
            emit('content_block_start', {'index': index, 'content_block': {'type': field, field: ''}})
            emit('content_block_delta', {'index': index, 'delta': {
                'type': field + '_delta', field: block[field]}})
        emit('content_block_stop', {'index': index})
    emit('message_delta', {'delta': {'stop_reason': body['stop_reason'],
                                     'stop_sequence': body['stop_sequence']},
                           'usage': body['usage']})
    emit('message_stop', {})
    return ''.join(out)


TOOL = {'id': 'call_1', 'name': 'lookup', 'input': {'sku': 'BLUE-17', 'n': 2}}


class AnthropicProtocolTests(unittest.TestCase):
    def test_blocking_blocks_and_summary(self):
        summary = validate_message(message(thinking='Recall.'), 'end_turn', 64, 4096)
        self.assertEqual((summary['text'], summary['thinking']), ('Tokyo', 'Recall.'))
        self.assertEqual(summary['blocks'], ['thinking', 'text'])
        tool = validate_message(message('', tools=[TOOL], stop='tool_use'), 'tool_use', 64, 4096)
        self.assertEqual(tool['tool_uses'][0]['input'], TOOL['input'])

    def test_blocking_violations(self):
        cases = {
            'stop': (message(stop='max_tokens'), 'end_turn', None),
            'stop-seq': (message(stop='stop_sequence'), 'stop_sequence', 'STOP'),
            'leak': (message('Tokyo</think>'), 'end_turn', None),
            'order': ({**message(thinking='x'), 'content': [{'type': 'text', 'text': 'a'},
                                                            {'type': 'thinking', 'thinking': 'b'}]},
                      'end_turn', None),
            'tool-mismatch': (message(tools=[TOOL]), 'end_turn', None),
            'dup-ids': (message('', tools=[TOOL, TOOL], stop='tool_use'), 'tool_use', None),
            'id': ({**message(), 'id': 'chat-1'}, 'end_turn', None),
            'usage': ({**message(), 'usage': {'input_tokens': 0, 'output_tokens': 1}}, 'end_turn', None),
        }
        for name, (body, stop, sequence) in cases.items():
            with self.assertRaises(ValueError, msg=name):
                validate_message(body, stop, 64, 4096, sequence)

    def test_stream_roundtrip_including_split_tool_json(self):
        for body in (message(thinking='Recall.'),
                     message('', tools=[TOOL, {**TOOL, 'id': 'call_2'}], stop='tool_use')):
            rebuilt, events = decode_messages_sse(frames(body))
            self.assertEqual(rebuilt, body)
            self.assertEqual(events[0]['type'], 'message_start')
            self.assertEqual(events[-1]['type'], 'message_stop')

    def test_stream_order_violations(self):
        body = message(thinking='Recall.')
        good = frames(body, keepalive=False)
        chunks = [c for c in good.split('\n\n') if c]
        broken = {
            'no-stop': '\n\n'.join(chunks[:-1]) + '\n\n',
            'no-start': '\n\n'.join(chunks[1:]) + '\n\n',
            'done': good + 'data: [DONE]\n\n',
            'misnamed': good.replace('event: content_block_stop', 'event: content_block_end', 1),
            'swapped-tail': '\n\n'.join(chunks[:-2] + [chunks[-1], chunks[-2]]) + '\n\n',
            'gap-index': good.replace('"index": 1', '"index": 2'),
            'wrong-delta': good.replace('"thinking_delta"', '"text_delta"'),
            'unclosed': '\n\n'.join(c for c in chunks if 'content_block_stop' not in c) + '\n\n',
        }
        for name, raw in broken.items():
            with self.assertRaises(ValueError, msg=name):
                decode_messages_sse(raw)
        bad_json = frames(message('', tools=[TOOL], stop='tool_use')).replace(
            '"partial_json": "{\\"s"', '"partial_json": "{{\\"s"')
        with self.assertRaises(ValueError):
            decode_messages_sse(bad_json)


if __name__ == '__main__':
    unittest.main()
