# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for q3_behavior_suite.py against an in-process fake server.

The fake is a tiny deterministic "engine" (word tokens, crc32 ids, EOS id 2)
that applies max_tokens, stop strings, stop_token_ids and seeds itself, then
encodes OpenAI chat / legacy completions / Anthropic Messages, blocking and
SSE. It is spec-correct by default; each FAULT mode injects one defect, and
the tests require the suite to fail exactly the named assertion it owns.
Never a qualification target.
"""
import contextlib
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import random
import re
import tempfile
import threading
import time
import unittest
import zlib

import q3_behavior_suite as q3
from test_behavior_anthropic_protocol import frames

EOS = 2
WORDS = ('cat', 'piano', 'moon', 'river', 'toast', 'lamp', 'otter', 'violin', 'cloud', 'pepper')
LONG_TEXT = ' '.join('word%d' % i for i in range(120))


def tok_id(piece):
    return 100 + zlib.crc32(piece.encode()) % 90000


def tokenize(text):
    return re.findall(r'\s*\S+', text)


def last_user(messages):
    for message in reversed(messages):
        if message.get('role') == 'user':
            content = message.get('content')
            if isinstance(content, list):
                return ''.join(b.get('text', '') for b in content if isinstance(b, dict))
            return content or ''
    return ''


class Fake:
    """Shared server state: fault mode, interruption flag, counters."""

    def __init__(self, mode='ok'):
        self.mode, self.interrupted, self.samples, self.lock = mode, False, 0, threading.Lock()

    def has(self, fault):
        return self.mode == fault

    # ---- what the "model" says -------------------------------------------
    def respond(self, payload, prompt, history, thinking):
        """Returns (reasoning, content, tool_calls) or None for unknown prompts."""
        tools = payload.get('tools') and payload.get('tool_choice') != 'none'
        if any(m.get('role') == 'tool' for m in history):
            results = [json.loads(m['content']) for m in history if m.get('role') == 'tool']
            answer = max(results, key=lambda r: r['temperature'])['city']
            return '', ('Tokyo' if self.has('followup-ignored') else answer), None
        if tools and prompt == q3.PARALLEL_PROMPT:
            unit = 'kelvin' if self.has('tool-bad-args') else 'celsius'
            cities = ('Paris',) if self.has('tool-one-call') else ('Paris', 'Tokyo')
            return '', '', [{'id': 'call_%d' % i, 'type': 'function', 'function': {
                'name': 'get_weather', 'arguments': json.dumps({'city': c, 'unit': unit})}}
                for i, c in enumerate(cities)]
        if tools and prompt == q3.INVENTORY_PROMPT:
            return '', '', [{'id': 'call_inv', 'type': 'function', 'function': {
                'name': 'lookup_inventory', 'arguments': json.dumps(q3.INVENTORY_ARGUMENTS)}}]
        if prompt == q3.TOKYO_PROMPT:
            return '', ('Kyoto' if self.has('wrong-answer') else 'Tokyo'), None
        if prompt == q3.COUNT_PROMPT:
            return '', ' '.join(str(i) for i in range(1, 31)), None
        if prompt == q3.REACT_PROMPT:
            return '', 'Thought: check the weather\nObservation: sunny and 21 C\nFinal: it is sunny', None
        if prompt == q3.CALC_PROMPT:
            if self.has('reasoning-merged') and thinking:
                return '', '37 * 14 = 518, minus 86. 432', None
            content = '<think>432' if self.has('think-leak') and thinking else '432'
            return ('37 * 14 = 518 and 518 - 86 = 432.' if thinking else ''), content, None
        if prompt == q3.JOKE_PROMPT:
            rng = self.rng(payload)
            return '', ' '.join(rng.choice(WORDS) for _ in range(12)), None
        if prompt == q3.CITY_PROMPT:
            value = dict(q3.CITY_VALUE)
            if self.has('schema-wrong-type'):
                value['is_capital'] = 'yes'
            text = json.dumps(value)
            if payload.get('seed') is not None and self.has('seed-drift'):
                with self.lock:
                    self.samples += 1
                    text = json.dumps(value, indent=self.samples)  # differs on every call
            return '', ('```json\n' + text + '\n```' if self.has('schema-fence') else text), None
        if prompt == q3.OBJECT_PROMPT:
            text = json.dumps({'city': 'Tokyo', 'country': 'Japan'})
            return '', ('```json\n' + text + '\n```' if self.has('schema-fence') else text), None
        if prompt == q3.RECALL_PROMPT:
            return '', 'OK', None
        if prompt == q3.RECALL_QUESTION:
            remembered = any(q3.CODEWORD in (m.get('content') or '') for m in history[:-1])
            return '', (q3.CODEWORD if remembered and not self.has('forget') else 'I do not know'), None
        return None

    def rng(self, payload):
        seed = payload.get('seed')
        if self.has('responses-drop-seed') and 'input' in payload:
            seed = None
        with self.lock:
            self.samples += 1
            counter = self.samples
        if seed is None:
            return random.Random()
        if self.has('greedy-sampling'):  # ignores seed and temperature
            return random.Random(0)
        return random.Random(seed * 1000 + counter if self.has('seed-drift') else seed)

    # ---- the "engine" ----------------------------------------------------
    def run(self, payload, reasoning, content, calls, stream):
        stops = payload.get('stop') or payload.get('stop_sequences') or []
        stops = [stops] if isinstance(stops, str) else stops
        if self.has('ignore-stop') or (stream and self.has('stream-stop-leak')):
            stops = []
        stop_ids = set() if self.has('ignore-stop-ids') else set(payload.get('stop_token_ids') or [])
        max_tokens = payload['max_tokens']
        if calls:
            return {'reasoning': [], 'content': [], 'tail': [EOS], 'finish': 'tool_calls',
                    'matched': None, 'count': 12, 'calls': calls}
        sequence = [('r', t) for t in tokenize(reasoning)] + [('c', t) for t in tokenize(content)]
        out, finish, matched, cut, tail = [], None, None, None, []
        for kind, piece in sequence:
            if len(out) >= max_tokens:
                finish = 'length'
                break
            if tok_id(piece) in stop_ids:
                tail, finish = [tok_id(piece)], 'stop'  # generated, counted, not detokenized
                break
            out.append((kind, piece))
            text = ''.join(p for k, p in out if k == 'c')
            hits = [s for s in stops if s in text]
            if kind == 'c' and hits:
                matched = min(hits, key=text.find)
                end = text.find(matched)
                if payload.get('include_stop_str_in_output') and not self.has('exclude-stop-str'):
                    end += len(matched)
                cut, finish = text[:end], 'stop'
                break
        else:
            if len(out) < max_tokens:
                tail, finish = [EOS], 'stop'
            else:
                finish = 'length'
        if self.has('trim-stop') and stops:
            # Trims the matched string but keeps generating past it.
            out = [(k, p.replace(stops[0], '')) for k, p in sequence]
            cut, matched, finish, tail = None, stops[0], 'stop', [EOS]
        reasoning_pieces = [p for k, p in out if k == 'r']
        if cut is not None:
            content_pieces = tokenize(cut)
            rest = cut[len(''.join(content_pieces)):]
            if rest:
                content_pieces.append(rest)
        else:
            content_pieces = [p for k, p in out if k == 'c']
        if finish == 'stop' and matched is None and tail == [EOS] and self.has('eos-leak'):
            content_pieces.append('<|endoftext|>')
        count = len(out) + len(tail)
        if finish == 'length' and self.has('short-length'):
            count -= 1
        return {'reasoning': reasoning_pieces, 'content': content_pieces, 'tail': tail, 'finish': finish,
                'matched': matched, 'count': count, 'calls': None}

    # ---- encoders --------------------------------------------------------
    def usage(self, prompt, result):
        prompt_tokens, count = len(prompt) // 4 + 1, result['count']
        reasoning = 0 if self.has('reasoning-uncounted') else len(result['reasoning'])
        return {'prompt_tokens': prompt_tokens, 'completion_tokens': count,
                'total_tokens': prompt_tokens + count + (1 if self.has('usage-sum') else 0),
                'completion_tokens_details': {'reasoning_tokens': reasoning}}

    def chat(self, payload, result, prompt):
        model, usage = payload['model'], self.usage(prompt, result)
        content = ''.join(result['content'])
        reasoning = ''.join(result['reasoning'])
        message = {'role': 'assistant', 'content': content or (None if result['calls'] else '')}
        if reasoning:
            message['reasoning_content'] = reasoning
        if result['calls']:
            message['tool_calls'] = result['calls']
        if not payload.get('stream'):
            return 'application/json', [json.dumps({
                'id': 'chatcmpl-fake', 'object': 'chat.completion', 'model': model,
                'choices': [{'index': 0, 'message': message, 'finish_reason': result['finish']}],
                'usage': usage})]

        def chunk(delta, finish=None, ids=None):
            choice = {'index': 0, 'delta': delta, 'finish_reason': finish}
            if ids is not None and payload.get('return_token_ids') and not self.has('no-token-ids'):
                choice['token_ids'] = ids
            return {'id': 'chatcmpl-fake', 'object': 'chat.completion.chunk', 'model': model,
                    'choices': [choice]}
        events = [chunk({'role': 'assistant', 'content': ''})]
        events += [chunk({'reasoning_content': p}, ids=[tok_id(p)]) for p in result['reasoning']]
        events += [chunk({'content': p}, ids=[tok_id(p)]) for p in result['content']]
        for index, call in enumerate(result['calls'] or []):
            arguments = call['function']['arguments']
            if self.has('args-spacing'):  # same JSON, different whitespace than blocking
                arguments = json.dumps(json.loads(arguments), separators=(',', ':'))
            events.append(chunk({'tool_calls': [{'index': index, 'id': call['id'], 'type': 'function',
                                                 'function': {'name': call['function']['name'],
                                                              'arguments': arguments[:4]}}]}))
            events.append(chunk({'tool_calls': [{'index': index, 'function': {'arguments': arguments[4:]}}]}))
        finish = chunk({}, result['finish'], result['tail'])
        usage_chunk = {'id': 'chatcmpl-fake', 'object': 'chat.completion.chunk', 'model': model,
                       'choices': [], 'usage': usage}
        events += [usage_chunk, finish] if self.has('usage-first') else [finish, usage_chunk]
        return 'text/event-stream', ['data: ' + json.dumps(e) + '\n\n' for e in events] + ['data: [DONE]\n\n']

    def completion(self, payload, result):
        model, usage = payload['model'], self.usage(payload['prompt'], result)
        text = ''.join(result['content'])
        if not payload.get('stream'):
            return 'application/json', [json.dumps({
                'id': 'cmpl-fake', 'object': 'text_completion', 'model': model,
                'choices': [{'index': 0, 'text': text, 'finish_reason': result['finish']}], 'usage': usage})]

        def chunk(piece, finish=None):
            return {'id': 'cmpl-fake', 'object': 'text_completion', 'model': model,
                    'choices': [{'index': 0, 'text': piece, 'finish_reason': finish}]}
        finish = chunk('', result['finish'])
        usage_chunk = {'id': 'cmpl-fake', 'object': 'text_completion', 'model': model, 'choices': [],
                       'usage': usage}
        events = [chunk(p) for p in result['content']]
        events += [usage_chunk, finish] if self.has('usage-first') else [finish, usage_chunk]
        return 'text/event-stream', ['data: ' + json.dumps(e) + '\n\n' for e in events] + ['data: [DONE]\n\n']

    def message(self, payload, result, prompt):
        content = []
        if result['reasoning']:
            content.append({'type': 'thinking', 'thinking': ''.join(result['reasoning'])})
        if result['content']:
            content.append({'type': 'text', 'text': ''.join(result['content'])})
        for call in result['calls'] or []:
            content.append({'type': 'tool_use', 'id': call['id'], 'name': call['function']['name'],
                            'input': json.loads(call['function']['arguments'])})
        stop_reason = {'stop': 'stop_sequence' if result['matched'] else 'end_turn', 'length': 'max_tokens',
                       'tool_calls': 'tool_use'}[result['finish']]
        if stop_reason == 'max_tokens' and self.has('msg-length-end-turn'):
            stop_reason = 'end_turn'
        usage = self.usage(prompt, result)
        body = {'id': 'msg_fake', 'type': 'message', 'role': 'assistant', 'model': payload['model'],
                'content': content, 'stop_reason': stop_reason, 'stop_sequence': result['matched'],
                'usage': {'input_tokens': usage['prompt_tokens'], 'output_tokens': usage['completion_tokens']}}
        if not payload.get('stream'):
            return 'application/json', [json.dumps(body)]
        if self.has('msg-stop-seq-null') and stop_reason == 'stop_sequence':
            body = {**body, 'stop_reason': 'end_turn', 'stop_sequence': None}
        parts = [part + '\n\n' for part in frames(body).split('\n\n') if part]
        if self.has('msg-start-usage'):
            parts = [start_usage_frame(part, body['usage']) for part in parts]
        if self.has('msg-order'):
            parts[-2], parts[-1] = parts[-1], parts[-2]
        return 'text/event-stream', parts

    def responses(self, payload, result):
        content = [{'type': 'output_text', 'text': ''.join(result['content']), 'annotations': []}]
        output = [{'type': 'message', 'id': 'msg_fake', 'role': 'assistant', 'status': 'completed',
                   'content': content}]
        if result['reasoning']:
            output.insert(0, {'type': 'reasoning', 'id': 'rs_fake', 'summary': [],
                              'content': [{'type': 'reasoning_text', 'text': ''.join(result['reasoning'])}]})
        return 'application/json', [json.dumps({
            'id': 'resp_fake', 'object': 'response', 'status': 'completed', 'model': payload['model'],
            'output': output, 'usage': {'input_tokens': 4, 'output_tokens': result['count']}})]

    # ---- dispatch --------------------------------------------------------
    def post(self, path, payload):
        """Returns (status, content_type, body parts, per-part delay)."""
        if self.has('wedge') and self.interrupted:
            return 500, 'application/json', [json.dumps({'error': 'scheduler wedged'})], 0
        stops = payload.get('stop')
        stops = [stops] if isinstance(stops, str) else stops or []
        if '' in stops and not self.has('accept-empty-stop'):
            return 400, 'application/json', [json.dumps({'error': {
                'message': 'stop cannot contain an empty string.', 'type': 'BadRequestError'}})], 0
        stream = bool(payload.get('stream'))
        if path == '/v1/responses':
            spec = self.respond(payload, payload['input'], [], False)
            if spec is None:
                return 400, 'application/json', [json.dumps({'error': 'unknown fake prompt'})], 0
            result = self.run({**payload, 'max_tokens': payload['max_output_tokens']}, *spec, stream)
            kind, parts = self.responses(payload, result)
            return 200, kind, parts, 0
        if path == '/v1/completions':
            prompt = payload['prompt']
            text = {q3.CMPL_PROMPT: ' Paris. It is also the largest city',
                    q3.CMPL_COUNT_PROMPT: ' 6, 7, 8, 9, 10, 11, 12, 13, 14, 15'}.get(prompt)
            spec = None if text is None else ('', text, None)
        else:
            history = payload['messages']
            prompt = last_user(history)
            if path == '/v1/messages':
                thinking = (payload.get('thinking') or {}).get('type') == 'enabled'
            else:
                thinking = bool((payload.get('chat_template_kwargs') or {}).get('enable_thinking'))
            if prompt == q3.LONG_PROMPT:
                return self.long(path, payload, stream)
            spec = self.respond(payload, prompt, history, thinking)
        if spec is None:
            return 400, 'application/json', [json.dumps({'error': 'unknown fake prompt'})], 0
        result = self.run(payload, *spec, stream)
        if path == '/v1/completions':
            kind, parts = self.completion(payload, result)
        elif path == '/v1/messages':
            kind, parts = self.message(payload, result, prompt)
        else:
            kind, parts = self.chat(payload, result, prompt)
        return 200, kind, parts, 0

    def long(self, path, payload, stream):
        self.interrupted = True
        if not stream and not self.has('fast-long'):
            time.sleep(1.5)  # outlasts the client timeout used by the tests
        result = {'reasoning': [], 'content': tokenize(LONG_TEXT), 'tail': [EOS], 'finish': 'stop',
                  'matched': None, 'count': 121, 'calls': None}
        if path == '/v1/messages':
            kind, parts = self.message(payload, result, q3.LONG_PROMPT)
        else:
            kind, parts = self.chat(payload, result, q3.LONG_PROMPT)
        return 200, kind, parts, (0.02 if stream else 0)

    def get(self, path):
        if path == '/health':
            if self.has('wedge') and self.interrupted:
                return 503, 'application/json', json.dumps({'status': 'faulted'})
            return 200, 'application/json', json.dumps({'status': 'ready', 'model': 'test'})
        if path == '/metrics':
            if self.has('metrics-down'):
                return None  # handler closes the connection without a response
            value = 1 if self.has('gauge') else 0
            return 200, 'text/plain', '# TYPE atlas_requests_active gauge\natlas_requests_active %d\n' % value
        return 404, 'application/json', json.dumps({'error': 'not found'})


def start_usage_frame(part, usage):
    """Anthropic-API framing: input_tokens in message_start, only
    output_tokens in message_delta."""
    name, data = part.split('\ndata: ', 1) if '\ndata: ' in part else (None, None)
    if name == 'event: message_start':
        event = json.loads(data)
        event['message']['usage'] = {'input_tokens': usage['input_tokens'], 'output_tokens': 1}
        return name + '\ndata: ' + json.dumps(event) + '\n\n'
    if name == 'event: message_delta':
        event = json.loads(data)
        event['usage'] = {'output_tokens': usage['output_tokens']}
        return name + '\ndata: ' + json.dumps(event) + '\n\n'
    return part


def make_handler(fake):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            reply = fake.get(self.path)
            if reply is None:
                self.close_connection = True
                return
            status, kind, raw = reply
            self.reply(status, kind, [raw], 0)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            self.reply(*fake.post(self.path, payload))

        def reply(self, status, kind, parts, delay):
            try:
                self.send_response(status)
                self.send_header('Content-Type', kind)
                self.end_headers()  # HTTP/1.0: the body ends when the connection closes
                for part in parts:
                    self.wfile.write(part.encode())
                    self.wfile.flush()
                    if delay:
                        time.sleep(delay)
            except (BrokenPipeError, ConnectionResetError):
                pass  # client cancelled or timed out
    return Handler


@contextlib.contextmanager
def serve(mode='ok'):
    fake = Fake(mode)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(fake))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield 'http://127.0.0.1:%d' % server.server_port
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def run(mode, only=None, eos=(EOS,)):
    with serve(mode) as endpoint, tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'receipts'
        summary = q3.run_suite(endpoint, 'test', out, only, timeout=10, eos_token_ids=eos,
                               client_timeout=0.5, gauge_wait_s=0.3)
        receipts = {name: json.loads((out / (name + '.json')).read_text()) for name in summary['cases']}
        return summary, receipts


def statuses(receipt):
    return {a['assertion']: a['status'] for a in receipt['assertions']}


# (fault mode, case, assertion that must fail). Each fault is absent from the
# ok fixture (which passes every case), so each row fails only because the
# suite detected the injected defect.
FAULTS = [
    ('wrong-answer', 'chat-basic', 'blocking:content'),
    ('usage-first', 'chat-basic', 'stream:usage_chunk_last'),
    ('usage-sum', 'chat-basic', 'blocking:usage'),
    ('eos-leak', 'chat-basic', 'blocking:no_special_tokens'),
    ('short-length', 'chat-length', 'blocking:max_tokens_consumed'),
    ('usage-first', 'completions-basic', 'stream:usage_chunk_last'),
    ('ignore-stop', 'completions-basic', 'blocking:text'),
    ('short-length', 'completions-length', 'stream:max_tokens_consumed'),
    ('trim-stop', 'stop-string', 'blocking:stopped_at_match'),
    ('trim-stop', 'stop-string', 'blocking:fewer_tokens_than_baseline'),
    ('stream-stop-leak', 'stop-string', 'stream:stopped_at_match'),
    ('stream-stop-leak', 'stop-string', 'stream_matches_blocking'),
    ('exclude-stop-str', 'stop-include', 'blocking:stop_string_included'),
    ('exclude-stop-str', 'stop-include', 'stream:stop_string_included'),
    ('stream-stop-leak', 'stop-include', 'stream:stop_string_included'),
    ('accept-empty-stop', 'stop-empty', 'chat-1:rejected_400'),
    ('accept-empty-stop', 'stop-empty', 'chat-2:rejected_400'),
    ('accept-empty-stop', 'stop-empty', 'completions:rejected_400'),
    ('accept-empty-stop', 'stop-empty', 'chat-string:rejected_400'),
    ('responses-drop-seed', 'responses-seed', 'seeded:identical_output'),
    ('greedy-sampling', 'responses-seed', 'seeded:other_seed_differs'),
    ('ignore-stop-ids', 'stop-token-ids', 'stopped:completion_tokens'),
    ('ignore-stop-ids', 'stop-token-ids', 'stopped:content_truncated'),
    ('no-token-ids', 'stop-token-ids', 'baseline:token_ids_exposed'),
    ('eos-leak', 'eos', 'stream:no_special_tokens'),
    ('wrong-answer', 'messages-end-turn', 'blocking:text'),
    ('msg-order', 'messages-end-turn', 'stream:event_order'),
    ('msg-length-end-turn', 'messages-max-tokens', 'blocking:stop_reason'),
    ('msg-stop-seq-null', 'messages-stop-sequence', 'stream:stop_sequence'),
    ('msg-stop-seq-null', 'messages-stop-sequence', 'stream:stop_reason'),
    ('trim-stop', 'messages-stop-sequence', 'blocking:stopped_at_match'),
    ('seed-drift', 'seed', 'seeded:first_token'),
    ('seed-drift', 'seed', 'seeded:identical_output'),
    ('seed-drift', 'seed-json-schema', 'seeded:identical_output'),
    ('greedy-sampling', 'seed', 'seeded:other_seed_differs'),
    ('schema-fence', 'structured-json-schema', 'blocking:json_parses'),
    ('schema-wrong-type', 'structured-json-schema', 'stream:schema_valid'),
    ('schema-fence', 'structured-json-object', 'blocking:json_parses'),
    ('tool-bad-args', 'tools-parallel', 'blocking:arguments'),
    ('tool-one-call', 'tools-parallel', 'stream:call_count'),
    ('followup-ignored', 'tools-followup', 'followup:uses_tool_result'),
    ('think-leak', 'reasoning-on', 'stream:no_special_tokens'),
    ('reasoning-merged', 'reasoning-on', 'blocking:reasoning_present'),
    ('reasoning-uncounted', 'reasoning-on', 'stream:reasoning_accounting'),
    ('forget', 'multi-turn', 'turn-2:recalled'),
    ('wedge', 'cancel-chat', 'health-after'),
    ('wedge', 'cancel-chat', 'next:response'),
    ('gauge', 'cancel-messages', 'metrics:requests_active_zero'),
    ('metrics-down', 'cancel-chat', 'metrics:requests_active_zero'),
    ('fast-long', 'timeout-chat', 'interrupt:client_timed_out'),
    ('wedge', 'timeout-messages', 'next:response'),
]


class SuiteAgainstFakeTests(unittest.TestCase):
    def test_spec_correct_fake_passes_every_case(self):
        summary, receipts = run('ok')
        self.assertEqual(set(summary['cases']), set(q3.CASES))
        failed = {n: r['failed_assertions'] or r['error'] for n, r in receipts.items() if not r['passed']}
        self.assertEqual(failed, {})
        self.assertTrue(summary['passed'])
        skipped = {n: r['skipped_assertions'] for n, r in receipts.items() if r['skipped_assertions']}
        self.assertEqual(skipped, {})  # the fake exposes token ids, the gauge and EOS
        self.assertEqual(statuses(receipts['stop-token-ids'])['stopped:completion_tokens'], 'pass')
        self.assertEqual(statuses(receipts['eos'])['stream:last_token_is_eos'], 'pass')
        self.assertEqual(statuses(receipts['cancel-chat'])['interrupt:cancelled_mid_stream'], 'pass')
        self.assertEqual(statuses(receipts['timeout-chat'])['interrupt:client_timed_out'], 'pass')

    def test_every_fault_fails_its_named_assertion(self):
        by_mode = {}
        for mode, case, assertion in FAULTS:
            by_mode.setdefault(mode, {}).setdefault(case, []).append(assertion)
        for mode, cases in by_mode.items():
            _, receipts = run(mode, list(cases))
            for case, assertions in cases.items():
                for assertion in assertions:
                    with self.subTest(mode=mode, case=case, assertion=assertion):
                        self.assertFalse(receipts[case]['passed'])
                        self.assertEqual(statuses(receipts[case]).get(assertion), 'fail',
                                         receipts[case]['failed_assertions'])

    def test_fault_matrix_covers_every_family(self):
        families = {q3.CASES[case]['family'] for _, case, _ in FAULTS}
        self.assertEqual(families, {spec['family'] for spec in q3.CASES.values()})

    def test_missing_optional_capabilities_are_skipped_not_passed(self):
        _, receipts = run('no-token-ids', ['eos', 'seed'], eos=(EOS,))
        self.assertEqual(statuses(receipts['eos'])['stream:last_token_is_eos'], 'skipped')
        self.assertIn('stream:last_token_is_eos', receipts['eos']['skipped_assertions'])
        # Without token ids the first-token gate falls back to the first delta.
        self.assertEqual(statuses(receipts['seed'])['seeded:first_token'], 'pass')
        _, receipts = run('ok', ['eos'], eos=())
        self.assertEqual(statuses(receipts['eos'])['stream:last_token_is_eos'], 'skipped')
        self.assertTrue(receipts['eos']['passed'])

    def test_benign_wire_variants_pass(self):
        # input_tokens only in message_start, and streamed tool arguments with
        # different whitespace, are both contract-conformant.
        for mode, cases in (('msg-start-usage', ['messages-end-turn', 'messages-stop-sequence']),
                            ('args-spacing', ['tools-single', 'tools-parallel'])):
            _, receipts = run(mode, cases)
            for case in cases:
                with self.subTest(mode=mode, case=case):
                    self.assertTrue(receipts[case]['passed'], receipts[case]['failed_assertions'])

    def test_duplicate_only_names_run_once(self):
        summary, _ = run('ok', ['chat-basic', 'chat-basic'])
        self.assertEqual(list(summary['cases']), ['chat-basic'])

    def test_unreachable_server_records_failures_and_continues(self):
        with serve() as endpoint:
            pass  # server stopped: every request now fails to connect
        with tempfile.TemporaryDirectory() as tmp:
            summary = q3.run_suite(endpoint, 'test', Path(tmp) / 'r', ['chat-basic', 'tools-followup'],
                                   timeout=2)
        self.assertEqual(list(summary['cases']), ['chat-basic', 'tools-followup'])
        self.assertFalse(summary['passed'])
        self.assertIn('blocking:response', summary['cases']['chat-basic']['failed_assertions'])
        self.assertIn('calls:response', summary['cases']['tools-followup']['failed_assertions'])


class CliTests(unittest.TestCase):
    def main(self, argv):
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            try:
                return q3.main(argv), stderr.getvalue()
            except SystemExit as exit_:
                return exit_.code, stderr.getvalue()

    def test_exit_codes_only_and_receipts(self):
        with serve() as endpoint, tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'r'
            code, _ = self.main(['--endpoint', endpoint, '--model', 'm', '--out', str(out),
                                 '--only', 'chat-basic', 'seed'])
            self.assertEqual(code, 0)
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ['chat-basic.json', 'raw', 'seed.json', 'summary.json'])
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual((summary['model'], summary['cases_total']), ('m', 2))
            sent = json.loads((out / 'raw' / 'chat-basic-blocking.request.json').read_text())
            self.assertEqual(sent['model'], 'm')
            code, err = self.main(['--endpoint', endpoint, '--model', 'm', '--out', str(out)])
            self.assertEqual(code, 2)
            self.assertIn('refusing to overwrite', err)
        with serve('wrong-answer') as endpoint, tempfile.TemporaryDirectory() as tmp:
            code, _ = self.main(['--endpoint', endpoint, '--model', 'm', '--out', str(Path(tmp) / 'r'),
                                 '--only', 'chat-basic'])
            self.assertEqual(code, 1)

    def test_unknown_case_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'r'
            code, err = self.main(['--endpoint', 'http://127.0.0.1:9', '--model', 'm', '--out', str(out),
                                   '--only', 'no-such-case'])
            self.assertEqual(code, 2)
            self.assertIn('no-such-case', err)
            self.assertFalse(out.exists())


class PureCheckTests(unittest.TestCase):
    def test_usage_problems(self):
        good = {'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8}
        self.assertEqual(q3.usage_problems(good, 3), [])
        self.assertTrue(q3.usage_problems({**good, 'total_tokens': 9}, 3))
        self.assertTrue(q3.usage_problems(good, 2))
        self.assertTrue(q3.usage_problems({**good, 'prompt_tokens': True}, 3))
        self.assertTrue(q3.usage_problems(None, 3))

    def test_usage_chunk_last(self):
        delta = {'choices': [{'index': 0, 'delta': {'content': 'a'}, 'finish_reason': None}]}
        finish = {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}
        usage = {'choices': [], 'usage': {'total_tokens': 1}}
        self.assertTrue(q3.usage_chunk_last([delta, finish, usage])[0])
        self.assertFalse(q3.usage_chunk_last([delta, usage, finish])[0])
        self.assertFalse(q3.usage_chunk_last([delta, {**finish, 'usage': {'total_tokens': 1}}])[0])

    def test_stop_token_choice_skips_earlier_duplicates_and_last_token(self):
        ids = [7, 8, 9, 9, 10, 11]
        # content at positions 1..5; content index >= 2 starts at position 3,
        # whose id 9 already occurred at position 2 -> position 4.
        self.assertEqual(q3.pick_stop_token(ids, [1, 2, 3, 4, 5]), 4)
        self.assertIsNone(q3.pick_stop_token([1, 2, 3], [0, 1, 2]))

    def test_stream_token_ids_split_content_from_reasoning(self):
        events = [{'choices': [{'delta': {'reasoning_content': 'r'}, 'token_ids': [5]}]},
                  {'choices': [{'delta': {'content': 'c'}, 'token_ids': [6, 7]}]},
                  {'choices': [{'delta': {}, 'finish_reason': 'stop', 'token_ids': [2]}]}]
        self.assertEqual(q3.stream_token_ids(events), ([5, 6, 7, 2], [1, 2]))
        self.assertEqual(q3.stream_token_ids([{'choices': [{'delta': {'content': 'c'}}]}]), (None, None))

    def test_stop_cut(self):
        base = 'Thought: check the weather\nObservation: sunny and 21 C\nFinal: it is sunny'
        self.assertEqual(q3.stop_cut_problems('Thought: check the weather\n', base), [])
        self.assertTrue(q3.stop_cut_problems('Thought: check the weather\n sunny and 21 C\nFinal: it is sunny',
                                             base))
        self.assertTrue(q3.stop_cut_problems(base, base))
        self.assertTrue(q3.stop_cut_problems('Thought: check', base))
        self.assertTrue(q3.stop_cut_problems('Thought: check the weather\n', 'Thought: no marker'))

    def test_include_cut(self):
        base = 'Thought: check the weather\nObservation: sunny and 21 C\nFinal: it is sunny'
        self.assertEqual(q3.include_cut_problems('Thought: check the weather\nObservation:', base), [])
        self.assertTrue(q3.include_cut_problems('Thought: check the weather\n', base))  # stop cut off
        self.assertTrue(q3.include_cut_problems(base, base))  # ran past the stop
        self.assertTrue(q3.include_cut_problems('Thought:  check the weather\nObservation:', base))
        self.assertTrue(q3.include_cut_problems('Observation: Observation:', None))

    def test_responses_text(self):
        body = {'output': [
            {'type': 'reasoning', 'summary': [{'type': 'summary_text', 'text': 's'}],
             'content': [{'type': 'reasoning_text', 'text': 'r'}]},
            {'type': 'message', 'content': [{'type': 'output_text', 'text': 'a'},
                                            {'type': 'refusal', 'refusal': 'x'},
                                            {'type': 'output_text', 'text': 'b'}]}]}
        self.assertEqual(q3.responses_text(body), 'ab')
        self.assertEqual(q3.responses_text(body, 'reasoning'), 'sr')
        self.assertEqual(q3.responses_text({'output': None}), '')
        self.assertEqual(q3.responses_text(None), '')

    def test_message_event_order(self):
        body = {'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'model': 'm',
                'content': [{'type': 'text', 'text': 'Hi'}], 'stop_reason': 'end_turn',
                'stop_sequence': None, 'usage': {'input_tokens': 1, 'output_tokens': 1}}
        raw = frames(body)
        self.assertEqual(q3.message_event_order(raw)[0], [])
        parts = [p for p in raw.split('\n\n') if p]
        parts[-1], parts[-2] = parts[-2], parts[-1]
        self.assertTrue(q3.message_event_order('\n\n'.join(parts) + '\n\n')[0])
        self.assertTrue(q3.message_event_order('data: {}\n\n')[0])

    def test_leaks(self):
        self.assertEqual(q3.leaks('Tokyo'), [])
        self.assertTrue(q3.leaks('</think>Tokyo'))
        self.assertTrue(q3.leaks('Tokyo<|user|>'))

    def test_tool_call_problems(self):
        call = {'id': 'c1', 'type': 'function', 'function': {
            'name': 'lookup_inventory', 'arguments': json.dumps(q3.INVENTORY_ARGUMENTS)}}
        expected = [('lookup_inventory', q3.INVENTORY_ARGUMENTS)]
        self.assertEqual(q3.tool_call_problems([call], expected)[0], [])
        bad = {**call, 'function': {**call['function'], 'arguments': '{"sku": "BLUE-17"}'}}
        self.assertTrue(q3.tool_call_problems([bad], expected)[0])
        self.assertTrue(q3.tool_call_problems([call, call], expected * 2)[0])  # duplicate ids
        self.assertTrue(q3.tool_call_problems(None, expected)[0])


if __name__ == '__main__':
    unittest.main()
