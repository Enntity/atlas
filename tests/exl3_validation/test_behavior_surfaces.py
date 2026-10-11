# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for the OpenAI chat + legacy completions behavior surfaces.

Exercises collect_surfaces against a local fake HTTP server (no GPU, no
numpy): the happy matrix must pass with objective summaries, and each
contract violation (special-token leak, seeded-repeat divergence,
stream/blocking accounting drift, wrong finish reason, incomplete SSE)
must fail exactly its own receipts without aborting the run.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

from behavior_io import BehaviorIORouter
from behavior_surfaces import (ANSWER, CALC_ANSWER, CODEWORD, LENGTH_TOKENS,
                               STOP_MARKER, collect_surfaces)

JOKE = 'Why did the scarecrow win? He was outstanding in his field.'
COUNT_TEXT = ' '.join(str(i) for i in range(1, 31))
MATH = {'37 * 14': '432', '91 - 38': '53', '12 * 13': '156', '144 / 12': '12'}
CHAT_COUNT = 18
CMPL_COUNT = 12
TOTAL = CHAT_COUNT + CMPL_COUNT
SAFE_LEAK = '[gMASK]'
SPECIAL_LEAK = '<|user|>'


def fake_reply(payload, mode='ok', calls=None):
    """Deterministic fake Atlas reply keyed on prompt content.

    Modes: ok | wrong | leak | flaky-seed | drift-usage | wrong-finish |
    broken-stream | usage-first.
    """
    calls = {'jokes': 0} if calls is None else calls
    chat = 'messages' in payload
    last = payload['messages'][-1] if chat else None
    prompt = last['content'] if chat else payload['prompt']
    stream = payload.get('stream', False)
    model = payload['model']

    def emit(body, content_type='application/json'):
        raw = json.dumps(body).encode()
        return content_type, raw

    def emit_sse(events):
        raw = ''.join('data: ' + json.dumps(e) + '\n\n' for e in events)
        if mode != 'broken-stream':
            raw += 'data: [DONE]\n\n'
        return 'text/event-stream', raw.encode()

    def usage_for(text, reasoning):
        completion = max(1, len(text.split()) + len(reasoning.split()))
        if mode == 'drift-usage' and stream:
            completion += 1
        return {'prompt_tokens': 7, 'completion_tokens': completion,
                'total_tokens': 7 + completion,
                'completion_tokens_details': {'reasoning_tokens': len(reasoning.split())}}

    # Tool turns keep the established contract (id 'call-owned', fixed args).
    if chat and payload.get('tools') and payload.get('tool_choice') != 'none':
        call = {'id': 'call-owned', 'type': 'function', 'function': {
            'name': 'lookup_inventory',
            'arguments': '{"sku":"BLUE-17","warehouse":"north"}'}}
        usage = {'prompt_tokens': 7, 'completion_tokens': 2, 'total_tokens': 9,
                 'completion_tokens_details': {'reasoning_tokens': 0}}
        if stream:
            events = [
                {'id': 'chat-test', 'model': model, 'choices': [{'index': 0,
                  'delta': {'role': 'assistant', 'tool_calls': [{'index': 0, **call}]},
                  'finish_reason': None}]},
                {'id': 'chat-test', 'model': model, 'choices': [{'index': 0,
                  'delta': {}, 'finish_reason': 'tool_calls'}]},
                {'id': 'chat-test', 'model': model, 'choices': [], 'usage': usage}]
            return emit_sse(events)
        return emit({'id': 'chat-test', 'object': 'chat.completion', 'model': model,
                     'choices': [{'index': 0, 'finish_reason': 'tool_calls',
                                  'message': {'role': 'assistant', 'content': None,
                                              'tool_calls': [call]}}],
                     'usage': usage})

    if chat and last.get('role') == 'tool':
        text, reasoning = '19', ''
    elif '37 * 14' in prompt:
        text, reasoning = ('433' if mode == 'wrong' else '432'), 'Compute the product first.'
    elif any(key in prompt for key in MATH):
        text, reasoning = next(v for k, v in MATH.items() if k in prompt), ''
    elif 'capital of Japan' in prompt:
        text, reasoning = ANSWER, ''
    elif 'Count from 1 to 30' in prompt:
        text, reasoning = COUNT_TEXT, ''
    elif 'alpha STOPNOW' in prompt:
        text, reasoning = 'alpha STOPNOW omega', ''
    elif 'What code word' in prompt:
        text, reasoning = CODEWORD, ''
    elif 'Remember this code word' in prompt:
        text, reasoning = 'OK', ''
    elif 'joke' in prompt:
        calls['jokes'] += 1
        text, reasoning = JOKE + (str(calls['jokes']) if mode == 'flaky-seed' else ''), ''
    else:
        text, reasoning = 'unmatched prompt', ''
    if chat and payload.get('chat_template_kwargs', {}).get('enable_thinking'):
        reasoning = reasoning or 'Compute the answer.'
    else:
        reasoning = ''
    finish = 'stop'
    for stop in payload.get('stop') or []:
        if stop in text:
            text = text[:text.index(stop)]
            break
    else:
        if len(text.split()) > payload['max_tokens']:
            text = ' '.join(text.split()[:payload['max_tokens']])
            finish = 'length'
    if mode == 'wrong-finish':
        finish = 'stop'
    if mode == 'leak':
        text = text + SAFE_LEAK + SPECIAL_LEAK
    usage = usage_for(text, reasoning)
    if stream:
        events = []
        if chat:
            def chunk(delta):
                return {'id': 'chat-test', 'model': model,
                        'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]}
            events.append(chunk({'role': 'assistant'}))
            if reasoning:
                events.append(chunk({'reasoning_content': reasoning}))
            events.append(chunk({'content': text}))
            tail = [{'id': 'chat-test', 'model': model, 'choices': [
                        {'index': 0, 'delta': {}, 'finish_reason': finish}]},
                    {'id': 'chat-test', 'model': model, 'choices': [], 'usage': usage}]
            # 'usage-first' mimics encode_stream.rs (usage chunk BEFORE the
            # finish chunk), the reverse of OpenAI's include_usage framing.
            events.extend(reversed(tail) if mode == 'usage-first' else tail)
        else:
            def cmpl(text_value, fin=None):
                return {'id': 'cmpl-test', 'object': 'text_completion', 'model': model,
                        'choices': [{'index': 0, 'text': text_value,
                                     **({'finish_reason': fin} if fin else {})}]}
            events.append(cmpl(text))
            events.append(cmpl('', finish))
            events.append({'id': 'cmpl-test', 'object': 'text_completion',
                           'model': model, 'choices': [], 'usage': usage})
        return emit_sse(events)
    if chat:
        message = {'role': 'assistant', 'content': text}
        if reasoning:
            message['reasoning_content'] = reasoning
        return emit({'id': 'chat-test', 'object': 'chat.completion', 'model': model,
                     'choices': [{'index': 0, 'message': message,
                                  'finish_reason': finish}], 'usage': usage})
    return emit({'id': 'cmpl-test', 'object': 'text_completion', 'model': model,
                 'choices': [{'index': 0, 'text': text, 'finish_reason': finish}],
                 'usage': usage, 'system_fingerprint': 'fp_atlas'})


def make_handler(mode='ok', seen=None):
    """HTTP handler wrapping fake_reply (and the Q3 fake for tools,
    structured output, /v1/messages, probes and interruptions); records
    (path, payload) per POST."""
    from fake_atlas_q3 import q3_get, q3_reply
    calls = {'jokes': 0, 'interrupted': 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, content_type, raw):
            try:
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client hung up (cancel / timeout cases)

        def do_GET(self):
            self.send(*q3_get(self.path, mode, calls))

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if seen is not None:
                seen.append((self.path, payload))
            reply = q3_reply(self.path, payload, mode, calls, fake_reply)
            if reply is None:
                reply = (200,) + fake_reply(payload, mode, calls)
            self.send(*reply)

    return Handler


class SurfaceMatrixTests(unittest.TestCase):
    def collect_with(self, mode):
        seen = []
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(mode, seen))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                io = BehaviorIORouter('http://127.0.0.1:' + str(server.server_port),
                                      Path(tmp) / 'capture', 5)
                rows = collect_surfaces(io, 'test', 4096)
                by_id = {row['id']: row for row in rows}
                return by_id, seen
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_full_matrix_passes_with_objective_summaries(self):
        rows, seen = self.collect_with('ok')
        self.assertEqual(len(rows), TOTAL)
        self.assertTrue(all(row['passed'] for row in rows.values()))
        self.assertEqual(len(seen), TOTAL)
        paths = [path for path, _ in seen]
        self.assertEqual(paths.count('/v1/chat/completions'), CHAT_COUNT)
        self.assertEqual(paths.count('/v1/completions'), CMPL_COUNT)
        # finish_reason stop vs length + max_tokens/length accounting
        self.assertEqual(rows['chat-length']['summary']['finish_reason'], 'length')
        self.assertEqual(rows['chat-length']['summary']['accounting']['completion_tokens'],
                         LENGTH_TOKENS)
        self.assertEqual(rows['cmpl-stream-length']['summary']['finish_reason'], 'length')
        self.assertEqual(rows['chat-stop']['summary']['finish_reason'], 'stop')
        # stop sequences stripped from both surfaces, both modes
        self.assertEqual(rows['chat-stop-seq']['summary']['content'], 'alpha ')
        self.assertNotIn(STOP_MARKER, rows['cmpl-stop-seq']['summary']['text'])
        self.assertNotIn(STOP_MARKER, rows['chat-stream-stop-seq']['summary']['content'])
        # multi-turn recall
        self.assertIn(CODEWORD, rows['chat-multiturn-recall']['summary']['content'])
        self.assertIn(CODEWORD, rows['chat-stream-multiturn-recall']['summary']['content'])
        self.assertIn(CODEWORD, rows['cmpl-multiturn-recall']['summary']['text'])
        # reasoning on/off separation (reasoning_content != content)
        self.assertEqual(rows['chat-reason-on']['summary']['content'].strip(), CALC_ANSWER)
        self.assertTrue(rows['chat-reason-on']['summary']['reasoning'])
        self.assertGreater(rows['chat-reason-on']['summary']['reasoning_tokens'], 0)
        self.assertEqual(rows['chat-stream-reason-off']['summary']['content'].strip(),
                         CALC_ANSWER)
        # seeded repeats + stream/blocking token-accounting identities
        self.assertTrue(all(rows['chat-seeded-b']['seeded_identity'].values()))
        self.assertTrue(all(rows['cmpl-stream-seeded-b']['seeded_identity'].values()))
        self.assertTrue(all(rows['chat-stream-stop']['blocking_identity'].values()))
        self.assertTrue(all(rows['cmpl-stream-stop']['blocking_identity'].values()))

    def test_leaked_special_tokens_fail_every_text_channel(self):
        rows, _ = self.collect_with('leak')
        for name in ('chat-stop', 'chat-stream-stop', 'cmpl-stop', 'cmpl-stream-stop'):
            self.assertFalse(rows[name]['passed'], name)
            self.assertIn('special token leaked', rows[name]['error'], name)
        self.assertEqual(len(rows), TOTAL)  # the run continues; failures are retained

    def test_seeded_repeat_divergence_fails_the_pair(self):
        rows, _ = self.collect_with('flaky-seed')
        for name in ('chat-seeded-b', 'chat-stream-seeded-b',
                     'cmpl-seeded-b', 'cmpl-stream-seeded-b'):
            self.assertFalse(rows[name]['passed'], name)
            self.assertIn('seeded_identity mismatch', rows[name]['error'], name)
        self.assertTrue(rows['chat-stop']['passed'])  # unrelated cases unaffected

    def test_stream_blocking_accounting_drift_fails_the_pair(self):
        rows, _ = self.collect_with('drift-usage')
        for name in ('chat-stream-stop', 'cmpl-stream-stop'):
            self.assertFalse(rows[name]['passed'], name)
            self.assertIn('blocking_identity mismatch', rows[name]['error'], name)

    def test_wrong_finish_reason_fails_validation(self):
        rows, _ = self.collect_with('wrong-finish')
        for name in ('chat-length', 'cmpl-length', 'cmpl-stream-length'):
            self.assertFalse(rows[name]['passed'], name)
            self.assertIn('wrong finish', rows[name]['error'], name)

    def test_usage_before_finish_fails_the_one_order_gate(self):
        rows, _ = self.collect_with('usage-first')
        self.assertFalse(rows['chat-stream-stop']['passed'])
        self.assertIn('include_usage chunk must follow', rows['chat-stream-stop']['error'])
        self.assertEqual(rows['chat-stream-stop']['sse_shape'][-2:], ['usage', 'finish'])
        # content gates on every other stream case are not masked
        self.assertTrue(rows['chat-stream-reason-on']['passed'])
        self.assertTrue(rows['cmpl-stream-stop']['passed'])

    def test_incomplete_sse_fails_stream_only(self):
        rows, _ = self.collect_with('broken-stream')
        self.assertFalse(rows['chat-stream-stop']['passed'])
        self.assertIn('incomplete SSE', rows['chat-stream-stop']['error'])
        self.assertFalse(rows['cmpl-stream-stop']['passed'])
        self.assertTrue(rows['chat-stop']['passed'])  # blocking unaffected


if __name__ == '__main__':
    unittest.main()
