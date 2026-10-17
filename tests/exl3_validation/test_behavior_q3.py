# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for the Q3 tool / structured-output / Anthropic / robustness
behavior cases against a local fake server (no GPU, no numpy, no network
beyond 127.0.0.1). The spec-correct fake must pass every case; each injected
defect must fail exactly the receipts that own it."""
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

from behavior_io import BehaviorIORouter
from behavior_messages import collect_messages
from behavior_robustness import collect_robustness
from behavior_tools import collect_tools
from test_behavior_surfaces import make_handler

TOOL_CASES = {'tools-required', 'tools-named-stream', 'tools-none', 'tools-auto-no-call',
              'tools-invalid-choice', 'tools-parallel', 'tools-parallel-stream',
              'tools-parallel-followup', 'format-json-schema', 'format-json-schema-stream',
              'format-json-object'}


class Q3Tests(unittest.TestCase):
    def collect_with(self, collector, mode='ok'):
        seen = []
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(mode, seen))
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                io = BehaviorIORouter('http://127.0.0.1:' + str(server.server_port),
                                      Path(tmp) / 'capture', 5)
                rows = collector(io, 'test', 4096)
                return {row['id']: row for row in rows}, seen
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def failed(self, rows):
        return {name for name, row in rows.items() if not row['passed']}

    # ---- tools + structured output ---------------------------------------
    def test_tools_and_formats_pass_with_objective_summaries(self):
        rows, seen = self.collect_with(collect_tools)
        self.assertEqual(set(rows), TOOL_CASES)
        self.assertEqual(self.failed(rows), set(), {k: rows[k]['error'] for k in self.failed(rows)})
        calls = rows['tools-parallel-stream']['summary']['checked']
        self.assertEqual(sorted(c['arguments']['city'] for c in calls), ['Paris', 'Tokyo'])
        self.assertTrue(rows['tools-parallel-stream']['blocking_identity']['tool_calls'])
        self.assertEqual(rows['tools-parallel-followup']['summary']['checked'], 'Paris')
        followup = [p for _, p in seen if p['messages'][-1].get('content', '').startswith('Using only')]
        self.assertEqual([m['tool_call_id'] for m in followup[0]['messages'] if m['role'] == 'tool'],
                         ['call_0000000000000000', 'call_0000000000000001'])
        self.assertEqual(rows['format-json-schema']['summary']['checked']['city'], 'Tokyo')
        self.assertEqual(rows['tools-invalid-choice']['http_status'], 400)

    def test_fenced_structured_output_fails(self):
        rows, _ = self.collect_with(collect_tools, 'fenced')
        self.assertEqual(self.failed(rows), {'format-json-schema', 'format-json-schema-stream',
                                             'format-json-object'})

    def test_out_of_schema_arguments_fail_and_block_the_followup(self):
        rows, _ = self.collect_with(collect_tools, 'bad-args')
        self.assertIn('schema violation', rows['tools-parallel']['error'])
        self.assertIn('schema violation', rows['tools-parallel-stream']['error'])
        self.assertIn('no fabricated tool IDs', rows['tools-parallel-followup']['error'])
        self.assertTrue(rows['tools-required']['passed'])

    def test_reasoning_on_tool_turn_without_directive_fails(self):
        rows, _ = self.collect_with(collect_tools, 'tool-reasoning')
        for name in ('tools-required', 'tools-named-stream', 'tools-parallel', 'tools-parallel-stream'):
            self.assertIn('thinking_in_tools', rows[name]['error'], name)

    # ---- Anthropic /v1/messages ------------------------------------------
    def test_messages_matrix_and_openai_parity_pass(self):
        rows, seen = self.collect_with(collect_messages)
        self.assertEqual(self.failed(rows), set(), {k: rows[k]['error'] for k in self.failed(rows)})
        self.assertEqual(rows['msg-stream-stop-seq']['summary']['stop_sequence'], 'STOPNOW')
        self.assertEqual(rows['msg-think-on']['summary']['blocks'], ['thinking', 'text'])
        self.assertTrue(all(rows['msg-stream-think-on']['blocking_identity'].values()))
        self.assertTrue(all(rows['msg-parity-chat']['openai_parity'].values()))
        self.assertEqual(rows['msg-tool-result']['summary']['text'], '19')
        result = [p for path, p in seen if path == '/v1/messages'
                  and isinstance(p['messages'][-1]['content'], list)][0]
        self.assertEqual(result['messages'][-1]['content'][0]['tool_use_id'], 'call-owned')
        self.assertEqual(rows['msg-stream-stop']['sse_shape'][0], 'message_start')
        self.assertEqual(rows['msg-stream-stop']['sse_shape'][-1], 'message_stop')

    def test_streamed_stop_sequence_drift_fails_only_that_case(self):
        rows, _ = self.collect_with(collect_messages, 'anthropic-drift')
        self.assertEqual(self.failed(rows), {'msg-stream-stop-seq'})
        self.assertIn('stop_reason', rows['msg-stream-stop-seq']['error'])

    # ---- cancellation / timeout / recovery --------------------------------
    def test_interruptions_recover_with_identical_output(self):
        rows, seen = self.collect_with(collect_robustness)
        self.assertEqual(self.failed(rows), set(), {k: rows[k]['error'] for k in self.failed(rows)})
        self.assertEqual(rows['robust-cancel-chat']['frames_before_cancel'], 3)
        self.assertFalse(rows['robust-timeout-chat']['completed_before_timeout'])
        self.assertEqual(rows['robust-active-gauge']['atlas_requests_active'], ['0'])
        self.assertEqual(len([1 for path, _ in seen if path == '/v1/messages']), 5)

    def test_state_leak_after_interruption_fails_the_next_request(self):
        rows, _ = self.collect_with(collect_robustness, 'leaky')
        self.assertEqual(self.failed(rows), {'robust-cancel-chat-next'})

    def test_pinned_gauge_and_unready_health_fail(self):
        rows, _ = self.collect_with(collect_robustness, 'gauge')
        self.assertEqual(self.failed(rows), {'robust-active-gauge'})
        rows, _ = self.collect_with(collect_robustness, 'not-ready')
        self.assertEqual(self.failed(rows), {name for name in rows if name.endswith('-health')})


if __name__ == '__main__':
    unittest.main()
