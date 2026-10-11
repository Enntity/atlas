# SPDX-License-Identifier: AGPL-3.0-only
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from collect_behavior import collect
from test_parity import provenance
from test_behavior_surfaces import make_handler

EXISTING_CASES = 12


class BehavioralHttpTests(unittest.TestCase):
    def run_capture(self,wrong=False,broken_stream=False):
        seen=[]
        mode = 'wrong' if wrong else ('broken-stream' if broken_stream else 'ok')
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(mode,seen))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'capture';p=provenance('atlas')
                p['serve_flags']=['serve','test','--max-seq-len','4096','--max-prefill-tokens','128',
                                  '--max-batch-size','1','--max-num-seqs','1']
                report=collect('http://127.0.0.1:'+str(server.server_port),'test',p,out,4096,4,5)
                total=len(report['requests'])
                self.assertGreater(total,EXISTING_CASES)  # surface cases join the same manifest
                posted=[row for row in report['requests'] if 'payload' in row]
                fabricated=[row for row in report['requests'] if 'raw_file' not in row]
                self.assertEqual(len(list(out.glob('*.request.json'))),len(posted))
                self.assertEqual(len(list(out.glob('*.response.raw'))),total-len(fabricated))
                self.assertTrue(all(not row['passed'] for row in fabricated))
                self.assertEqual(json.loads((out/'manifest.json').read_text()),report)
                self.assertEqual(report['graph']['status'],'SKIP_UNQUALIFIED')
                self.assertEqual(report['concurrency_scope'],'simultaneous HTTP; serial C1 GPU admission')
                return report,seen
        finally:
            server.shutdown();thread.join();server.server_close()

    def test_actual_http_filesystem_roundtrip_stream_and_distinct_queued_requests(self):
        report,seen=self.run_capture();self.assertTrue(report['passed'],report)
        followups=[pl for _,pl in seen if 'messages' in pl and pl['messages'][-1].get('role')=='tool']
        self.assertEqual(len(followups),1)
        self.assertEqual(followups[0]['messages'][-1]['tool_call_id'],'call-owned')
        self.assertEqual(followups[0]['tool_choice'],'none')
        paths=[path for path,_ in seen]
        self.assertIn('/v1/chat/completions',paths)
        self.assertIn('/v1/completions',paths)  # legacy surface exercised
        self.assertEqual(len(seen),len([row for row in report['requests'] if 'payload' in row]))
        self.assertIn('/v1/messages',paths)

    def test_incomplete_actual_http_sse_retains_raw_and_finishes_remaining_checks(self):
        report,seen=self.run_capture(broken_stream=True)
        self.assertFalse(report['passed'])
        self.assertEqual(len(seen),len([row for row in report['requests'] if 'payload' in row]))
        for row in report['requests']:
            if row['id'] in ('stream-chat','stream-tool'):
                self.assertFalse(row['passed']);self.assertIn('incomplete SSE',row['error'])
        blocked={row['id']:row for row in report['requests'] if row['id']=='chat-stop'}
        self.assertTrue(blocked['chat-stop']['passed'])  # blocking path unaffected

    def test_wrong_visible_answer_is_retained_and_fails_gate(self):
        report,_=self.run_capture(True);self.assertFalse(report['passed'])
        self.assertTrue(any(r['error'] and 'answer' in r['error'] for r in report['requests']))

    def test_server_template_default_voids_the_profile(self):
        p=provenance('atlas')
        p['serve_flags']=['serve','test','--max-seq-len','4096','--max-prefill-tokens','128',
                          '--max-batch-size','1','--max-num-seqs','1',
                          '--default-chat-template-kwargs={"enable_thinking":true}']
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'default-chat-template-kwargs'):
                collect('http://127.0.0.1:9','test',p,Path(tmp)/'capture',4096,1,5)
            self.assertFalse((Path(tmp)/'capture').exists())

if __name__=='__main__':
    unittest.main()
