# SPDX-License-Identifier: AGPL-3.0-only
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from collect_behavior import collect
from test_behavior import body,stream
from test_parity import provenance


class BehavioralHttpTests(unittest.TestCase):
    def run_capture(self,wrong=False,broken_stream=False):
        seen=[];lock=threading.Lock()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with lock:seen.append(payload)
                prompt=payload['messages'][-1]['content'];answer='Tokyo'
                if payload['messages'][-1]['role']=='tool':answer='19'
                elif '37 * 14' in prompt:answer='432'
                elif '91 - 38' in prompt:answer='53'
                elif '12 * 13' in prompt:answer='156'
                elif '144 / 12' in prompt:answer='12'
                if wrong and '37 * 14' in prompt:answer='433'
                result=body(answer)
                if payload.get('tools') and payload.get('tool_choice')!='none':
                    result=body(None,'tool_calls');result['choices'][0]['message']['tool_calls']=[{
                        'id':'call-owned','type':'function','function':{'name':'lookup_inventory',
                        'arguments':'{"sku":"BLUE-17","warehouse":"north"}'}}]
                if payload.get('chat_template_kwargs',{}).get('enable_thinking'):
                    result['choices'][0]['message']['reasoning_content']='Calculate.'
                    result['usage']['completion_tokens_details']['reasoning_tokens']=1
                if payload.get('stream'):
                    if result['choices'][0]['finish_reason']=='tool_calls':
                        events=[{'id':'chat-test','model':'test','choices':[{'index':0,'delta':{
                            'role':'assistant','tool_calls':[{'index':0,**result['choices'][0]['message']['tool_calls'][0]}]},'finish_reason':None}]},
                            {'id':'chat-test','model':'test','choices':[{'index':0,'delta':{},'finish_reason':'tool_calls'}]},
                            {'id':'chat-test','model':'test','choices':[],'usage':result['usage']}]
                        raw=(''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n').encode()
                    else:raw=stream(result).encode()
                    if broken_stream:raw=raw.replace(b'data: [DONE]',b'')
                    content_type='text/event-stream'
                else:raw=json.dumps(result).encode();content_type='application/json'
                self.send_response(200);self.send_header('Content-Type',content_type)
                self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'capture';p=provenance('atlas')
                p['serve_flags']=['serve','test','--max-seq-len','4096','--max-prefill-tokens','128',
                                  '--max-batch-size','1','--max-num-seqs','1']
                report=collect('http://127.0.0.1:'+str(server.server_port),'test',p,out,4096,4,5)
                self.assertEqual(len(list(out.glob('*.request.json'))),12)
                self.assertEqual(len(list(out.glob('*.response.raw'))),12)
                self.assertEqual(json.loads((out/'manifest.json').read_text()),report)
                self.assertEqual(report['graph']['status'],'SKIP_UNQUALIFIED')
                self.assertEqual(report['concurrency_scope'],'simultaneous HTTP; serial C1 GPU admission')
                return report,seen
        finally:
            server.shutdown();thread.join();server.server_close()

    def test_actual_http_filesystem_roundtrip_stream_and_distinct_queued_requests(self):
        report,seen=self.run_capture();self.assertTrue(report['passed'],report)
        followups=[p for p in seen if p['messages'][-1]['role']=='tool'];self.assertEqual(len(followups),1)
        self.assertEqual(followups[0]['messages'][-1]['tool_call_id'],'call-owned')
        self.assertEqual(followups[0]['tool_choice'],'none')
        self.assertEqual(len(report['requests']),12)

    def test_incomplete_actual_http_sse_retains_raw_and_finishes_remaining_checks(self):
        report,seen=self.run_capture(broken_stream=True)
        self.assertFalse(report['passed']);self.assertEqual(len(seen),12)
        for row in report['requests']:
            if row['id'] in ('stream-chat','stream-tool'):
                self.assertFalse(row['passed']);self.assertIn('incomplete SSE',row['error'])

    def test_wrong_visible_answer_is_retained_and_fails_gate(self):
        report,_=self.run_capture(True);self.assertFalse(report['passed'])
        self.assertTrue(any(r['error'] and 'answer' in r['error'] for r in report['requests']))

if __name__=='__main__':unittest.main()
