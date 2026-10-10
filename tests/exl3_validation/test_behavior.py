# SPDX-License-Identifier: AGPL-3.0-only
import json
import unittest
from behavior_protocol import validate_body, decode_sse, validate_tool


def body(content='Tokyo',finish='stop'):
    return {'id':'chat-test','model':'test','choices':[{'index':0,'finish_reason':finish,
            'message':{'role':'assistant','content':content}}],
            'usage':{'prompt_tokens':10,'completion_tokens':3,'total_tokens':13,
                     'completion_tokens_details':{'reasoning_tokens':0}}}


def stream(value):
    choice=value['choices'][0]
    events=[{'id':value['id'],'model':value['model'],'choices':[{'index':0,
             'delta':{'role':'assistant','content':choice['message']['content']},'finish_reason':None}]},
            {'id':value['id'],'model':value['model'],'choices':[{'index':0,'delta':{},
             'finish_reason':choice['finish_reason']}]},
            {'id':value['id'],'model':value['model'],'choices':[],'usage':value['usage']}]
    return ''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n'


class BehaviorBoundaryTests(unittest.TestCase):
    def test_visible_and_reasoning_envelope_are_distinct_and_budgeted(self):
        b=body();self.assertEqual(validate_body(b,'stop',128,4096)['content'],'Tokyo')
        b['choices'][0]['message']['reasoning_content']='Compute carefully'
        b['usage']['completion_tokens_details']['reasoning_tokens']=2
        self.assertEqual(validate_body(b,'stop',128,4096,32)['reasoning'],'Compute carefully')
        for mutation in ('empty','length','nan','context','usage','reasoning'):
            b=body()
            if mutation=='empty':b['choices'][0]['message']['content']=''
            elif mutation=='length':b['choices'][0]['finish_reason']='length'
            elif mutation=='nan':b['usage']['diagnostic']=float('nan')
            elif mutation=='context':b['usage']['prompt_tokens']=4000
            elif mutation=='usage':b['usage']['total_tokens']=12
            else:
                b['choices'][0]['message']['reasoning_content']='hidden'
                b['usage']['completion_tokens_details']['reasoning_tokens']=0
            with self.assertRaises(ValueError):validate_body(b,'stop',128,4096)

    def test_sse_reconstruction_and_incomplete_identity_failures(self):
        b=body();parsed,events=decode_sse(stream(b));self.assertEqual(parsed,b);self.assertEqual(len(events),3)
        for raw in [stream(b).replace('data: [DONE]',''),stream(b)+'data: {}\n\n',
                    stream(b).replace('chat-test','other',1),
                    stream(b).replace('"finish_reason": "stop"','"finish_reason": "length"'),
                    'data: {"error":"failed"}\n\ndata: [DONE]\n\n']:
            with self.assertRaises(ValueError):validate_body(decode_sse(raw)[0],'stop',128,4096)

    def test_tool_call_owns_exact_name_arguments_and_id(self):
        b=body(None,'tool_calls');message=b['choices'][0]['message']
        message['tool_calls']=[{'id':'call-1','type':'function','function':{
            'name':'lookup_inventory','arguments':'{"sku":"BLUE-17","warehouse":"north"}'}}]
        expected={'sku':'BLUE-17','warehouse':'north'}
        self.assertEqual(validate_tool(b,'lookup_inventory',expected),'call-1')
        for bad in ['{}','{"sku":"BLUE-17","sku":"BLUE-17","warehouse":"north"}',
                    '{"sku":"OTHER","warehouse":"north"}']:
            message['tool_calls'][0]['function']['arguments']=bad
            with self.assertRaises(ValueError):validate_tool(b,'lookup_inventory',expected)

if __name__=='__main__':unittest.main()
