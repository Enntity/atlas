# SPDX-License-Identifier: AGPL-3.0-only
"""HTTP and artifact I/O boundary for bounded behavioral checks."""
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request
from behavior_protocol import decode_sse,unique_object


class BehaviorIORouter:
    def __init__(self,base_url,out,timeout):
        parsed=urllib.parse.urlparse(base_url)
        if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.path not in ('','/'):
            raise ValueError('base URL must be an explicit HTTP server origin, without /v1')
        if type(timeout) not in (int,float) or not 0<timeout<=1200:raise ValueError('explicit bounded timeout required')
        self.url=base_url.rstrip('/')+'/v1/chat/completions'
        self.timeout=timeout;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False)

    def write(self,name,value):
        with (self.out/name).open('x') as handle:json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n')

    def sources(self):
        repo=Path(__file__).resolve().parents[2]
        paths=[Path(__file__),Path(__file__).with_name('collect_behavior.py'),
               Path(__file__).with_name('behavior_protocol.py'),
               repo/'jinja-templates/openai/glm5_next.jinja',
               repo/'kernels/gb10/glm-5.3-flash-exl3/MODEL.toml',
               repo/'crates/spark-model/src/model/trait_impl/decode_a.rs',
               repo/'crates/spark-server/src/main_modules/serve_phases/preflight.rs']
        return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    def request(self,name,payload):
        if not re.fullmatch(r'[a-z][a-z0-9-]+',name):raise ValueError('safe request identity required')
        receipt={'id':name,'payload':payload,'passed':False,'error':None}
        self.write(name+'.request.json',payload)
        req=urllib.request.Request(self.url,data=json.dumps(payload).encode(),
                                   headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=self.timeout) as response:
                receipt['http_status']=response.status
                receipt['content_type']=response.headers.get_content_type()
                raw=response.read()
        except urllib.error.HTTPError as error:
            receipt['http_status']=error.code;raw=error.read();receipt['error']=str(error)
        except Exception as error:
            receipt['error']=type(error).__name__+': '+str(error);raw=b''
        path=self.out/(name+'.response.raw');path.write_bytes(raw)
        receipt['raw_file']=path.name;receipt['raw_sha256']=hashlib.sha256(raw).hexdigest()
        if receipt.get('http_status')!=200:
            receipt['error']=receipt['error'] or 'HTTP success required';return receipt,None
        try:
            if payload.get('stream'):
                if receipt['content_type']!='text/event-stream':raise ValueError('SSE media type required')
                body,events=decode_sse(raw.decode());self.write(name+'.events.json',events)
            else:
                if receipt['content_type']!='application/json':raise ValueError('JSON media type required')
                body=json.loads(raw.decode(),object_pairs_hook=unique_object)
            self.write(name+'.response.json',body)
            receipt['response']=body
            return receipt,body
        except Exception as error:
            receipt['error']=type(error).__name__+': '+str(error);return receipt,None
