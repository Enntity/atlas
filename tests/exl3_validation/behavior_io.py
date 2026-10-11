# SPDX-License-Identifier: AGPL-3.0-only
"""HTTP and artifact I/O boundary for bounded behavioral checks."""
import hashlib
import http.client
import json
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request
from behavior_anthropic_protocol import decode_messages_sse
from behavior_protocol import decode_completions_sse,decode_sse,sse_shape,unique_object

DECODERS={'chat':decode_sse,'completions':decode_completions_sse,'messages':decode_messages_sse}
PROBES={'health':'/health','metrics':'/metrics'}


class BehaviorIORouter:
    def __init__(self,base_url,out,timeout):
        parsed=urllib.parse.urlparse(base_url)
        if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.path not in ('','/'):
            raise ValueError('base URL must be an explicit HTTP server origin, without /v1')
        if type(timeout) not in (int,float) or not 0<timeout<=1200:raise ValueError('explicit bounded timeout required')
        self.base=base_url.rstrip('/')
        self.paths={'chat':'/v1/chat/completions','completions':'/v1/completions','messages':'/v1/messages'}
        self.urls={key:self.base+path for key,path in self.paths.items()}
        self.parsed=parsed
        self.timeout=timeout;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False)

    def write(self,name,value):
        with (self.out/name).open('x') as handle:json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n')

    def sources(self):
        repo=Path(__file__).resolve().parents[2]
        paths=[Path(__file__),Path(__file__).with_name('collect_behavior.py'),
               Path(__file__).with_name('behavior_protocol.py'),
               Path(__file__).with_name('behavior_surfaces.py'),
               Path(__file__).with_name('behavior_schema.py'),
               Path(__file__).with_name('behavior_anthropic_protocol.py'),
               Path(__file__).with_name('behavior_messages.py'),
               Path(__file__).with_name('behavior_tools.py'),
               Path(__file__).with_name('behavior_robustness.py'),
               repo/'jinja-templates/openai/glm5_next.jinja',
               repo/'kernels/gb10/glm-5.3-flash-exl3/MODEL.toml',
               repo/'crates/spark-model/src/model/trait_impl/decode_a.rs',
               repo/'crates/spark-server/src/main_modules/serve_phases/preflight.rs']
        return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    def request(self,name,payload,surface='chat'):
        if not re.fullmatch(r'[a-z][a-z0-9-]+',name):raise ValueError('safe request identity required')
        if surface not in self.urls:raise ValueError('unknown API surface: '+surface)
        receipt={'id':name,'payload':payload,'passed':False,'error':None}
        self.write(name+'.request.json',payload)
        req=urllib.request.Request(self.urls[surface],data=json.dumps(payload).encode(),
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
            receipt['error_body']=raw[:4096].decode(errors='replace')
            receipt['error']=receipt['error'] or 'HTTP success required';return receipt,None
        try:
            if payload.get('stream'):
                if receipt['content_type']!='text/event-stream':raise ValueError('SSE media type required')
                body,events=DECODERS[surface](raw.decode());self.write(name+'.events.json',events)
                receipt['sse_shape']=([event['type'] for event in events] if surface=='messages'
                                      else sse_shape(events))
            else:
                if receipt['content_type']!='application/json':raise ValueError('JSON media type required')
                body=json.loads(raw.decode(),object_pairs_hook=unique_object)
            self.write(name+'.response.json',body)
            receipt['response']=body
            return receipt,body
        except Exception as error:
            receipt['error']=type(error).__name__+': '+str(error);return receipt,None

    def probe(self,name,probe):
        """GET a fixed server probe (/health or /metrics); raw bytes retained."""
        if not re.fullmatch(r'[a-z][a-z0-9-]+',name) or probe not in PROBES:raise ValueError('safe probe identity required')
        receipt={'id':name,'probe':PROBES[probe],'passed':False,'error':None}
        try:
            with urllib.request.urlopen(self.base+PROBES[probe],timeout=self.timeout) as response:
                receipt['http_status']=response.status;receipt['content_type']=response.headers.get_content_type()
                raw=response.read()
        except urllib.error.HTTPError as error:
            receipt['http_status']=error.code;raw=error.read();receipt['error']=str(error)
        except Exception as error:
            receipt['error']=type(error).__name__+': '+str(error);raw=b''
        path=self.out/(name+'.response.raw');path.write_bytes(raw)
        receipt['raw_file']=path.name;receipt['raw_sha256']=hashlib.sha256(raw).hexdigest()
        return receipt,raw.decode(errors='replace')

    def interrupt(self,name,payload,surface,mode,limit):
        """Abandon a request client-side and record what was received.

        mode 'cancel': stream, read `limit` SSE data frames, then close the
        socket mid-stream. mode 'timeout': send, then give up after `limit`
        seconds without the complete response (socket read timeout)."""
        if not re.fullmatch(r'[a-z][a-z0-9-]+',name) or surface not in self.paths:raise ValueError('safe request identity required')
        if mode=='cancel' and not (payload.get('stream') and type(limit) is int and limit>0):
            raise ValueError('cancel requires a stream and a positive frame count')
        if mode=='timeout' and not (type(limit) in (int,float) and 0<limit<self.timeout):
            raise ValueError('client timeout must be shorter than the router timeout')
        if mode not in ('cancel','timeout'):raise ValueError('unknown interrupt mode')
        receipt={'id':name,'payload':payload,'interrupt':mode,'limit':limit,'passed':False,'error':None}
        self.write(name+'.request.json',payload)
        port=self.parsed.port or (443 if self.parsed.scheme=='https' else 80)
        factory=http.client.HTTPSConnection if self.parsed.scheme=='https' else http.client.HTTPConnection
        conn=factory(self.parsed.hostname,port,timeout=limit if mode=='timeout' else self.timeout)
        raw=b'';frames=0
        try:
            conn.request('POST',self.paths[surface],body=json.dumps(payload).encode(),
                         headers={'Content-Type':'application/json'})
            response=conn.getresponse();receipt['http_status']=response.status
            receipt['content_type']=response.getheader('Content-Type','').split(';')[0].strip()
            if mode=='cancel':
                while frames<limit:
                    line=response.fp.readline()
                    if not line:break
                    raw+=line
                    if line.startswith(b'data:'):frames+=1
                receipt['frames_before_cancel']=frames
                receipt['completed_before_cancel']=frames<limit
            else:
                raw=response.read();receipt['completed_before_timeout']=True
        except (TimeoutError,OSError) as error:
            receipt['client_error']=type(error).__name__+': '+str(error)
            if mode=='timeout':receipt['completed_before_timeout']=False
        finally:
            conn.close()
        path=self.out/(name+'.response.raw');path.write_bytes(raw)
        receipt['raw_file']=path.name;receipt['raw_sha256']=hashlib.sha256(raw).hexdigest()
        return receipt,raw
