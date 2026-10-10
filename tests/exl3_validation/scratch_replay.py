# SPDX-License-Identifier: AGPL-3.0-only
"""Source-bound caller of the actual Rust two-layer scratch replay executable."""
import argparse,json,subprocess
from pathlib import Path
from native_ptx import sha,validate_binding

REQUIRED={name:{'entries':(),'sources':()} for name in
          ('exl3','exl3_moe','exl3_moe_precision','exl3_moe_mma','exl3_shared','residual_add')}
REQUIRED['exl3_moe_mma']['argument_counts']={'exl3_moe_mul1_matmul_precision_mma':19}


def validate_report(report,plan_sha,binary_sha,ptx_hashes,blas_hashes=None):
    if (report.get('schema')!='atlas-native-two-layer-scratch-replay-v1' or report.get('passed') is not True
            or report.get('plan_sha256')!=plan_sha or report.get('executable_sha256')!=binary_sha
            or report.get('guards_passed') is not True or report.get('input_unchanged') is not True):
        raise ValueError('actual production replay report/plan/binary/guards mismatch')
    if blas_hashes is not None and report.get('loaded_blas_sha256')!=blas_hashes:
        raise ValueError('actual mapped BLAS differs from immutable image library pair')
    if not report.get('output_sha256'):
        raise ValueError('actual replay outputs required')
    if any(report.get('ptx_sha256',{}).get(name)!=digest for name,digest in ptx_hashes.items()):
        raise ValueError('actual loaded PTX differs from image-bound replay modules')


class ReplayIORouter:
    def __init__(self,binary,plan,out,ptx_dir,binding,common):
        self.binary=Path(binary).resolve(strict=True);self.plan=Path(plan).resolve(strict=True)
        self.out=Path(out);self.ptx_dir=Path(ptx_dir);self.binding_path=Path(binding)
        self.binding=validate_binding(ptx_dir,binding,common,required=REQUIRED)
        self.blas_hashes=self.binding.get('runtime_library_sha256')
        if not isinstance(self.blas_hashes,dict) or len(self.blas_hashes)!=2:
            raise ValueError('explicit immutable image classic/Lt pair fingerprints required')
        self.binary_sha=sha(self.binary);self.plan_sha=sha(self.plan)
        if self.binary_sha!=self.binding['embedded_binary_sha256']:
            raise ValueError('PTX binding must attest the replay executable itself')
        if self.out.exists():raise ValueError('outer replay record directory must be new')
        image=self.binary.read_bytes()
        self.ptx_hashes={}
        for name in REQUIRED:
            item=self.binding['modules'][name];data=(self.ptx_dir/item['file']).read_bytes()
            if data not in image:raise ValueError('actual PTX bytes absent from replay executable: '+name)
            self.ptx_hashes[name]=item['sha256']
    def run(self):
        self.out.mkdir(parents=True)
        (self.out/'plan.json').write_bytes(self.plan.read_bytes())
        request={'binary':str(self.binary),'binary_sha256':self.binary_sha,'plan_sha256':self.plan_sha,
                 'ptx_binding_sha256':sha(self.binding_path),'image':self.binding['image'],
                 'build_source_sha256':self.binding['build_source_sha256'],'ptx_sha256':self.ptx_hashes,
                 'scope':'actual production Rust loader/rank-local forward; shared-vs-isolated scratch correctness only'}
        (self.out/'registration.json').write_text(json.dumps(request,indent=2))
        process=subprocess.run([str(self.binary),'--plan',str(self.plan),'--out',str(self.out/'raw')],
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
        (self.out/'stdout.log').write_bytes(process.stdout);(self.out/'stderr.log').write_bytes(process.stderr)
        receipt={**request,'exit_code':process.returncode,'passed':False}
        try:
            if process.returncode!=0:raise ValueError('actual Rust replay failed; raw evidence retained')
            report=json.loads((self.out/'raw/report.json').read_text())
            validate_report(report,self.plan_sha,self.binary_sha,self.ptx_hashes,self.blas_hashes)
            for case,digest in report['output_sha256'].items():
                a=self.out/'raw'/('isolated-'+case+'.bf16');b=self.out/'raw'/('shared-'+case+'.bf16')
                if sha(a)!=digest or sha(b)!=digest or a.read_bytes()!=b.read_bytes():
                    raise ValueError('shared/isolated typed raw bytes differ: '+case)
            receipt['report_sha256']=sha(self.out/'raw/report.json');receipt['passed']=True
        except Exception as error:
            receipt['failure']=str(error)
        (self.out/'receipt.json').write_text(json.dumps(receipt,indent=2))
        return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','plan','out','native-ptx-dir','ptx-binding','common-dir'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    receipt=ReplayIORouter(args.binary,args.plan,args.out,args.native_ptx_dir,args.ptx_binding,args.common_dir).run()
    print(json.dumps(receipt));raise SystemExit(0 if receipt['passed'] else 1)

if __name__=='__main__':main()
