# SPDX-License-Identifier: AGPL-3.0-only
import pytest
from l1_reference_attest import select_worker


def process(pid, command, parent=1, cgroup='docker-owned'):
    return {'pid':pid,'ppid':parent,'cmdline':command,'cgroup':cgroup,'maps':
            '1-2 r-xp 0 00:00 1 /lib/libcuda.so.1\n'
            '2-3 r-xp 0 00:00 2 /lib/libcublas.so.13\n'
            '3-4 r-xp 0 00:00 3 /lib/libcublasLt.so.13\n'}


def test_select_actual_worker_not_shell_or_other_container():
    records=[process(1,'bash /start.sh',0),process(2,'ray::RayWorkerWrapper'),
             process(3,'ray::RayWorkerWrapper',cgroup='other')]
    selected=select_worker(records,1,'docker-owned',r'ray::RayWorkerWrapper')
    assert selected['pid']==2


def test_reject_ambiguous_or_unmapped_workers():
    with pytest.raises(ValueError,match='exactly one'):
        select_worker([process(1,'bash',0),process(2,'worker'),process(3,'worker')],1,'docker-owned','worker')
    p=process(2,'worker');p['maps']=p['maps'].replace('libcublasLt','unexpected')
    with pytest.raises(ValueError,match='exactly one'):
        select_worker([process(1,'bash',0),p],1,'docker-owned','worker')


def test_reject_pid_not_descendant_even_with_owned_cgroup():
    with pytest.raises(ValueError,match='exactly one'):
        select_worker([process(1,'bash',0),process(2,'worker',parent=99)],1,'docker-owned','worker')


def test_attest_actual_proc_fixture_binds_container_worker_and_libraries(tmp_path):
    import hashlib,json
    from l1_reference_attest import attest
    cid='1'*64;classic=b'classic';lt=b'Lt'
    pins=tmp_path/'pins';pins.write_text(hashlib.sha256(classic).hexdigest()+'  /lib/libcublas.so.13\n'+
                                        hashlib.sha256(lt).hexdigest()+'  /lib/libcublasLt.so.13\n')
    class IO:
        root=tmp_path
        def command(self,args):
            if args[1]=='inspect':return json.dumps([{'Id':cid,'State':{'Running':True,'Pid':1}}]).encode()
            return b'PID COMMAND\n1 bash\n2 worker\n'
        def process(self,pid):return process(pid,'worker' if pid==2 else 'bash',parent=1 if pid==2 else 0)
        def library_hash(self,pid,path):
            assert pid==2
            return hashlib.sha256(classic if path.endswith('libcublas.so.13') else lt).hexdigest()
        def read(self,path):return Path(path).read_bytes()
    from pathlib import Path
    report=attest(cid,pins,'worker',IO())
    assert report['container']==cid and report['pid']==2 and report['owned_processes']==[1,2]
    assert report['cmdline']=='worker' and report['ok']
    assert len(report['libraries'])==2


def test_deleted_or_wrong_provider_bytes_fail_after_worker_selection(tmp_path):
    import hashlib,json
    from l1_reference_attest import attest
    cid='1'*64; pins=tmp_path/'pins';pins.write_text('a'*64+'  /lib/libcublas.so.13\n'+'b'*64+'  /lib/libcublasLt.so.13\n')
    class IO:
        root=tmp_path
        def command(self,args):
            if args[1]=='inspect':return json.dumps([{'Id':cid,'State':{'Running':True,'Pid':1}}]).encode()
            return b'PID COMMAND\n1 bash\n2 worker\n'
        def process(self,pid):return process(pid,'worker' if pid==2 else 'bash',parent=1 if pid==2 else 0)
        def library_hash(self,pid,path):return 'c'*64
    with pytest.raises(ValueError,match='providers differ'):
        attest(cid,pins,'worker',IO())
