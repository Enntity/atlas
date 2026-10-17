# SPDX-License-Identifier: AGPL-3.0-only
import copy
import pytest
from l1_shutdown import validate_shutdown


def evidence():
    cid='a'*64
    before={'Id':cid,'Name':'/owned','State':{'Running':True,'OOMKilled':False,'ExitCode':0},'captured_epoch':10}
    after={'Id':cid,'Name':'/owned','State':{'Running':False,'OOMKilled':False,'ExitCode':137},'captured_epoch':50}
    events=[{'Actor':{'ID':cid,'Attributes':{'signal':'15'}},'Action':'kill','timeNano':11000000000},
            {'Actor':{'ID':cid,'Attributes':{'signal':'9'}},'Action':'kill','timeNano':41000000000},
            {'Actor':{'ID':cid,'Attributes':{'exitCode':'137'}},'Action':'die','timeNano':42000000000},
            {'Actor':{'ID':cid,'Attributes':{}},'Action':'stop','timeNano':43000000000}]
    for event in events:event.update(Type='container',scope='local')
    return {'policy':{'stop_timeout_seconds':30,'worker_exit_policy':'controlled-headless137'},
            'engine':'reference','rank':'1','cid':cid,'name':'owned','before':before,'after':after,
            'stop_stdout':cid,'wait_stdout':'137','events':events,'collection_complete':True,'attested':True}


def test_controlled_timeout_is_explicit_and_not_clean_exit():
    result=validate_shutdown(**evidence())
    assert result=='controlled-timeout-stop137'


@pytest.mark.parametrize('case',['candidate','leader','oom','already-dead','unowned','incomplete','unattested','missing-kill','wrong-exit','wrong-wait','foreign-event','outside-bound','early-kill'])
def test_137_requires_all_owned_post_collection_state_and_event_guards(case):
    value=evidence()
    if case=='candidate':value['engine']='candidate'
    elif case=='leader':value['rank']='0'
    elif case=='oom':value['after']['State']['OOMKilled']=True
    elif case=='already-dead':value['before']['State']['Running']=False
    elif case=='unowned':value['before']['Id']='b'*64
    elif case=='incomplete':value['collection_complete']=False
    elif case=='unattested':value['attested']=False
    elif case=='missing-kill':del value['events'][1]
    elif case=='wrong-exit':value['after']['State']['ExitCode']=1
    elif case=='wrong-wait':value['wait_stdout']='0'
    elif case=='foreign-event':value['events'][0]['Actor']['ID']='b'*64
    elif case=='outside-bound':value['events'][1]['timeNano']=9000000000
    elif case=='early-kill':value['events'][1]['timeNano']=20000000000
    with pytest.raises(ValueError):validate_shutdown(**value)


def test_exit_zero_remains_clean_and_other_nonzero_is_refused():
    value=evidence();value['engine']='candidate';value['policy']['worker_exit_policy']='forbid';value['after']['State']['ExitCode']=0;value['wait_stdout']='0';value['events']=[]
    assert validate_shutdown(**value)=='clean-stop0'
    value['after']['State']['ExitCode']=143;value['wait_stdout']='143'
    with pytest.raises(ValueError):validate_shutdown(**value)


@pytest.mark.parametrize('case',['noncontainer','missing-type','wrong-scope','nan-bound','boolean-nano','fractional-nano'])
def test_event_kind_and_time_types_are_explicit(case):
    value=evidence()
    if case=='noncontainer':value['events'][0]['Type']='network'
    elif case=='missing-type':value['events'][0].pop('Type',None)
    elif case=='wrong-scope':value['events'][0]['scope']='swarm'
    elif case=='nan-bound':value['before']['captured_epoch']=float('nan')
    elif case=='boolean-nano':value['events'][0]['timeNano']=True
    else:value['events'][0]['timeNano']=11000000000.0
    with pytest.raises(ValueError):validate_shutdown(**value)
