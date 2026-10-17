# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit L1 shutdown classification; forced timeout stop is never clean exit."""
import math

from c2_remote_adapter import require


def validate_policy(policy,engine):
    require(set(policy)=={'stop_timeout_seconds','worker_exit_policy'},'explicit shutdown policy required')
    require(type(policy['stop_timeout_seconds']) is int and policy['stop_timeout_seconds']>0,
            'positive explicit stop timeout required')
    require(policy['worker_exit_policy'] in ('forbid','controlled-headless137')
            and (engine=='reference' or policy['worker_exit_policy']=='forbid'),
            'worker timeout policy is reference-only')


def validate_shutdown(*,policy,engine,rank,cid,name,before,after,stop_stdout,wait_stdout,
                      events,collection_complete,attested):
    validate_policy(policy,engine)
    require(all(record['Id']==cid and record['Name']=='/'+name for record in (before,after)),
            'shutdown container ownership differs')
    require(stop_stdout.strip()==cid,'owned explicit stop did not succeed')
    require(before['State']['OOMKilled'] is False and after['State']['OOMKilled'] is False,
            'OOM cannot qualify as controlled shutdown')
    require(after['State']['Running'] is False and type(after['State']['ExitCode']) is int
            and wait_stdout.strip()==str(after['State']['ExitCode']),'post-state/wait exit differs')
    code=after['State']['ExitCode']
    if code==0:
        return 'clean-stop0'
    require(code==137 and engine=='reference' and rank=='1'
            and policy['worker_exit_policy']=='controlled-headless137','nonzero shutdown exit refused')
    require(before['State']['Running'] is True and collection_complete is True and attested is True,
            'worker137 needs running, attested, completed source-bound collection')
    require(all(type(record['captured_epoch']) in (int,float) and math.isfinite(record['captured_epoch'])
                for record in (before,after)), 'finite explicit shutdown interval required')
    require(before['captured_epoch']<after['captured_epoch'],'bounded shutdown time interval required')
    require(bool(events),'controlled stop event evidence missing')
    actions=[];times=[]
    for event in events:
        require(event.get('Type')=='container' and event.get('scope')=='local', 'local container events required')
        require(type(event.get('timeNano')) is int and 0<event['timeNano']<=2**63-1,
                'integer positive Docker nanosecond timestamp required')
        require(event['Actor']['ID']==cid,'foreign container shutdown event')
        timestamp=event['timeNano']/1e9
        require(before['captured_epoch']<=timestamp<=after['captured_epoch'],'shutdown event outside bound')
        action=event['Action'];attributes=event['Actor']['Attributes']
        require(action not in ('oom','start','restart','destroy'),'invalid shutdown event')
        if action in ('kill','die','stop'):
            actions.append((action,attributes.get('signal') if action=='kill' else attributes.get('exitCode') if action=='die' else None))
            times.append(event['timeNano'])
    require(times==sorted(times),'shutdown events out of order')
    require(len(actions)==4 and actions[:2]==[('kill','15'),('kill','9')]
            and set(actions[2:])=={('die','137'),('stop',None)},
            'exact SIGTERM15, SIGKILL9, stop/die137 evidence required')
    require(times[1]-times[0]>=policy['stop_timeout_seconds']*1_000_000_000,
            'SIGKILL occurred before registered stop timeout')
    return 'controlled-timeout-stop137'
