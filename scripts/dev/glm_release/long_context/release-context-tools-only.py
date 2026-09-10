#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""One C1 near-full auto-tool/result probe, not full-context qualification.

Uses the exact SHA-pinned quality helper without changing requests, caps or
validators. At most16 HTTP calls in300s; ordinary watchdog launch is external.
Literal tool results are fixtures, never execution of an external tool.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


def load(path, expected):
    path = Path(path)
    if not path.is_absolute() or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('explicit absolute quality path and SHA required')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('quality source pin mismatch')
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('tools_quality', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-url', 'model', 'output-dir', 'quality-path', 'quality-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--context-limit', type=int, choices=(4096, 8192, 16384), required=True)
    args = parser.parse_args()
    args.deadline = args.timeout = 300
    q = load(args.quality_path, args.quality_sha256)
    path = Path(args.output_dir)
    q.require(path.is_absolute() and path == path.resolve(), 'canonical new output directory required')

    class Client(q.Client):
        def request(self, label, path, body=None):
            # C1 wave has one worker; preparation/health run only after it joins.
            q.require(self.serial < 16, 'sixteen HTTP request bound')
            self.remaining()
            data, receipt = super().request(label, path, body)
            if path == '/health':
                q.require(data.get('status') == 'ready' and data.get('model') == args.model,
                          'health/identity failed')
            return data, receipt

    client = Client(args)
    report = {
        'passed': False, 'full_context_qualification_passed': None,
        'scope': 'one C1 near-full auto-tool and actual-ID result roundtrip only; no needle, prose, concurrency, occupancy or speed qualification',
        'context_limit': args.context_limit, 'concurrency': 1,
        'quality_path': args.quality_path, 'quality_sha256': args.quality_sha256,
        'deadline_seconds': 300, 'http_limit': 16,
    }
    try:
        models, _ = client.request('identity', '/v1/models')
        q.require(any(row.get('id') == args.model for row in models.get('data', [])),
                  'model identity mismatch')
        client.request('initial-health', '/health')
        prepared = q.prepare_tool(client, 0)
        report['auto_tool_calibrations'] = prepared['calibrations']
        rows = q.wave(client, [prepared])
        # Persist the actual assistant/call ID before any followup preparation;
        # a later calibration/generation failure cannot erase this evidence.
        client.save('original-auto-tool-row.json', rows[0])
        report['auto_tool'] = rows[0]
        followup = q.prepare_tool(client, 0, rows[0])
        report['tool_result_calibrations'] = followup['calibrations']
        report['tool_result'] = q.wave(client, [followup])[0]
        client.request('final-health', '/health')
        report['passed'] = True
    except Exception as error:
        report['error'] = str(error)
        try:
            _, report['failure_health_receipt'] = client.request('failure-health', '/health')
        except Exception as health_error:
            report['failure_health_error'] = str(health_error)
    finally:
        report['http_requests'] = client.serial
        client.save('summary.json', report)
        print(json.dumps(report, allow_nan=False), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
