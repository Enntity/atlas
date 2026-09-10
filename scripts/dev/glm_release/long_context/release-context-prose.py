#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""One near-full C1 prose request; automatic checks never replace human review.

Exact pinned quality Client; <=3 real-chat sizing probes, one tokenizer call,
one final prose generation. Input window[context-608,context-512], output384,
thinking16, temperature0/seed1. <=9 HTTP calls,300s; no launch or tool execution.
Final exit0 means bounded automatic checks only, NOT coherence qualification.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

PROJECT, LEADER, CITY = 'VIOLET-7429', 'Elena Marin', 'Ghent'
FACTS = (f'\nFACT: Project {PROJECT} is led by {LEADER}.\n',
         f'\nFACT: Project {PROJECT} is based in {CITY} and had 31 sealed sample kits.\n',
         f'\nFACT: Exactly 8 sealed sample kits were removed from Project {PROJECT}; no other inventory changes occurred.\n')
QUESTION = (f'\nWrite a concise coherent brief of 3 or 4 complete sentences about Project {PROJECT}. '
            'State its project name, leader, city and exact remaining number of sealed sample kits. '
            'Use only the archive facts. Add no other people, places, dates, purposes or claims. '
            'Return prose only, without headings, bullet lists or JSON.')


def load(path, expected):
    path = Path(path)
    if not path.is_absolute() or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('explicit absolute quality path and SHA required')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('quality source pin mismatch')
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('prose_quality', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prepare(q, client):
    target = client.args.context_limit - 512
    calibrations = []
    def calibrate(repeats):
        q.require(len(calibrations) < 3, 'three chat calibration bound')
        content, used, offsets = 'Read this archive and answer its final request.\n', 0, []
        for position, fact in zip((.05, .50, .95), FACTS):
            before = round(repeats * position)
            content += q.FILLER * (before-used); used = before
            offsets.append(len(content)); content += fact
        content += q.FILLER * (repeats-used) + QUESTION
        body = q.chat_body(client, [{'role': 'user', 'content': content}], [], 384)
        data, receipt = client.request('prose-chat-calibration', '/v1/chat/completions', {**body, 'max_tokens': 1})
        count = q.usage(data, None, 1, client.args.context_limit)['prompt_tokens']
        calibrations.append({'receipt': receipt, 'prompt_tokens': count})
        return count, offsets, body
    base, _, _ = calibrate(0)
    filler_width = len(client.tokens(q.FILLER))
    repeats = (target-base) // filler_width
    q.require(repeats > 0, 'insufficient context for archive')
    count, offsets, body = calibrate(repeats)
    if not target-96 <= count <= target:
        width = (count-base) / repeats
        q.require(width > 0, 'nonpositive actual filler growth')
        count, offsets, body = calibrate(max(0, math.floor((target-base)/width)))
    q.require(target-96 <= count <= target, 'near-full input missed after three probes')
    return body, count, {'calibrations': calibrations, 'fact_character_offsets': offsets,
                         'requested_filler_positions': [.05, .50, .95], 'token_offsets': None,
                         'input_window': [target-96, target], 'expected_remaining_kits': 23}


def check(q, data):
    choice = q.choice(data); message = choice.get('message', {})
    q.require(choice.get('finish_reason') == 'stop', 'prose did not stop normally')
    q.require(not message.get('tool_calls'), 'unexpected tool calls')
    text = message.get('content'); q.require(isinstance(text, str) and text.strip(), 'missing visible prose')
    for fact in (PROJECT, LEADER, CITY, 'sealed sample kits'):
        q.require(fact.casefold() in text.casefold(), 'missing required fact: ' + fact)
    q.require(re.search(r'\b23\b', text), 'missing exact remaining count23')
    returned = json.dumps(data['choices'], ensure_ascii=False)
    foreign = [value for value in (*q.NAMES, *q.RESULTS, *q.CITIES)
               if re.search(r'(?<!\w)' + re.escape(value) + r'(?!\w)', returned, re.IGNORECASE)]
    q.require(not foreign, 'known prior-owner marker present: ' + str(foreign))
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    q.require(3 <= len(sentences) <= 4 and all(s.endswith(('.', '!', '?')) for s in sentences), 'expected3–4 complete sentences')
    normalized = [re.sub(r'\W+', ' ', s.casefold()).strip() for s in sentences]
    q.require(len(set(normalized)) == len(normalized), 'repeated sentence')
    q.require(not re.search(r'(?m)^\s*(?:[#*]|[-]\s|\d+[.)]\s)', text), 'headings/list instead of prose')
    return {'sentence_count': len(sentences), 'known_peer_markers': foreign}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-url', 'model', 'output-dir', 'quality-path', 'quality-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--context-limit', type=int, choices=(4096, 8192, 16384), required=True)
    parser.add_argument('--deadline', type=int, choices=(300,), required=True)
    args = parser.parse_args(); args.timeout = 300
    q = load(args.quality_path, args.quality_sha256)
    path = Path(args.output_dir)
    q.require(path.is_absolute() and path == path.resolve(), 'canonical new output directory required')
    class Client(q.Client):
        def request(self, label, path, body=None):
            q.require(self.serial < 9, 'nine HTTP request bound')
            data, receipt = super().request(label, path, body)
            if path == '/health':
                q.require(data.get('status') == 'ready' and data.get('model') == args.model, 'health/identity failed')
            return data, receipt
    client = Client(args)
    report = {'automatic_checks_passed': False, 'manual_coherence_review_required': True,
              'qualification_passed': None, 'scope': 'one C1 near-full prose sample; no occupancy/speed/general-coherence proof',
              'context_limit': args.context_limit, 'quality_sha256': args.quality_sha256,
              'manual_review': 'Check connected readable sentences, accurate relationships and inventory arithmetic, no invented facts or repetitive/unrelated content; inspect reasoning separately.'}
    try:
        models, _ = client.request('identity', '/v1/models')
        q.require(any(m.get('id') == args.model for m in models.get('data', [])), 'model identity mismatch')
        client.request('initial-health', '/health')
        body, count, preparation = prepare(q, client); report.update(preparation)
        data, receipt = client.request('near-full-prose', '/v1/chat/completions', body)
        report.update(receipt=receipt, usage=q.usage(data, count, 384, args.context_limit), response=data,
                      prompt_sha256=hashlib.sha256(body['messages'][0]['content'].encode()).hexdigest())
        client.request('final-health', '/health'); report.update(check(q, data))
        report['automatic_checks_passed'] = True
    except Exception as error:
        report['error'] = str(error)
    client.save('summary.json', report); print(json.dumps(report, allow_nan=False))
    return 0 if report['automatic_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
