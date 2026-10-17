#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Q3 live API behavior suite: one JSON receipt per case plus a summary.

Runs against any OpenAI/Anthropic-compatible origin (Atlas, vLLM, ...):
/v1/chat/completions, /v1/completions and /v1/messages, blocking and SSE.
Every assertion is recorded with its evidence; a failing assertion never
stops the case and a failing case never stops the suite. Assertions that
need an optional server capability the endpoint does not expose (token IDs,
an EOS id list, the Atlas `atlas_requests_active` gauge) are recorded as
`skipped` with the reason, never as passed.

    python3 tests/exl3_validation/q3_behavior_suite.py \
        --endpoint http://HOST:PORT --model MODEL --out DIR [--only CASE ...]

DIR must not exist. It receives `<case>.json` receipts, `summary.json`, and
`raw/` with every request payload, raw response bytes and decoded SSE events
(behavior_io.BehaviorIORouter). Exit status 0 only when every case passes.

Wire contracts gated (vLLM reference, 0.1.dev20051+g487ecf187):
- include_usage: the finish chunk, then one usage-only chunk with
  `choices: []` as the last data event
  (entrypoints/openai/chat_completion/serving.py:766).
- stop_token_ids: the stop token is appended and then stops the request, so
  it counts in completion_tokens (v1/core/sched/utils.py:108 check_stop).
- include_stop_str_in_output: the output ends WITH the matched stop string
  (v1/engine/detokenizer.py:443 check_stop_strings include_in_output).
- empty stop string: rejected with HTTP 400 (sampling_params.py:763).
- Responses API seed: accepted and passed to sampling
  (entrypoints/openai/responses/protocol.py:283, :445).
- Anthropic stop_sequence: gated per the Anthropic Messages contract (echo
  the matched string with stop_reason 'stop_sequence'), which vLLM's own
  translator does not do on the stream (entrypoints/anthropic/serving.py:811).
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

from behavior_io import BehaviorIORouter
from behavior_anthropic_protocol import parse_named_sse
from behavior_protocol import LEAK_MARKERS, LEAK_SPECIAL, sse_shape, usage_follows_finish
from behavior_schema import parse_strict_json, schema_errors
from behavior_surfaces import (CALC_ANSWER, CALC_PROMPT, CODEWORD, JOKE_PROMPT, RECALL_PROMPT,
                               RECALL_QUESTION)
from behavior_surfaces import ANSWER as TOKYO
from behavior_surfaces import LONG_PROMPT as COUNT_PROMPT
from behavior_surfaces import TOKYO_PROMPT
from behavior_tools import (CITY_PROMPT, CITY_SCHEMA, CITY_VALUE, FOLLOWUP_ANSWER, FOLLOWUP_PROMPT,
                            INVENTORY_ARGUMENTS, INVENTORY_PROMPT, OBJECT_PROMPT, PARALLEL_ARGUMENTS,
                            PARALLEL_PROMPT, SCHEMAS, TOOLS, WEATHER_RESULTS)

MAX_TOKENS = 256
LENGTH_TOKENS = 8
SEED = 1234
SAMPLE_TEMPERATURE = 1.0
# ReAct-style text that provably continues past the stop string, so a server
# that only trims a trailing match (instead of stopping at it) leaves the tail.
STOP_MARKER = 'Observation:'
REACT_HEAD = 'Thought: check the weather'
REACT_TAIL = ('sunny', 'Final')
REACT_PROMPT = ('Copy the following three lines exactly, with no other text:\n'
                'Thought: check the weather\nObservation: sunny and 21 C\nFinal: it is sunny')
# Legacy /v1/completions applies no chat template; a plain completion prefix
# keeps the case endpoint-agnostic. '.' ends the answer sentence.
CMPL_PROMPT = 'Q: What is the capital of France?\nA: The capital of France is'
CMPL_ANSWER = 'Paris'
CMPL_STOP = '.'
CMPL_COUNT_PROMPT = 'Counting: 1, 2, 3, 4, 5,'
LONG_PROMPT = ('Write a detailed 600-word explanation of how TCP congestion control works, '
               'aimed at a systems engineer.')
LONG_TOKENS = 512
ACTIVE_GAUGE = re.compile(r'^atlas_requests_active\s+(\S+)\s*$', re.M)
EVIDENCE_CHARS = 240


def clip(value, limit=EVIDENCE_CHARS):
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + '...(' + str(len(value)) + ' chars)'
    return value


# ---------------------------------------------------------------------------
# Pure accessors and checks (unit-tested directly)
# ---------------------------------------------------------------------------

def chat_message(body):
    try:
        message = body['choices'][0]['message']
    except (KeyError, IndexError, TypeError):
        return {}
    return message if isinstance(message, dict) else {}


def chat_finish(body):
    try:
        return body['choices'][0].get('finish_reason')
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


def chat_content(body):
    content = chat_message(body).get('content')
    return content if isinstance(content, str) else ''


def chat_reasoning(body):
    message = chat_message(body)
    value = message.get('reasoning_content', message.get('reasoning'))
    return value if isinstance(value, str) else ''


def completion_text(body):
    try:
        text = body['choices'][0].get('text')
    except (KeyError, IndexError, TypeError, AttributeError):
        return ''
    return text if isinstance(text, str) else ''


def message_text(body, kind='text'):
    blocks = body.get('content') if isinstance(body, dict) else None
    if not isinstance(blocks, list):
        return ''
    return ''.join(block.get(kind, '') for block in blocks
                   if isinstance(block, dict) and block.get('type') == kind
                   and isinstance(block.get(kind), str))


def leaks(text):
    """Special-token / envelope markers present in a visible channel."""
    if not isinstance(text, str):
        return ['non-text channel']
    found = [marker for marker in LEAK_MARKERS if marker in text]
    found += LEAK_SPECIAL.findall(text)
    return found


def usage_problems(usage, max_tokens, keys=('prompt_tokens', 'completion_tokens', 'total_tokens')):
    """OpenAI usage: positive integers, prompt + completion == total, and the
    completion within max_tokens. Returns a list of violations."""
    if not isinstance(usage, dict):
        return ['usage object missing']
    problems = [key + ' must be a positive integer' for key in keys
                if type(usage.get(key)) is not int or usage[key] <= 0]
    if problems:
        return problems
    prompt, completion, total = (usage[key] for key in keys)
    if prompt + completion != total:
        problems.append('prompt_tokens + completion_tokens != total_tokens')
    if completion > max_tokens:
        problems.append('completion_tokens exceeds max_tokens')
    return problems


def message_usage_problems(usage, max_tokens):
    if not isinstance(usage, dict):
        return ['usage object missing']
    problems = [key + ' must be a positive integer' for key in ('input_tokens', 'output_tokens')
                if type(usage.get(key)) is not int or usage[key] <= 0]
    if not problems and usage['output_tokens'] > max_tokens:
        problems.append('output_tokens exceeds max_tokens')
    return problems


def usage_chunk_last(events):
    """include_usage framing: exactly one finish chunk, then exactly one
    usage-only chunk with `choices: []` as the final data event."""
    shape = sse_shape(events or [])
    last = (events or [None])[-1]
    ok = (usage_follows_finish(shape) and isinstance(last, dict) and last.get('choices') == [])
    return ok, {'tail_shape': shape[-3:], 'events': len(shape)}


def stream_token_ids(events):
    """Concatenate per-chunk `choices[0].token_ids` (vLLM return_token_ids
    extension). Returns (ids, content_positions) or (None, None) when the
    stream carries no token IDs."""
    ids, content_positions, seen = [], [], False
    for event in events or []:
        for choice in event.get('choices') or []:
            chunk = choice.get('token_ids')
            if chunk is None:
                continue
            seen = True
            if not isinstance(chunk, list) or not all(type(i) is int for i in chunk):
                return None, None
            is_content = bool((choice.get('delta') or {}).get('content'))
            for token in chunk:
                if is_content:
                    content_positions.append(len(ids))
                ids.append(token)
    return (ids, content_positions) if seen else (None, None)


def pick_stop_token(ids, content_positions, min_content_index=2):
    """Position p of a content token whose id does not occur earlier in the
    sequence and which is not the last content token; None when no such
    token exists. Stopping on it must truncate the visible content."""
    for index, position in enumerate(content_positions[:-1]):
        if index >= min_content_index and ids[position] not in ids[:position]:
            return position
    return None


MESSAGE_ORDER = re.compile(r'S(?:bd*e)*DE')
MESSAGE_CODES = {'message_start': 'S', 'content_block_start': 'b', 'content_block_delta': 'd',
                 'content_block_stop': 'e', 'message_delta': 'D', 'message_stop': 'E'}


def message_event_order(raw):
    """Anthropic stream order: message_start, (content_block_start,
    content_block_delta*, content_block_stop)*, message_delta, message_stop;
    `ping` allowed anywhere. Returns (problems, event types)."""
    try:
        types = [event['type'] for event in parse_named_sse(raw)]
    except (ValueError, KeyError) as error:
        return ['unparseable stream: ' + str(error)], []
    codes = ''.join(MESSAGE_CODES.get(kind, '?') for kind in types if kind != 'ping')
    return ([] if MESSAGE_ORDER.fullmatch(codes) else ['event order violates the Messages contract']), types


def with_start_usage(body, events):
    """Anthropic streams may carry input_tokens only in message_start
    (message_delta usage then holds output_tokens). decode_messages_sse reads
    message_delta alone, so fill a missing/zero input_tokens from the start."""
    usage = dict(body.get('usage') or {})
    start = (((events or [{}])[0].get('message') or {}).get('usage') or {})
    if not usage.get('input_tokens') and start.get('input_tokens'):
        usage['input_tokens'] = start['input_tokens']
        return {**body, 'usage': usage}
    return body


def first_delta(events):
    """Text of the first non-empty reasoning/content delta of a chat stream."""
    for event in events or []:
        for choice in event.get('choices') or []:
            delta = choice.get('delta') or {}
            for key in ('reasoning_content', 'reasoning', 'content'):
                if isinstance(delta.get(key), str) and delta[key]:
                    return delta[key]
    return None


def first_divergence(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def stop_cut_problems(text, baseline):
    """A multi-token stop string must end generation AT the match: marker and
    everything after it absent, preceding text present, and the stopped text
    a prefix of the unstopped greedy baseline."""
    problems = []
    if STOP_MARKER in text:
        problems.append('stop string present in output')
    tail = [word for word in REACT_TAIL if word in text]
    if tail:
        problems.append('text after the stop string present: ' + repr(tail))
    if REACT_HEAD.split()[0] not in text:
        problems.append('text before the stop string missing')
    if baseline is not None:
        cut = baseline.find(STOP_MARKER)
        if cut < 0:
            problems.append('baseline never produced the stop string (precondition)')
        elif text.rstrip() != baseline[:cut].rstrip():
            problems.append('stopped text is not the baseline prefix before the stop string')
    return problems


def include_cut_problems(text, baseline):
    """include_stop_str_in_output: generation ends AT the match and the
    matched string is kept: the output is the unstopped greedy baseline up to
    and including the first stop string, nothing after it."""
    problems = []
    if not text.endswith(STOP_MARKER):
        problems.append('output does not end with the stop string')
    if text.count(STOP_MARKER) != 1:
        problems.append('stop string occurs %d times' % text.count(STOP_MARKER))
    tail = [word for word in REACT_TAIL if word in text]
    if tail:
        problems.append('text after the stop string present: ' + repr(tail))
    if baseline is not None:
        cut = baseline.find(STOP_MARKER)
        if cut < 0:
            problems.append('baseline never produced the stop string (precondition)')
        elif text != baseline[:cut + len(STOP_MARKER)]:
            problems.append('output is not the baseline prefix through the stop string')
    return problems


def responses_text(body, kind='output_text'):
    """Concatenated text of a Responses body: `output_text` parts of message
    items, or (kind='reasoning') every text part of reasoning items."""
    items = body.get('output') if isinstance(body, dict) else None
    if not isinstance(items, list):
        return ''
    parts = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if kind == 'reasoning' and item.get('type') == 'reasoning':
            for key in ('summary', 'content'):
                parts += [p.get('text') for p in item.get(key) or [] if isinstance(p, dict)]
        elif kind == 'output_text' and item.get('type') == 'message':
            parts += [p.get('text') for p in item.get('content') or []
                      if isinstance(p, dict) and p.get('type') == 'output_text']
    return ''.join(part for part in parts if isinstance(part, str))


def tool_call_problems(calls, expected):
    """Every call: identity, declared name, strict-JSON arguments valid for
    the declared schema; together exactly the expected (name, args) multiset."""
    if not isinstance(calls, list) or not calls:
        return ['tool_calls missing'], []
    problems, seen, ids = [], [], []
    for call in calls:
        function = call.get('function') if isinstance(call, dict) else None
        if not isinstance(function, dict):
            problems.append('call without function object')
            continue
        if call.get('type') != 'function' or not isinstance(call.get('id'), str) or not call['id']:
            problems.append('tool call identity missing')
        ids.append(call.get('id'))
        name = function.get('name')
        if name not in SCHEMAS:
            problems.append('undeclared tool: ' + repr(name))
            continue
        try:
            arguments = parse_strict_json(function.get('arguments'))
        except ValueError as error:
            problems.append(name + ' arguments are not valid JSON: ' + str(error))
            continue
        errors = schema_errors(arguments, SCHEMAS[name])
        if errors:
            problems.append(name + ' arguments violate schema: ' + '; '.join(errors))
        seen.append((name, arguments))
    if len(set(ids)) != len(ids):
        problems.append('duplicate tool call ids')
    key = (lambda pair: json.dumps(pair, sort_keys=True))
    if not problems and sorted(seen, key=key) != sorted(expected, key=key):
        problems.append('tool calls differ from the expected set')
    return problems, seen


# ---------------------------------------------------------------------------
# Case recorder
# ---------------------------------------------------------------------------

class Case:
    """Records assertions and request references for one receipt."""

    def __init__(self, name, ctx):
        self.name, self.ctx = name, ctx
        self.assertions, self.requests, self.bodies = [], [], []

    def check(self, name, passed, evidence=None):
        self.assertions.append({'assertion': name, 'status': 'pass' if passed else 'fail',
                                'evidence': evidence or {}})
        return bool(passed)

    def skip(self, name, reason):
        self.assertions.append({'assertion': name, 'status': 'skipped', 'evidence': {'reason': reason}})

    def problems(self, name, problems, evidence=None):
        return self.check(name, not problems, {**(evidence or {}), 'problems': problems})

    def request(self, tag, payload, surface='chat'):
        """Send one request; returns (body, events). A transport, HTTP or
        decode failure (including an SSE lifecycle/order violation) is a
        failed `<tag>:response` assertion carrying the error."""
        rid = self.name + '-' + tag
        receipt, body = self.ctx.io.request(rid, payload, surface)
        events = None
        events_path = self.ctx.io.out / (rid + '.events.json')
        if payload.get('stream') and events_path.exists():
            events = json.loads(events_path.read_text())
            if surface == 'messages' and body is not None:
                body = with_start_usage(body, events)
        self.requests.append({key: receipt.get(key) for key in
                              ('id', 'http_status', 'raw_file', 'raw_sha256', 'error', 'sse_shape')})
        if surface == 'messages' and payload.get('stream') and receipt.get('http_status') == 200:
            raw = (self.ctx.io.out / receipt['raw_file']).read_text(errors='replace')
            problems, types = message_event_order(raw)
            self.problems(tag + ':event_order', problems, {'events': types})
        self.check(tag + ':response', body is not None,
                   {'http_status': receipt.get('http_status'), 'error': clip(receipt.get('error')),
                    'error_body': clip(receipt.get('error_body'))})
        return body, events

    def expect_status(self, tag, payload, status, surface='chat'):
        """Send one request that must be REJECTED with `status` (e.g. 400)."""
        rid = self.name + '-' + tag
        receipt, _ = self.ctx.io.request(rid, payload, surface)
        self.requests.append({key: receipt.get(key) for key in
                              ('id', 'http_status', 'raw_file', 'raw_sha256', 'error')})
        got = receipt.get('http_status')
        return self.check(tag + ':rejected_%d' % status, got == status,
                          {'http_status': got, 'error_body': clip(receipt.get('error_body'))})

    def receipt(self, family, description, started, error=None):
        failed = [a['assertion'] for a in self.assertions if a['status'] == 'fail']
        skipped = [a['assertion'] for a in self.assertions if a['status'] == 'skipped']
        passed = error is None and not failed and any(a['status'] == 'pass' for a in self.assertions)
        return {'case': self.name, 'family': family, 'description': description,
                'passed': passed, 'error': error, 'failed_assertions': failed,
                'skipped_assertions': skipped, 'elapsed_s': round(time.time() - started, 3),
                'requests': self.requests, 'assertions': self.assertions}


# ---------------------------------------------------------------------------
# Shared assertion groups
# ---------------------------------------------------------------------------

def check_chat(case, tag, body, payload, finish, events=None):
    """finish_reason, usage, leak and reasoning-accounting gates for one
    chat response (blocking body or SSE-reconstructed body)."""
    allowed = finish if isinstance(finish, tuple) else (finish,)
    case.check(tag + ':finish_reason', chat_finish(body) in allowed,
               {'expected': list(allowed), 'got': chat_finish(body)})
    usage = body.get('usage')
    case.problems(tag + ':usage', usage_problems(usage, payload['max_tokens']), {'usage': usage})
    content, reasoning = chat_content(body), chat_reasoning(body)
    case.check(tag + ':no_special_tokens', not leaks(content) and not leaks(reasoning),
               {'content_leaks': leaks(content), 'reasoning_leaks': leaks(reasoning)})
    details = (usage or {}).get('completion_tokens_details') if isinstance(usage, dict) else None
    count = details.get('reasoning_tokens') if isinstance(details, dict) else None
    if count is None:
        case.skip(tag + ':reasoning_accounting', 'server reports no completion_tokens_details.reasoning_tokens')
    else:
        completion = usage.get('completion_tokens')
        ok = (type(count) is int and type(completion) is int and 0 <= count <= completion
              and (count > 0) == bool(reasoning))
        case.check(tag + ':reasoning_accounting', ok,
                   {'reasoning_tokens': count, 'reasoning_chars': len(reasoning)})
    if events is not None:
        ok, evidence = usage_chunk_last(events)
        case.check(tag + ':usage_chunk_last', ok, evidence)


def check_completion(case, tag, body, payload, finish, events=None):
    case.check(tag + ':object', body.get('object') == 'text_completion', {'object': body.get('object')})
    case.check(tag + ':finish_reason', chat_finish(body) == finish,
               {'expected': finish, 'got': chat_finish(body)})
    usage = body.get('usage')
    case.problems(tag + ':usage', usage_problems(usage, payload['max_tokens']), {'usage': usage})
    text = completion_text(body)
    case.check(tag + ':no_special_tokens', not leaks(text), {'leaks': leaks(text)})
    if events is not None:
        ok, evidence = usage_chunk_last(events)
        case.check(tag + ':usage_chunk_last', ok, evidence)


def check_message(case, tag, body, payload, stop_reason, stop_sequence=None):
    envelope = (body.get('type') == 'message' and body.get('role') == 'assistant'
                and isinstance(body.get('content'), list) and isinstance(body.get('id'), str)
                and bool(body['id']))
    case.check(tag + ':envelope', envelope, {key: body.get(key) for key in ('id', 'type', 'role')})
    case.check(tag + ':stop_reason', body.get('stop_reason') == stop_reason,
               {'expected': stop_reason, 'got': body.get('stop_reason')})
    case.check(tag + ':stop_sequence', body.get('stop_sequence') == stop_sequence,
               {'expected': stop_sequence, 'got': body.get('stop_sequence')})
    case.problems(tag + ':usage', message_usage_problems(body.get('usage'), payload['max_tokens']),
                  {'usage': body.get('usage')})
    text, thinking = message_text(body), message_text(body, 'thinking')
    case.check(tag + ':no_special_tokens', not leaks(text) and not leaks(thinking),
               {'text_leaks': leaks(text), 'thinking_leaks': leaks(thinking)})


def identity(case, tag, first, second, fields):
    """Blocking vs streamed (or repeated) responses must agree field by field."""
    if first is None or second is None:
        return case.check(tag, False, {'error': 'prerequisite response missing'})
    same = {name: get(first) == get(second) for name, get in fields.items()}
    return case.check(tag, all(same.values()), {
        'fields': same, 'first': {n: clip(repr(g(first))) for n, g in fields.items() if not same[n]},
        'second': {n: clip(repr(g(second))) for n, g in fields.items() if not same[n]}})


CHAT_IDENTITY = {'content': chat_content, 'reasoning': chat_reasoning, 'finish': chat_finish,
                 'usage': lambda b: {k: (b.get('usage') or {}).get(k)
                                     for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

class Context:
    def __init__(self, io, model, eos_token_ids=(), client_timeout=1.0, cancel_frames=3,
                 gauge_wait_s=5.0):
        self.io, self.model = io, model
        self.eos_token_ids = tuple(eos_token_ids)
        self.client_timeout, self.cancel_frames, self.gauge_wait_s = client_timeout, cancel_frames, gauge_wait_s

    def chat(self, prompt=None, stream=False, max_tokens=MAX_TOKENS, messages=None, thinking=None,
             **extra):
        value = {'model': self.model, 'messages': messages or [{'role': 'user', 'content': prompt}],
                 'max_tokens': max_tokens, 'temperature': 0, 'stream': stream}
        if stream:
            value['stream_options'] = {'include_usage': True}
        if thinking is not None:
            value['chat_template_kwargs'] = {'enable_thinking': thinking}
        value.update(extra)
        return value

    def completion(self, prompt, stream=False, max_tokens=MAX_TOKENS, **extra):
        value = {'model': self.model, 'prompt': prompt, 'max_tokens': max_tokens,
                 'temperature': 0, 'stream': stream}
        if stream:
            value['stream_options'] = {'include_usage': True}
        value.update(extra)
        return value

    def responses(self, prompt, max_tokens=MAX_TOKENS, **extra):
        value = {'model': self.model, 'input': prompt, 'max_output_tokens': max_tokens,
                 'temperature': 0, 'stream': False}
        value.update(extra)
        return value

    def messages(self, prompt, stream=False, max_tokens=MAX_TOKENS, **extra):
        value = {'model': self.model, 'max_tokens': max_tokens, 'temperature': 0, 'stream': stream,
                 'messages': [{'role': 'user', 'content': prompt}]}
        value.update(extra)
        return value


def case_chat_basic(case, ctx):
    bodies = {}
    for tag, stream in (('blocking', False), ('stream', True)):
        payload = ctx.chat(TOKYO_PROMPT, stream)
        body, events = case.request(tag, payload)
        if body is None:
            continue
        bodies[tag] = body
        case.check(tag + ':content', chat_content(body).strip() == TOKYO,
                   {'expected': TOKYO, 'content': clip(chat_content(body))})
        check_chat(case, tag, body, payload, 'stop', events)
    identity(case, 'stream_matches_blocking', bodies.get('blocking'), bodies.get('stream'), CHAT_IDENTITY)


def case_chat_length(case, ctx):
    # Reasoning models may spend all max_tokens in the reasoning channel, so
    # "visible output" means content or reasoning is non-empty.
    bodies = {}
    for tag, stream in (('blocking', False), ('stream', True)):
        payload = ctx.chat(COUNT_PROMPT, stream, LENGTH_TOKENS)
        body, events = case.request(tag, payload)
        if body is None:
            continue
        bodies[tag] = body
        completion = (body.get('usage') or {}).get('completion_tokens')
        case.check(tag + ':max_tokens_consumed', completion == LENGTH_TOKENS,
                   {'completion_tokens': completion, 'max_tokens': LENGTH_TOKENS})
        case.check(tag + ':output_nonempty', bool((chat_content(body) + chat_reasoning(body)).strip()),
                   {'content': clip(chat_content(body)), 'reasoning': clip(chat_reasoning(body))})
        check_chat(case, tag, body, payload, 'length', events)
    identity(case, 'stream_matches_blocking', bodies.get('blocking'), bodies.get('stream'), CHAT_IDENTITY)


def case_completions_basic(case, ctx):
    bodies = {}
    for tag, stream in (('blocking', False), ('stream', True)):
        payload = ctx.completion(CMPL_PROMPT, stream, 32, stop=[CMPL_STOP])
        body, events = case.request(tag, payload, 'completions')
        if body is None:
            continue
        bodies[tag] = body
        text = completion_text(body)
        case.check(tag + ':text', CMPL_ANSWER in text and CMPL_STOP not in text,
                   {'expected': CMPL_ANSWER + ' without ' + repr(CMPL_STOP), 'text': clip(text)})
        check_completion(case, tag, body, payload, 'stop', events)
    identity(case, 'stream_matches_blocking', bodies.get('blocking'), bodies.get('stream'),
             {'text': completion_text, 'finish': chat_finish, 'usage': CHAT_IDENTITY['usage']})


def case_completions_length(case, ctx):
    bodies = {}
    for tag, stream in (('blocking', False), ('stream', True)):
        payload = ctx.completion(CMPL_COUNT_PROMPT, stream, LENGTH_TOKENS)
        body, events = case.request(tag, payload, 'completions')
        if body is None:
            continue
        bodies[tag] = body
        completion = (body.get('usage') or {}).get('completion_tokens')
        case.check(tag + ':max_tokens_consumed', completion == LENGTH_TOKENS,
                   {'completion_tokens': completion, 'max_tokens': LENGTH_TOKENS})
        case.check(tag + ':text_nonempty', bool(completion_text(body).strip()),
                   {'text': clip(completion_text(body))})
        check_completion(case, tag, body, payload, 'length', events)
    identity(case, 'stream_matches_blocking', bodies.get('blocking'), bodies.get('stream'),
             {'text': completion_text, 'finish': chat_finish, 'usage': CHAT_IDENTITY['usage']})


def case_stop_string(case, ctx):
    baseline, _ = case.request('baseline', ctx.chat(REACT_PROMPT, thinking=False))
    base_text = chat_content(baseline) if baseline is not None else None
    if baseline is not None:
        case.check('baseline:contains_stop_string', STOP_MARKER in base_text,
                   {'content': clip(base_text), 'note': 'precondition: without it the cut is untested'})
    bodies = {}
    for tag, stream in (('blocking', False), ('stream', True)):
        payload = ctx.chat(REACT_PROMPT, stream, thinking=False, stop=[STOP_MARKER])
        body, events = case.request(tag, payload)
        if body is None:
            continue
        bodies[tag] = body
        text = chat_content(body)
        case.problems(tag + ':stopped_at_match', stop_cut_problems(text, base_text),
                      {'content': clip(text), 'reasoning': clip(chat_reasoning(body))})
        check_chat(case, tag, body, payload, 'stop', events)
        stopped = (body.get('usage') or {}).get('completion_tokens')
        full = ((baseline or {}).get('usage') or {}).get('completion_tokens')
        case.check(tag + ':fewer_tokens_than_baseline',
                   type(stopped) is int and type(full) is int and stopped < full,
                   {'stopped_completion_tokens': stopped, 'baseline_completion_tokens': full})
    identity(case, 'stream_matches_blocking', bodies.get('blocking'), bodies.get('stream'), CHAT_IDENTITY)


def case_stop_token_ids(case, ctx):
    payload = ctx.chat(COUNT_PROMPT, True, 128, thinking=False, return_token_ids=True)
    baseline, events = case.request('baseline', payload)
    if baseline is None:
        return
    ids, content_positions = stream_token_ids(events)
    if not case.check('baseline:token_ids_exposed', ids is not None,
                      {'note': 'stop_token_ids needs return_token_ids to choose a real vocabulary id'}):
        return
    position = pick_stop_token(ids, content_positions)
    if not case.check('baseline:stop_token_candidate', position is not None,
                      {'token_count': len(ids), 'content_tokens': len(content_positions)}):
        return
    stop_id = ids[position]
    payload = ctx.chat(COUNT_PROMPT, True, 128, thinking=False, return_token_ids=True,
                       stop_token_ids=[stop_id])
    body, events = case.request('stopped', payload)
    if body is None:
        return
    check_chat(case, 'stopped', body, payload, 'stop', events)
    base_text, text = chat_content(baseline), chat_content(body)
    case.check('stopped:content_truncated', base_text.startswith(text) and len(text) < len(base_text),
               {'stop_token_id': stop_id, 'position': position, 'content': clip(text),
                'baseline': clip(base_text)})
    completion = (body.get('usage') or {}).get('completion_tokens')
    case.check('stopped:completion_tokens', completion == position + 1,
               {'expected': position + 1, 'got': completion,
                'rule': 'stop token is generated, then ends the request (vLLM check_stop)'})
    got_ids, _ = stream_token_ids(events)
    if got_ids is None:
        case.skip('stopped:token_ids_prefix', 'stopped stream carried no token_ids')
    else:
        case.check('stopped:token_ids_prefix', got_ids in (ids[:position + 1], ids[:position]),
                   {'stopped_ids': got_ids[-4:], 'expected_tail': ids[max(0, position - 3):position + 1]})


def case_stop_include(case, ctx):
    baseline, _ = case.request('baseline', ctx.chat(REACT_PROMPT, thinking=False))
    base_text = chat_content(baseline) if baseline is not None else None
    if baseline is not None:
        case.check('baseline:contains_stop_string', STOP_MARKER in base_text,
                   {'content': clip(base_text), 'note': 'precondition: without it the cut is untested'})
    bodies = {}
    for tag, stream in (('blocking', False), ('stream', True)):
        payload = ctx.chat(REACT_PROMPT, stream, thinking=False, stop=[STOP_MARKER],
                           include_stop_str_in_output=True)
        body, events = case.request(tag, payload)
        if body is None:
            continue
        bodies[tag] = body
        text = chat_content(body)
        case.problems(tag + ':stop_string_included', include_cut_problems(text, base_text),
                      {'content': clip(text)})
        check_chat(case, tag, body, payload, 'stop', events)
    identity(case, 'stream_matches_blocking', bodies.get('blocking'), bodies.get('stream'), CHAT_IDENTITY)


def case_stop_empty(case, ctx):
    for stops in ([''], ['Observation:', '']):
        tag = 'chat-' + str(len(stops))
        case.expect_status(tag, ctx.chat(TOKYO_PROMPT, stop=stops), 400)
    case.expect_status('completions', ctx.completion(CMPL_PROMPT, stop=['']), 400, 'completions')
    case.expect_status('chat-string', ctx.chat(TOKYO_PROMPT, stop=''), 400)


def case_eos(case, ctx):
    payload = ctx.chat(TOKYO_PROMPT, True, return_token_ids=True)
    body, events = case.request('stream', payload)
    if body is None:
        return
    check_chat(case, 'stream', body, payload, 'stop', events)
    completion = (body.get('usage') or {}).get('completion_tokens')
    case.check('stream:ended_before_max_tokens', type(completion) is int and completion < payload['max_tokens'],
               {'completion_tokens': completion, 'max_tokens': payload['max_tokens']})
    case.check('stream:content', chat_content(body).strip() == TOKYO,
               {'expected': TOKYO, 'content': clip(chat_content(body))})
    ids, _ = stream_token_ids(events)
    if not ctx.eos_token_ids:
        case.skip('stream:last_token_is_eos', 'no --eos-token-id given')
    elif ids is None:
        case.skip('stream:last_token_is_eos', 'server exposed no token_ids')
    elif type(completion) is int and len(ids) < completion:
        case.skip('stream:last_token_is_eos',
                  'terminal sampled ID cannot be verified: stream carries %d IDs for %d completion tokens'
                  % (len(ids), completion))
    else:
        case.check('stream:last_token_is_eos',
                   len(ids) == completion and bool(ids) and ids[-1] in ctx.eos_token_ids,
                   {'last_ids': ids[-3:], 'eos_token_ids': list(ctx.eos_token_ids),
                    'exposed_token_ids': len(ids), 'completion_tokens': completion})


def case_messages_end_turn(case, ctx):
    bodies = {}
    for tag, stream in (('blocking', False), ('stream', True)):
        payload = ctx.messages(TOKYO_PROMPT, stream)
        body, _ = case.request(tag, payload, 'messages')
        if body is None:
            continue
        bodies[tag] = body
        check_message(case, tag, body, payload, 'end_turn')
        case.check(tag + ':text', message_text(body).strip() == TOKYO,
                   {'expected': TOKYO, 'text': clip(message_text(body))})
    identity(case, 'stream_matches_blocking', bodies.get('blocking'), bodies.get('stream'), MESSAGE_IDENTITY)


MESSAGE_IDENTITY = {'text': message_text, 'thinking': lambda b: message_text(b, 'thinking'),
                    'stop_reason': lambda b: b.get('stop_reason'),
                    'stop_sequence': lambda b: b.get('stop_sequence'),
                    'output_tokens': lambda b: (b.get('usage') or {}).get('output_tokens')}


def case_messages_max_tokens(case, ctx):
    bodies = {}
    for tag, stream in (('blocking', False), ('stream', True)):
        payload = ctx.messages(COUNT_PROMPT, stream, LENGTH_TOKENS)
        body, _ = case.request(tag, payload, 'messages')
        if body is None:
            continue
        bodies[tag] = body
        check_message(case, tag, body, payload, 'max_tokens')
        output = (body.get('usage') or {}).get('output_tokens')
        case.check(tag + ':max_tokens_consumed', output == LENGTH_TOKENS,
                   {'output_tokens': output, 'max_tokens': LENGTH_TOKENS})
    identity(case, 'stream_matches_blocking', bodies.get('blocking'), bodies.get('stream'), MESSAGE_IDENTITY)


def case_messages_stop_sequence(case, ctx):
    bodies = {}
    for tag, stream in (('blocking', False), ('stream', True)):
        payload = ctx.messages(REACT_PROMPT, stream, thinking={'type': 'disabled'},
                               stop_sequences=[STOP_MARKER])
        body, _ = case.request(tag, payload, 'messages')
        if body is None:
            continue
        bodies[tag] = body
        check_message(case, tag, body, payload, 'stop_sequence', STOP_MARKER)
        text = message_text(body)
        case.problems(tag + ':stopped_at_match', stop_cut_problems(text, None), {'text': clip(text)})
    identity(case, 'stream_matches_blocking', bodies.get('blocking'), bodies.get('stream'), MESSAGE_IDENTITY)


def seeded_pair(case, ctx, prompt, extra, validate=None):
    bodies, streams = [], []
    for tag in ('seeded-a', 'seeded-b'):
        payload = ctx.chat(prompt, True, 128, temperature=SAMPLE_TEMPERATURE, seed=SEED,
                           return_token_ids=True, **extra)
        body, events = case.request(tag, payload)
        if body is not None:
            check_chat(case, tag, body, payload, ('stop', 'length'), events)
            if validate is not None:
                validate(tag, body)
        bodies.append(body)
        streams.append(events)
    case.bodies = bodies
    if any(body is None for body in bodies):
        case.check('seeded:first_token', False, {'error': 'prerequisite response missing'})
        case.check('seeded:identical_output', False, {'error': 'prerequisite response missing'})
        return
    ids = [stream_token_ids(events)[0] for events in streams]
    if all(item is not None for item in ids):
        first = [item[:1] for item in ids]
        evidence = {'first_token_ids': first, 'first_divergence': first_divergence(*ids)}
    else:
        first = [first_delta(events) for events in streams]
        evidence = {'first_deltas': first, 'note': 'no token_ids; compared first non-empty delta'}
    case.check('seeded:first_token', first[0] == first[1] and first[0] not in (None, []), evidence)
    text = [chat_reasoning(body) + '\x00' + chat_content(body) for body in bodies]
    case.check('seeded:identical_output', text[0] == text[1] and CHAT_IDENTITY['usage'](bodies[0])
               == CHAT_IDENTITY['usage'](bodies[1]),
               {'first_char_divergence': first_divergence(*text),
                'a': clip(chat_content(bodies[0])), 'b': clip(chat_content(bodies[1]))})


def schema_format():
    return {'type': 'json_schema', 'json_schema': {'name': 'capital', 'strict': True, 'schema': CITY_SCHEMA}}


def validate_document(case, tag, body, schema, expected=None):
    content = chat_content(body)
    try:
        value = parse_strict_json(content)
    except ValueError as error:
        case.check(tag + ':json_parses', False, {'error': str(error), 'content': clip(content)})
        case.check(tag + ':schema_valid', False, {'error': 'content did not parse'})
        return None
    case.check(tag + ':json_parses', True, {'content': clip(content)})
    case.problems(tag + ':schema_valid', schema_errors(value, schema))
    if expected is not None:
        got = value if isinstance(value, dict) else {}
        wrong = {key: got.get(key) for key in expected if got.get(key) != expected[key]}
        case.check(tag + ':values', not wrong, {'wrong': wrong, 'expected': expected})
    return value


def case_seed(case, ctx):
    seeded_pair(case, ctx, JOKE_PROMPT, {'thinking': False})
    # Control: without it a server that ignores temperature (greedy) passes
    # the identity gates vacuously. A 128-token joke at temperature 1.0 makes
    # a coincidental match across seeds negligible.
    seeded = case.bodies
    payload = ctx.chat(JOKE_PROMPT, True, 128, thinking=False, temperature=SAMPLE_TEMPERATURE,
                       seed=SEED + 1, return_token_ids=True)
    other, _ = case.request('other-seed', payload)
    if other is None or not seeded or seeded[0] is None:
        case.check('seeded:other_seed_differs', False, {'error': 'prerequisite response missing'})
        return
    a = chat_reasoning(seeded[0]) + '\x00' + chat_content(seeded[0])
    b = chat_reasoning(other) + '\x00' + chat_content(other)
    case.check('seeded:other_seed_differs', a != b, {'seed': SEED, 'other_seed': SEED + 1,
                                                     'other': clip(chat_content(other))})


def case_responses_seed(case, ctx):
    texts = []
    for tag, seed in (('seeded-a', SEED), ('seeded-b', SEED), ('other-seed', SEED + 1)):
        payload = ctx.responses(JOKE_PROMPT, 128, temperature=SAMPLE_TEMPERATURE, seed=seed)
        body, _ = case.request(tag, payload, 'responses')
        texts.append(None if body is None else
                     (responses_text(body, 'reasoning'), responses_text(body)))
    if any(text is None for text in texts):
        case.check('seeded:identical_output', False, {'error': 'prerequisite response missing'})
        case.check('seeded:other_seed_differs', False, {'error': 'prerequisite response missing'})
        return
    case.check('seeded:non_empty', bool(texts[0][1]), {'output_text': clip(texts[0][1])})
    case.check('seeded:identical_output', texts[0] == texts[1],
               {'first_char_divergence': first_divergence(texts[0][1], texts[1][1]),
                'a': clip(texts[0][1]), 'b': clip(texts[1][1])})
    # Control: a server that ignores temperature passes identity vacuously.
    case.check('seeded:other_seed_differs', texts[0] != texts[2],
               {'seed': SEED, 'other_seed': SEED + 1, 'other': clip(texts[2][1])})


def case_seed_json_schema(case, ctx):
    seeded_pair(case, ctx, CITY_PROMPT, {'thinking': False, 'response_format': schema_format()},
                lambda tag, body: validate_document(case, tag, body, CITY_SCHEMA))


def case_structured_json_schema(case, ctx):
    bodies = {}
    for tag, stream in (('blocking', False), ('stream', True)):
        payload = ctx.chat(CITY_PROMPT, stream, thinking=False, response_format=schema_format())
        body, events = case.request(tag, payload)
        if body is None:
            continue
        bodies[tag] = body
        check_chat(case, tag, body, payload, 'stop', events)
        validate_document(case, tag, body, CITY_SCHEMA, CITY_VALUE)
    identity(case, 'stream_matches_blocking', bodies.get('blocking'), bodies.get('stream'), CHAT_IDENTITY)


OBJECT_SCHEMA = {'type': 'object', 'required': ['city', 'country'],
                 'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'}}}


def case_structured_json_object(case, ctx):
    payload = ctx.chat(OBJECT_PROMPT, thinking=False, response_format={'type': 'json_object'})
    body, _ = case.request('blocking', payload)
    if body is not None:
        check_chat(case, 'blocking', body, payload, 'stop')
        validate_document(case, 'blocking', body, OBJECT_SCHEMA, {'city': 'Tokyo', 'country': 'Japan'})


def tool_pair(case, ctx, prompt, tools, expected):
    bodies = {}
    for tag, stream in (('blocking', False), ('stream', True)):
        payload = ctx.chat(prompt, stream, tools=[TOOLS[name] for name in tools], tool_choice='auto')
        body, events = case.request(tag, payload)
        if body is None:
            continue
        bodies[tag] = body
        check_chat(case, tag, body, payload, 'tool_calls', events)
        calls = chat_message(body).get('tool_calls')
        case.check(tag + ':call_count', isinstance(calls, list) and len(calls) == len(expected),
                   {'expected': len(expected), 'got': len(calls) if isinstance(calls, list) else None})
        problems, seen = tool_call_problems(calls, expected)
        case.problems(tag + ':arguments', problems, {'calls': clip(json.dumps(seen))})
    def parsed(arguments):
        try:
            return parse_strict_json(arguments)
        except ValueError:
            return arguments  # invalid JSON is gated by :arguments; compare raw here

    strip = (lambda body: sorted(json.dumps([c.get('function', {}).get('name'),
                                             parsed(c.get('function', {}).get('arguments'))], sort_keys=True)
                                 for c in chat_message(body).get('tool_calls') or []))
    identity(case, 'stream_matches_blocking', bodies.get('blocking'), bodies.get('stream'),
             {'tool_calls': strip, 'finish': chat_finish})
    return bodies.get('blocking')


def case_tools_single(case, ctx):
    tool_pair(case, ctx, INVENTORY_PROMPT, ('lookup_inventory', 'get_weather'),
              [('lookup_inventory', INVENTORY_ARGUMENTS)])


def case_tools_parallel(case, ctx):
    tool_pair(case, ctx, PARALLEL_PROMPT, ('get_weather',),
              [('get_weather', arguments) for arguments in PARALLEL_ARGUMENTS])


def case_tools_followup(case, ctx):
    payload = ctx.chat(PARALLEL_PROMPT, tools=[TOOLS['get_weather']], tool_choice='auto')
    body, _ = case.request('calls', payload)
    if body is None:
        return
    calls = chat_message(body).get('tool_calls')
    problems, _ = tool_call_problems(calls, [('get_weather', a) for a in PARALLEL_ARGUMENTS])
    if not case.problems('calls:valid', problems):
        case.check('followup:response', False, {'error': 'no valid tool calls; no fabricated tool ids'})
        return
    results = []
    for call in calls:
        city = json.loads(call['function']['arguments'])['city']
        results.append({'role': 'tool', 'tool_call_id': call['id'],
                        'content': json.dumps({'city': city, 'unit': 'celsius',
                                               'temperature': WEATHER_RESULTS[city]})})
    history = [{'role': 'user', 'content': PARALLEL_PROMPT}, chat_message(body)] + results
    followup = ctx.chat(messages=history + [{'role': 'user', 'content': FOLLOWUP_PROMPT}],
                        tools=[TOOLS['get_weather']], tool_choice='none')
    answer, _ = case.request('followup', followup)
    if answer is None:
        return
    check_chat(case, 'followup', answer, followup, 'stop')
    case.check('followup:no_tool_calls', not chat_message(answer).get('tool_calls'),
               {'tool_calls': chat_message(answer).get('tool_calls')})
    # Tool results are counter-seasonal: only a model that read them answers Paris.
    case.check('followup:uses_tool_result', chat_content(answer).strip() == FOLLOWUP_ANSWER,
               {'expected': FOLLOWUP_ANSWER, 'content': clip(chat_content(answer)),
                'results': WEATHER_RESULTS})


def case_reasoning_on(case, ctx):
    bodies = {}
    for tag, stream in (('blocking', False), ('stream', True)):
        payload = ctx.chat(CALC_PROMPT, stream, thinking=True)
        body, events = case.request(tag, payload)
        if body is None:
            continue
        bodies[tag] = body
        check_chat(case, tag, body, payload, 'stop', events)
        content, reasoning = chat_content(body), chat_reasoning(body)
        case.check(tag + ':reasoning_present', bool(reasoning.strip()), {'reasoning': clip(reasoning)})
        case.check(tag + ':content', content.strip() == CALC_ANSWER,
                   {'expected': CALC_ANSWER, 'content': clip(content)})
        case.check(tag + ':separated', bool(reasoning.strip()) and reasoning.strip() not in content
                   and content.strip() != reasoning.strip(),
                   {'content_chars': len(content), 'reasoning_chars': len(reasoning)})
    identity(case, 'stream_matches_blocking', bodies.get('blocking'), bodies.get('stream'), CHAT_IDENTITY)


def case_reasoning_off(case, ctx):
    # Emptiness of the reasoning channel is NOT gated: the GLM-5.3 template
    # keeps brief native reasoning with thinking off (README, Q3 section).
    bodies = {}
    for tag, stream in (('blocking', False), ('stream', True)):
        payload = ctx.chat(CALC_PROMPT, stream, thinking=False)
        body, events = case.request(tag, payload)
        if body is None:
            continue
        bodies[tag] = body
        check_chat(case, tag, body, payload, 'stop', events)
        content, reasoning = chat_content(body), chat_reasoning(body)
        case.check(tag + ':content', content.strip() == CALC_ANSWER,
                   {'expected': CALC_ANSWER, 'content': clip(content), 'reasoning_chars': len(reasoning)})
    identity(case, 'stream_matches_blocking', bodies.get('blocking'), bodies.get('stream'), CHAT_IDENTITY)


def case_multi_turn(case, ctx):
    first, _ = case.request('turn-1', ctx.chat(RECALL_PROMPT))
    if first is None:
        return
    reply = chat_content(first)
    messages = [{'role': 'user', 'content': RECALL_PROMPT}, {'role': 'assistant', 'content': reply},
                {'role': 'user', 'content': RECALL_QUESTION}]
    for tag, stream in (('turn-2', False), ('turn-2-stream', True)):
        payload = ctx.chat(stream=stream, messages=messages)
        body, events = case.request(tag, payload)
        if body is None:
            continue
        check_chat(case, tag, body, payload, 'stop', events)
        case.check(tag + ':recalled', CODEWORD in chat_content(body),
                   {'expected': CODEWORD, 'content': clip(chat_content(body))})


def interruption_case(surface, mode):
    """Baseline, health, interrupt (cancel after N SSE frames or client
    timeout), health, the identical baseline request again (must match: a
    leaked KV/scheduler state would perturb it), then the Atlas active-request
    gauge when the server exports it."""
    def run(case, ctx):
        def base_payload():
            return (ctx.chat(TOKYO_PROMPT, seed=42) if surface == 'chat'
                    else ctx.messages(TOKYO_PROMPT))
        get_text = chat_content if surface == 'chat' else message_text
        key = ((lambda b: (chat_content(b), chat_reasoning(b), CHAT_IDENTITY['usage'](b))) if surface == 'chat'
               else (lambda b: (message_text(b), message_text(b, 'thinking'),
                                (b.get('usage') or {}).get('output_tokens'))))
        baseline, _ = case.request('baseline', base_payload(), surface)
        if baseline is not None:
            case.check('baseline:answer', get_text(baseline).strip() == TOKYO, {'text': clip(get_text(baseline))})
        health(case, ctx, 'health-before')
        stream = mode == 'cancel'
        long_payload = (ctx.chat(LONG_PROMPT, stream, LONG_TOKENS) if surface == 'chat'
                        else ctx.messages(LONG_PROMPT, stream, LONG_TOKENS))
        limit = ctx.cancel_frames if stream else ctx.client_timeout
        receipt, _ = ctx.io.interrupt(case.name + '-interrupt', long_payload, surface, mode, limit)
        case.requests.append({key_: receipt.get(key_) for key_ in
                              ('id', 'interrupt', 'limit', 'http_status', 'raw_file', 'client_error',
                               'frames_before_cancel', 'completed_before_cancel', 'completed_before_timeout')})
        if stream:
            case.check('interrupt:cancelled_mid_stream', receipt.get('http_status') == 200
                       and receipt.get('frames_before_cancel') == limit and not receipt.get('completed_before_cancel'),
                       {key_: receipt.get(key_) for key_ in ('http_status', 'frames_before_cancel', 'client_error')})
        else:
            case.check('interrupt:client_timed_out', receipt.get('completed_before_timeout') is False,
                       {'client_timeout_s': limit, 'client_error': clip(receipt.get('client_error')),
                        'http_status': receipt.get('http_status')})
        health(case, ctx, 'health-after')
        after, _ = case.request('next', base_payload(), surface)
        if after is not None:
            case.check('next:answer', get_text(after).strip() == TOKYO, {'text': clip(get_text(after))})
        if baseline is not None and after is not None:
            case.check('next:matches_baseline', key(baseline) == key(after),
                       {'baseline': clip(repr(key(baseline))), 'next': clip(repr(key(after)))})
        gauge(case, ctx)
    return run


def health(case, ctx, tag):
    receipt, text = ctx.io.probe(case.name + '-' + tag, 'health')
    status = None
    try:
        parsed = json.loads(text) if text.strip() else None
        status = parsed.get('status') if isinstance(parsed, dict) else None
    except ValueError:
        parsed = None
    ok = receipt.get('http_status') == 200 and status in (None, 'ready', 'ok', 'healthy')
    case.check(tag, ok, {'http_status': receipt.get('http_status'), 'body': clip(text, 160)})


def gauge(case, ctx):
    deadline = time.time() + ctx.gauge_wait_s
    while True:
        receipt, text = ctx.io.probe(case.name + '-metrics', 'metrics')
        if receipt.get('http_status') is None:  # transport failure is not "no gauge"
            case.check('metrics:requests_active_zero', False, {'error': clip(receipt.get('error'))})
            return
        values = ACTIVE_GAUGE.findall(text) if receipt.get('http_status') == 200 else []
        if not values:
            case.skip('metrics:requests_active_zero', 'no atlas_requests_active gauge on /metrics')
            return
        if (len(values) == 1 and float(values[0]) == 0) or time.time() >= deadline:
            break
        time.sleep(0.25)
    case.check('metrics:requests_active_zero', len(values) == 1 and float(values[0]) == 0,
               {'atlas_requests_active': values})


CASES = {}


def register(name, family, description, fn):
    CASES[name] = {'family': family, 'description': description, 'run': fn}


register('chat-basic', 'openai-chat', 'blocking + SSE greedy answer, finish stop, usage, include_usage order',
         case_chat_basic)
register('chat-length', 'openai-chat', 'max_tokens finish length consumes exactly max_tokens', case_chat_length)
register('completions-basic', 'openai-completions', 'legacy completions blocking + SSE, stop, usage order',
         case_completions_basic)
register('completions-length', 'openai-completions', 'legacy completions finish length', case_completions_length)
register('stop-string', 'stop', 'multi-token stop string ends generation at the match (blocking + SSE)',
         case_stop_string)
register('stop-token-ids', 'stop', 'stop_token_ids ends generation at the chosen token', case_stop_token_ids)
register('stop-include', 'stop', 'include_stop_str_in_output keeps the matched stop string (blocking + SSE)',
         case_stop_include)
register('stop-empty', 'stop', 'an empty stop string is rejected with 400 (chat + completions)',
         case_stop_empty)
register('eos', 'stop', 'natural EOS: finish stop before max_tokens, no leaked specials', case_eos)
register('messages-end-turn', 'anthropic', 'Messages blocking + SSE event order, end_turn, usage',
         case_messages_end_turn)
register('messages-max-tokens', 'anthropic', 'Messages stop_reason max_tokens', case_messages_max_tokens)
register('messages-stop-sequence', 'anthropic', 'Messages stop_sequence echoed, text cut at the match',
         case_messages_stop_sequence)
register('seed', 'seed', 'same seed + temperature twice: identical first token and text', case_seed)
register('responses-seed', 'seed', 'Responses API: same seed twice identical, other seed differs',
         case_responses_seed)
register('seed-json-schema', 'seed', 'seeded repeat under json_schema', case_seed_json_schema)
register('structured-json-schema', 'structured', 'json_schema output parses, validates, exact values',
         case_structured_json_schema)
register('structured-json-object', 'structured', 'json_object output parses to the requested object',
         case_structured_json_object)
register('tools-single', 'tools', 'one tool call, schema-valid arguments, finish tool_calls', case_tools_single)
register('tools-parallel', 'tools', 'two parallel calls, distinct ids, exact arguments', case_tools_parallel)
register('tools-followup', 'tools', 'tool results sent back; the answer depends on them', case_tools_followup)
register('reasoning-on', 'reasoning', 'enable_thinking true: reasoning separated from content',
         case_reasoning_on)
register('reasoning-off', 'reasoning', 'enable_thinking false: exact answer, no leaked markers',
         case_reasoning_off)
register('multi-turn', 'recall', 'second turn recalls a code word from the first', case_multi_turn)
# Interruptions run last so a wedged server cannot mask the other families.
register('cancel-chat', 'robustness', 'cancel a chat stream mid-way, then health + identical next request',
         interruption_case('chat', 'cancel'))
register('cancel-messages', 'robustness', 'cancel a Messages stream mid-way, then recovery',
         interruption_case('messages', 'cancel'))
register('timeout-chat', 'robustness', 'client timeout on a long chat request, then recovery',
         interruption_case('chat', 'timeout'))
register('timeout-messages', 'robustness', 'client timeout on a long Messages request, then recovery',
         interruption_case('messages', 'timeout'))


# ---------------------------------------------------------------------------
# Runner and CLI
# ---------------------------------------------------------------------------

def write_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def run_suite(endpoint, model, out, only=None, timeout=600.0, eos_token_ids=(), client_timeout=1.0,
              cancel_frames=3, gauge_wait_s=5.0):
    """Run the selected cases; never stops early. Returns the summary."""
    names = list(CASES) if not only else list(dict.fromkeys(only))
    unknown = [name for name in names if name not in CASES]
    if unknown:
        raise ValueError('unknown case(s): ' + ', '.join(unknown))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    io = BehaviorIORouter(endpoint, out / 'raw', timeout)
    ctx = Context(io, model, eos_token_ids, client_timeout, cancel_frames, gauge_wait_s)
    rows = {}
    for name in names:
        spec, case, started, error = CASES[name], Case(name, ctx), time.time(), None
        try:
            spec['run'](case, ctx)
        except Exception as exc:  # recorded; the suite always continues
            error = type(exc).__name__ + ': ' + str(exc)
        receipt = case.receipt(spec['family'], spec['description'], started, error)
        write_json(out / (name + '.json'), receipt)
        rows[name] = receipt
    source = Path(__file__).read_bytes()
    summary = {'suite': 'q3_behavior', 'endpoint': endpoint, 'model': model,
               'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'suite_sha256': hashlib.sha256(source).hexdigest(),
               'passed': all(row['passed'] for row in rows.values()),
               'cases_passed': sum(row['passed'] for row in rows.values()), 'cases_total': len(rows),
               'cases': {name: {'passed': row['passed'], 'family': row['family'], 'error': row['error'],
                                'failed_assertions': row['failed_assertions'],
                                'skipped_assertions': row['skipped_assertions']}
                         for name, row in rows.items()}}
    write_json(out / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='Q3 live API behavior suite (receipts per case).')
    parser.add_argument('--endpoint', required=True, help='server origin, e.g. http://host:8000 (no /v1)')
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True, help='new receipt directory; refuses to overwrite')
    parser.add_argument('--only', nargs='+', metavar='CASE', help='run only these cases: ' + ' '.join(CASES))
    parser.add_argument('--timeout', type=float, default=600.0, help='per-request timeout seconds')
    parser.add_argument('--eos-token-id', type=int, action='append', default=[],
                        help='EOS token id(s) for the eos case; without it that assertion is skipped')
    parser.add_argument('--client-timeout', type=float, default=1.0,
                        help='client timeout (s) for the timeout-* cases')
    args = parser.parse_args(argv)
    if Path(args.out).exists():
        parser.error('--out exists; refusing to overwrite: ' + args.out)
    try:
        summary = run_suite(args.endpoint, args.model, args.out, args.only, args.timeout,
                            args.eos_token_id, args.client_timeout)
    except ValueError as error:
        parser.error(str(error))
    for name, row in summary['cases'].items():
        state = 'PASS' if row['passed'] else 'FAIL'
        print(state, name, ' '.join(row['failed_assertions']) or row['error'] or '')
    print('{}/{} cases passed; receipts in {}'.format(summary['cases_passed'], summary['cases_total'], args.out))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
