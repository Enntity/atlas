# SPDX-License-Identifier: AGPL-3.0-only
"""OpenAI tool-calling and structured-output behavior cases.

Server facts used (crates/spark-server/src):
- tool_choice accepts 'auto' | 'none' | 'required' | {type:function,
  function:{name}}; any other string is a 400 naming tool_choice
  (api/chat_phases.rs). 'required'/named are grammar-forced and stop after
  the FIRST call; parallel calls are only possible under 'auto'.
- poolside_v1 grammar constrains tag/key names, not value types or required
  keys, and backfill can insert a missing required string as "" — so every
  argument object is checked against the declared schema here.
- Streaming poolside_v1 arguments arrive whole after </tool_call>; the
  decoder still concatenates fragments, so split delivery also passes.
- thinking_in_tools=false (MODEL.toml): a tool turn with no thinking
  directive renders '<think></think>' (glm5_next.jinja: low_think needs
  `not tools`), so it must report no reasoning.
- response_format json_schema / json_object is grammar-enforced only after
  </think> and only without active tools; reasoning may precede the JSON in
  reasoning_content, content must be the bare document.
"""
import json

from behavior_protocol import validate_body
from behavior_schema import validate_json_text

INVENTORY_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['sku', 'warehouse'],
                    'properties': {'sku': {'type': 'string', 'minLength': 1},
                                   'warehouse': {'type': 'string', 'enum': ['north', 'south']}}}
WEATHER_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['city', 'unit'],
                  'properties': {'city': {'type': 'string', 'minLength': 1},
                                 'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}}}
SCHEMAS = {'lookup_inventory': INVENTORY_SCHEMA, 'get_weather': WEATHER_SCHEMA}
TOOLS = {name: {'type': 'function', 'function': {'name': name, 'description': description,
                                                 'parameters': SCHEMAS[name]}}
         for name, description in (('lookup_inventory', 'Look up inventory for a SKU.'),
                                   ('get_weather', 'Current temperature for one city.'))}
INVENTORY_PROMPT = 'Use lookup_inventory for SKU BLUE-17 in warehouse north.'
INVENTORY_ARGUMENTS = {'sku': 'BLUE-17', 'warehouse': 'north'}
PARALLEL_PROMPT = ('Get the current weather in celsius for Paris and for Tokyo. Call get_weather '
                   'once per city, both calls in this same turn, then wait for the results.')
PARALLEL_ARGUMENTS = [{'city': 'Paris', 'unit': 'celsius'}, {'city': 'Tokyo', 'unit': 'celsius'}]
# Deliberately counter-seasonal so the follow-up answer depends on the
# tool results, not on the model's priors.
WEATHER_RESULTS = {'Paris': 31, 'Tokyo': 12}
FOLLOWUP_PROMPT = 'Using only the tool results, reply with only the name of the warmer city.'
FOLLOWUP_ANSWER = 'Paris'
TOKYO_PROMPT = 'What is the capital of Japan? Reply only with Tokyo and no punctuation.'
NO_TOOL_PROMPT = 'What is 2 + 2? Do not use any tool. Reply only with the integer.'
CITY_SCHEMA = {'type': 'object', 'additionalProperties': False,
               'required': ['city', 'country', 'is_capital'],
               'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'},
                              'is_capital': {'type': 'boolean'}}}
CITY_PROMPT = ('Return JSON describing the capital of Japan with fields city, country and '
               'is_capital.')
CITY_VALUE = {'city': 'Tokyo', 'country': 'Japan', 'is_capital': True}
OBJECT_PROMPT = ('Return a JSON object with exactly two string keys, city and country, for the '
                 'capital of Japan.')
# json_object only guarantees an object; an extra key would be a model
# choice, not an Atlas fault, so extras are allowed and city/country exact.
OBJECT_SCHEMA = {'type': 'object', 'additionalProperties': True, 'required': ['city', 'country'],
                 'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'}}}
MAX_TOKENS = 256


def tool_calls_summary(body, expected, ordered=False):
    """Validate every tool call: identity, declared name, schema-valid JSON
    arguments, and the exact expected (name, arguments) multiset."""
    calls = body['choices'][0]['message'].get('tool_calls')
    if not isinstance(calls, list) or not calls:
        raise ValueError('tool_calls required')
    seen, ids = [], set()
    for call in calls:
        function = call.get('function') or {}
        if call.get('type') != 'function' or not isinstance(call.get('id'), str) or not call['id']:
            raise ValueError('tool call identity missing')
        if call['id'] in ids:
            raise ValueError('duplicate tool call id')
        ids.add(call['id'])
        name = function.get('name')
        if name not in SCHEMAS:
            raise ValueError('undeclared tool called: ' + repr(name))
        if not isinstance(function.get('arguments'), str):
            raise ValueError('arguments must be a JSON string')
        seen.append((name, validate_json_text(function['arguments'], SCHEMAS[name])))
    key = (lambda pair: json.dumps(pair, sort_keys=True))
    if (seen if ordered else sorted(seen, key=key)) != (expected if ordered else sorted(expected, key=key)):
        raise ValueError('tool calls differ from the expected set: ' + repr(seen))
    return [{'id': call['id'], 'name': name, 'arguments': arguments}
            for call, (name, arguments) in zip(calls, seen)]


def collect_tools(io, model, context_limit):
    rows = {}

    def chat(prompt=None, stream=False, tools=(), choice=None, messages=None, fmt=None, thinking=None):
        value = {'model': model, 'messages': messages or [{'role': 'user', 'content': prompt}],
                 'temperature': 0, 'seed': 42, 'max_tokens': MAX_TOKENS, 'stream': stream}
        if stream:
            value['stream_options'] = {'include_usage': True}
        if tools:
            value['tools'] = [TOOLS[name] for name in tools]
        if choice is not None:
            value['tool_choice'] = choice
        if fmt is not None:
            value['response_format'] = fmt
        if thinking is not None:
            value['chat_template_kwargs'] = {'enable_thinking': thinking}
        return value

    def run(name, task, finish, check):
        receipt, body = io.request(name, task, 'chat')
        if body is not None:
            try:
                summary = validate_body(body, finish, task['max_tokens'], context_limit)
                summary['checked'] = check(body, summary)
                receipt['summary'] = summary
                receipt['passed'] = True
            except Exception as error:
                receipt['error'] = type(error).__name__ + ': ' + str(error)
        rows[name] = receipt
        return receipt, body

    def no_reasoning(summary):
        if summary['reasoning'] or summary['reasoning_tokens'] != 0:
            raise ValueError('tool turn without a thinking directive reasoned '
                             '(MODEL.toml thinking_in_tools=false)')

    def calls(expected, ordered=False):
        def check(body, summary):
            no_reasoning(summary)
            return tool_calls_summary(body, expected, ordered)
        return check

    def text(answer):
        def check(body, summary):
            if body['choices'][0]['message'].get('tool_calls'):
                raise ValueError('unexpected tool call')
            if summary['content'].strip() != answer:
                raise ValueError('wrong visible answer')
            return summary['content'].strip()
        return check

    def document(schema, value):
        def check(body, summary):
            parsed = validate_json_text(summary['content'], schema)
            if any(parsed.get(key) != expected for key, expected in value.items()):
                raise ValueError('structured output values wrong: ' + repr(parsed))
            return parsed
        return check

    # ---- tool_choice modes -------------------------------------------------
    run('tools-required', chat(INVENTORY_PROMPT, tools=('lookup_inventory', 'get_weather'),
                               choice='required'), 'tool_calls',
        calls([('lookup_inventory', INVENTORY_ARGUMENTS)]))
    run('tools-named-stream', chat(INVENTORY_PROMPT, True, ('lookup_inventory', 'get_weather'),
                                   {'type': 'function', 'function': {'name': 'lookup_inventory'}}),
        'tool_calls', calls([('lookup_inventory', INVENTORY_ARGUMENTS)]))
    run('tools-none', chat(TOKYO_PROMPT, tools=('lookup_inventory',), choice='none'), 'stop',
        text('Tokyo'))
    run('tools-auto-no-call', chat(NO_TOOL_PROMPT, tools=('get_weather',), choice='auto'), 'stop',
        text('4'))
    receipt, _ = io.request('tools-invalid-choice',
                            chat(TOKYO_PROMPT, tools=('get_weather',), choice='sometimes'), 'chat')
    if receipt.get('http_status') == 400 and 'tool_choice' in receipt.get('error_body', ''):
        receipt['passed'] = True
        receipt['error'] = None
    else:
        receipt['error'] = 'invalid tool_choice must be a 400 naming tool_choice: ' + repr(receipt.get('error'))
    rows['tools-invalid-choice'] = receipt

    # ---- parallel calls, streamed assembly, dependent follow-up ------------
    expected = [('get_weather', arguments) for arguments in PARALLEL_ARGUMENTS]
    first, body = run('tools-parallel', chat(PARALLEL_PROMPT, tools=('get_weather',), choice='auto'),
                      'tool_calls', calls(expected))
    streamed, _ = run('tools-parallel-stream',
                      chat(PARALLEL_PROMPT, True, ('get_weather',), 'auto'), 'tool_calls', calls(expected))
    if first.get('passed') and streamed.get('passed'):
        strip = (lambda items: [(item['name'], item['arguments']) for item in items])
        same = strip(first['summary']['checked']) == strip(streamed['summary']['checked'])
        streamed['blocking_identity'] = {'tool_calls': same}
        if not same:
            streamed['passed'] = False
            streamed['error'] = 'streamed tool-call assembly differs from blocking'
    if first.get('passed'):
        message = body['choices'][0]['message']
        results = [{'role': 'tool', 'tool_call_id': call['id'],
                    'content': json.dumps({'city': call['arguments']['city'], 'unit': 'celsius',
                                           'temperature': WEATHER_RESULTS[call['arguments']['city']]})}
                   for call in first['summary']['checked']]
        history = [{'role': 'user', 'content': PARALLEL_PROMPT}, message] + results
        run('tools-parallel-followup',
            chat(tools=('get_weather',), choice='none',
                 messages=history + [{'role': 'user', 'content': FOLLOWUP_PROMPT}]),
            'stop', text(FOLLOWUP_ANSWER))
    else:
        rows['tools-parallel-followup'] = {'id': 'tools-parallel-followup', 'passed': False,
                                           'error': 'valid parallel calls missing; no fabricated tool IDs'}

    # ---- structured output -------------------------------------------------
    schema_format = {'type': 'json_schema', 'json_schema': {'name': 'capital', 'strict': True,
                                                            'schema': CITY_SCHEMA}}
    blocking, _ = run('format-json-schema', chat(CITY_PROMPT, fmt=schema_format, thinking=False),
                      'stop', document(CITY_SCHEMA, CITY_VALUE))
    streamed, _ = run('format-json-schema-stream',
                      chat(CITY_PROMPT, True, fmt=schema_format, thinking=False),
                      'stop', document(CITY_SCHEMA, CITY_VALUE))
    if blocking.get('passed') and streamed.get('passed'):
        same = blocking['summary']['content'] == streamed['summary']['content']
        streamed['blocking_identity'] = {'content': same}
        if not same:
            streamed['passed'] = False
            streamed['error'] = 'streamed structured output differs from blocking'
    run('format-json-object', chat(OBJECT_PROMPT, fmt={'type': 'json_object'}, thinking=False),
        'stop', document(OBJECT_SCHEMA, {'city': 'Tokyo', 'country': 'Japan'}))
    return list(rows.values())
