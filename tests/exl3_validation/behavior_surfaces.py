# SPDX-License-Identifier: AGPL-3.0-only
"""OpenAI chat + legacy /v1/completions behavior cases: streaming and nonstreaming.

Covers finish_reason stop vs length, stop sequences, max_tokens/length
accounting, stream(include_usage)/blocking token-accounting consistency for
one seeded greedy request, multi-turn recall (including a per-turn thinking
on/off toggle inside one conversation), seeded sampling, and reasoning
on/off (chat_template_kwargs / reasoning fields) with reasoning_content
separation. The legacy surface exposes no thinking fields, so reasoning
coverage is chat-only; the special-token leak gate runs on every text channel
of both surfaces (behavior_protocol). Every case asserts an exact expected
outcome — never mere non-emptiness.

This module is numpy-free so the CPU tests can exercise it directly; it is
run from collect_behavior.collect() (one manifest, one receipt set).
"""
from behavior_io import BehaviorIORouter
from behavior_protocol import usage_follows_finish, validate_body, validate_completion

ANSWER = 'Tokyo'
TOKYO_PROMPT = 'What is the capital of Japan? Reply only with Tokyo and no punctuation.'
CALC_PROMPT = 'Calculate 37 * 14 - 86. Reply only with the integer answer.'
CALC_ANSWER = '432'
LONG_PROMPT = 'Count from 1 to 30, separated by single spaces.'
STOP_MARKER = 'STOPNOW'
STOP_PROMPT = 'Write exactly the phrase alpha STOPNOW omega and nothing else.'
CODEWORD = 'ZORBLATT-7'
RECALL_PROMPT = 'Remember this code word: ZORBLATT-7. Reply only with OK.'
RECALL_QUESTION = 'What code word did I ask you to remember? Reply with only the code word.'
JOKE_PROMPT = 'Give me one short lighthearted joke.'
MAX_TOKENS = 256
LENGTH_TOKENS = 8
SEED = 42
SAMPLE_TEMPERATURE = 0.8
THINKING_BUDGET = 32

# Legacy /v1/completions applies no chat template (api/completions.rs encodes
# the prompt verbatim, encode(text, add_special_tokens=false)) and runs no
# reasoning parser (extract_thinking is imported there but never called), so
# any reasoning the model wrote would land in `text`. Raw prose also gives a
# chat checkpoint no reason to emit EOS. The legacy prompts therefore carry
# the exact render jinja-templates/openai/glm5_next.jinja produces for
# thinking off with glm_empty_think_off (ATLAS_GLM_EMPTY_THINK_OFF=1):
# [gMASK]<sop><|system|>Reasoning Effort: Max, <|user|> turns, and
# <|assistant|><think></think>. The default (low_think) render ends in an
# open <think>, which on this surface would put reasoning and </think> into
# the visible text. The model then ends its turn naturally, making
# finish_reason 'stop' an objective expectation.
def glm_frame(*turns):
    """turns: alternating user / assistant texts, ending with a user turn."""
    out = '[gMASK]<sop><|system|>Reasoning Effort: Max'
    for index, text in enumerate(turns):
        out += ('<|user|>' + text) if index % 2 == 0 else ('<|assistant|><think></think>' + text)
    return out + '<|assistant|><think></think>'


TOKYO_LINE = glm_frame(TOKYO_PROMPT)
LONG_LINE = glm_frame(LONG_PROMPT)
STOP_LINE = glm_frame(STOP_PROMPT)
RECALL_LINE = glm_frame(RECALL_PROMPT, 'OK', RECALL_QUESTION)
JOKE_LINE = glm_frame(JOKE_PROMPT)


def collect_surfaces(io, model, context_limit):
    """Run the surface matrix; returns receipts in the existing format."""
    rows = {}

    def chat_payload(prompt, stream=False, thinking=None, stop=None,
                     max_tokens=MAX_TOKENS, seed=SEED, temperature=0,
                     budget=None, messages=None, effort=None):
        value = {'model': model,
                 'messages': messages or [{'role': 'user', 'content': prompt}],
                 'temperature': temperature, 'seed': seed,
                 'max_tokens': max_tokens, 'stream': stream}
        if stream:
            value['stream_options'] = {'include_usage': True}
        if thinking is not None:
            value['chat_template_kwargs'] = {'enable_thinking': thinking}
        if budget is not None:
            value['thinking_token_budget'] = budget
        if effort is not None:
            value['reasoning_effort'] = effort
        if stop:
            value['stop'] = stop
        return value

    def cmpl_payload(prompt, stream=False, stop=None,
                     max_tokens=MAX_TOKENS, seed=SEED, temperature=0):
        value = {'model': model, 'prompt': prompt, 'temperature': temperature,
                 'seed': seed, 'max_tokens': max_tokens, 'stream': stream}
        if stream:
            value['stream_options'] = {'include_usage': True}
        if stop:
            value['stop'] = stop
        return value

    def run(name, task, surface, finish, budget=None):
        receipt, body = io.request(name, task, surface)
        if body is not None:
            try:
                if surface == 'chat':
                    summary = validate_body(body, finish, task['max_tokens'],
                                            context_limit, budget)
                else:
                    summary = validate_completion(body, finish, task['max_tokens'],
                                                  context_limit)
                receipt['summary'] = summary
                receipt['passed'] = True
            except Exception as error:
                receipt['error'] = type(error).__name__ + ': ' + str(error)
        rows[name] = receipt
        return receipt

    def demand(receipt, predicate, message):
        """Objective post-condition; skipped when the base gate already failed."""
        if not receipt.get('passed') or 'summary' not in receipt:
            return
        if not predicate(receipt['summary']):
            receipt['passed'] = False
            receipt['error'] = message

    def visible(summary):
        return summary['content'] if 'content' in summary else summary['text']

    def answered(word):
        return lambda summary: word in visible(summary)

    def exact(word):
        return lambda summary: visible(summary).strip() == word

    def finished_length():
        return lambda summary: summary['accounting']['completion_tokens'] == LENGTH_TOKENS

    def stop_stripped():
        # The marker sits MID-phrase: a server that only strips a trailing
        # stop string (instead of stopping generation at the match) leaves
        # 'omega' behind and fails here.
        return lambda summary: (STOP_MARKER not in visible(summary) and 'omega' not in visible(summary)
                                and 'alpha' in visible(summary))

    def separated(summary):
        return summary['content'] != summary['reasoning']

    def reasoning_present(summary):
        return bool(summary['reasoning']) and summary['reasoning_tokens'] > 0

    def pair_gate(a, b, identity, key):
        if not (a.get('passed') and b.get('passed')):
            return
        b[key] = identity
        if not all(identity.values()):
            b['passed'] = False
            b['error'] = key + ' mismatch: ' + repr(identity)

    def seed_pair(first, second):
        if not (first.get('passed') and second.get('passed')):
            return
        identity = {
            'text': visible(first['summary']) == visible(second['summary']),
            'accounting': first['summary']['accounting'] == second['summary']['accounting'],
        }
        pair_gate(first, second, identity, 'seeded_identity')

    def stream_pair(blocked, streamed):
        if not (blocked.get('passed') and streamed.get('passed')):
            return
        identity = {
            'text': visible(blocked['summary']) == visible(streamed['summary']),
            'accounting': blocked['summary']['accounting'] == streamed['summary']['accounting'],
        }
        pair_gate(blocked, streamed, identity, 'blocking_identity')

    def usage_order(receipt):
        # Gated on ONE row per surface so an ordering defect stays visible
        # without masking the content checks of every other stream case.
        if receipt.get('passed') and not usage_follows_finish(receipt.get('sse_shape', [])):
            receipt['passed'] = False
            receipt['error'] = ('include_usage chunk must follow the finish chunk: '
                                + repr(receipt.get('sse_shape')))

    def recall_messages(turn_one):
        return [{'role': 'user', 'content': RECALL_PROMPT},
                {'role': 'assistant', 'content': turn_one},
                {'role': 'user', 'content': RECALL_QUESTION}]

    # ---- chat, nonstreaming --------------------------------------------
    receipt = run('chat-stop', chat_payload(TOKYO_PROMPT), 'chat', 'stop')
    demand(receipt, exact(ANSWER), 'wrong visible answer')

    receipt = run('chat-length', chat_payload(LONG_PROMPT, max_tokens=LENGTH_TOKENS),
                  'chat', 'length')
    demand(receipt, finished_length(), 'length finish must consume exactly max_tokens')

    receipt = run('chat-stop-seq', chat_payload(STOP_PROMPT, stop=[STOP_MARKER]),
                  'chat', 'stop')
    demand(receipt, stop_stripped(), 'stop sequence not honored at the match point')

    first = run('chat-multiturn-first', chat_payload(RECALL_PROMPT), 'chat', 'stop')
    demand(first, answered('OK'), 'turn-1 acknowledgement missing')
    receipt = run('chat-multiturn-recall',
                  chat_payload(RECALL_QUESTION, messages=recall_messages(
                      first.get('summary', {}).get('content', ''))),
                  'chat', 'stop')
    demand(receipt, answered(CODEWORD), 'multi-turn code word not recalled')

    # Per-turn thinking toggle inside ONE conversation: turn 1 arms the
    # budget and owns the reasoning channel, turn 2 (same transcript, thinking
    # off) must not inherit that channel and must still recall the codeword.
    # Guards per-turn template re-render and thinking-state isolation — the
    # template-mismatch bug class; multi-turn recall above never toggles.
    toggle_first = run('chat-multiturn-toggle-on',
                       chat_payload(RECALL_PROMPT, thinking=True, budget=THINKING_BUDGET),
                       'chat', 'stop', THINKING_BUDGET)
    demand(toggle_first, answered('OK'), 'turn-1 acknowledgement missing')
    demand(toggle_first, reasoning_present, 'thinking-on turn produced no reasoning channel')
    toggle_second = run('chat-multiturn-toggle-off',
                        chat_payload(RECALL_QUESTION, thinking=False,
                                     messages=recall_messages(
                                         toggle_first.get('summary', {}).get('content', ''))),
                        'chat', 'stop')
    demand(toggle_second, answered(CODEWORD), 'recall across a thinking toggle failed')
    demand(toggle_second,
           lambda summary: not summary['reasoning'],
           'thinking-off turn leaked the prior turn reasoning channel')

    seeded_a = run('chat-seeded-a', chat_payload(JOKE_PROMPT, temperature=SAMPLE_TEMPERATURE),
                   'chat', 'stop')
    seeded_b = run('chat-seeded-b', chat_payload(JOKE_PROMPT, temperature=SAMPLE_TEMPERATURE),
                   'chat', 'stop')
    seed_pair(seeded_a, seeded_b)

    receipt = run('chat-reason-on',
                  chat_payload(CALC_PROMPT, thinking=True, budget=THINKING_BUDGET),
                  'chat', 'stop', THINKING_BUDGET)
    demand(receipt, exact(CALC_ANSWER), 'wrong visible answer')
    demand(receipt, reasoning_present, 'thinking-on produced no reasoning channel')
    demand(receipt, separated, 'reasoning_content and content are not separated')

    receipt = run('chat-reason-off', chat_payload(CALC_PROMPT, thinking=False),
                  'chat', 'stop')
    demand(receipt, exact(CALC_ANSWER), 'wrong visible answer')
    demand(receipt, separated, 'reasoning_content and content are not separated')

    # Qualitative effort field (reasoning.effort / reasoning_effort vocabulary).
    receipt = run('chat-effort-low', chat_payload(CALC_PROMPT, effort='low'),
                  'chat', 'stop')
    demand(receipt, exact(CALC_ANSWER), 'wrong visible answer')
    demand(receipt, separated, 'reasoning_content and content are not separated')

    # ---- chat, streaming (include_usage) --------------------------------
    receipt = run('chat-stream-stop', chat_payload(TOKYO_PROMPT, stream=True),
                  'chat', 'stop')
    demand(receipt, exact(ANSWER), 'wrong visible answer')

    receipt = run('chat-stream-length',
                  chat_payload(LONG_PROMPT, stream=True, max_tokens=LENGTH_TOKENS),
                  'chat', 'length')
    demand(receipt, finished_length(), 'length finish must consume exactly max_tokens')

    receipt = run('chat-stream-stop-seq',
                  chat_payload(STOP_PROMPT, stream=True, stop=[STOP_MARKER]),
                  'chat', 'stop')
    demand(receipt, stop_stripped(), 'stop sequence not honored at the match point')

    receipt = run('chat-stream-multiturn-recall',
                  chat_payload(RECALL_QUESTION, stream=True,
                               messages=recall_messages(
                                   first.get('summary', {}).get('content', ''))),
                  'chat', 'stop')
    demand(receipt, answered(CODEWORD), 'multi-turn code word not recalled')

    stream_seeded_a = run('chat-stream-seeded-a',
                          chat_payload(JOKE_PROMPT, stream=True,
                                       temperature=SAMPLE_TEMPERATURE),
                          'chat', 'stop')
    stream_seeded_b = run('chat-stream-seeded-b',
                          chat_payload(JOKE_PROMPT, stream=True,
                                       temperature=SAMPLE_TEMPERATURE),
                          'chat', 'stop')
    seed_pair(stream_seeded_a, stream_seeded_b)

    receipt = run('chat-stream-reason-on',
                  chat_payload(CALC_PROMPT, stream=True, thinking=True,
                               budget=THINKING_BUDGET),
                  'chat', 'stop', THINKING_BUDGET)
    demand(receipt, exact(CALC_ANSWER), 'wrong visible answer')
    demand(receipt, reasoning_present, 'thinking-on produced no reasoning channel')
    demand(receipt, separated, 'reasoning_content and content are not separated')

    receipt = run('chat-stream-reason-off',
                  chat_payload(CALC_PROMPT, stream=True, thinking=False),
                  'chat', 'stop')
    demand(receipt, exact(CALC_ANSWER), 'wrong visible answer')
    demand(receipt, separated, 'reasoning_content and content are not separated')

    # Token accounting: identical seeded greedy request, stream vs blocking.
    stream_pair(rows['chat-stop'], rows['chat-stream-stop'])
    usage_order(rows['chat-stream-stop'])

    # ---- legacy /v1/completions, nonstreaming ---------------------------
    receipt = run('cmpl-stop', cmpl_payload(TOKYO_LINE), 'completions', 'stop')
    demand(receipt, answered(ANSWER), 'wrong visible answer')

    receipt = run('cmpl-length', cmpl_payload(LONG_LINE, max_tokens=LENGTH_TOKENS),
                  'completions', 'length')
    demand(receipt, finished_length(), 'length finish must consume exactly max_tokens')

    receipt = run('cmpl-stop-seq', cmpl_payload(STOP_LINE, stop=[STOP_MARKER]),
                  'completions', 'stop')
    demand(receipt, stop_stripped(), 'stop sequence not honored at the match point')

    receipt = run('cmpl-multiturn-recall', cmpl_payload(RECALL_LINE),
                  'completions', 'stop')
    demand(receipt, answered(CODEWORD), 'multi-turn code word not recalled')

    cmpl_seeded_a = run('cmpl-seeded-a',
                        cmpl_payload(JOKE_LINE, temperature=SAMPLE_TEMPERATURE),
                        'completions', 'stop')
    cmpl_seeded_b = run('cmpl-seeded-b',
                        cmpl_payload(JOKE_LINE, temperature=SAMPLE_TEMPERATURE),
                        'completions', 'stop')
    seed_pair(cmpl_seeded_a, cmpl_seeded_b)

    # ---- legacy /v1/completions, streaming (include_usage) --------------
    receipt = run('cmpl-stream-stop', cmpl_payload(TOKYO_LINE, stream=True),
                  'completions', 'stop')
    demand(receipt, answered(ANSWER), 'wrong visible answer')

    receipt = run('cmpl-stream-length',
                  cmpl_payload(LONG_LINE, stream=True, max_tokens=LENGTH_TOKENS),
                  'completions', 'length')
    demand(receipt, finished_length(), 'length finish must consume exactly max_tokens')

    receipt = run('cmpl-stream-stop-seq',
                  cmpl_payload(STOP_LINE, stream=True, stop=[STOP_MARKER]),
                  'completions', 'stop')
    demand(receipt, stop_stripped(), 'stop sequence not honored at the match point')

    receipt = run('cmpl-stream-multiturn-recall',
                  cmpl_payload(RECALL_LINE, stream=True), 'completions', 'stop')
    demand(receipt, answered(CODEWORD), 'multi-turn code word not recalled')

    cmpl_stream_seeded_a = run('cmpl-stream-seeded-a',
                               cmpl_payload(JOKE_LINE, stream=True,
                                            temperature=SAMPLE_TEMPERATURE),
                               'completions', 'stop')
    cmpl_stream_seeded_b = run('cmpl-stream-seeded-b',
                               cmpl_payload(JOKE_LINE, stream=True,
                                            temperature=SAMPLE_TEMPERATURE),
                               'completions', 'stop')
    seed_pair(cmpl_stream_seeded_a, cmpl_stream_seeded_b)

    stream_pair(rows['cmpl-stop'], rows['cmpl-stream-stop'])
    usage_order(rows['cmpl-stream-stop'])

    return list(rows.values())

