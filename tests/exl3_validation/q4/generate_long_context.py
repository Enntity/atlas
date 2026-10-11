#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic needle-in-haystack generator for Q4 long-context cases.

Each haystack is varied narrative filler (sentences assembled from fixed word
lists by `random.Random(seed)`, so the same arguments always give the same
bytes on every CPython >= 3.2) interleaved with maintenance-log records of
the form "locker N was assigned the access code ABC-1234". Exactly one record
names the asked locker (the needle, placed at a chosen depth); the others are
distractors whose codes the answer must NOT contain.

Size is set in characters. Token counts were measured offline with the
checkpoint's own tokenizer (MikeRoz/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3
tokenizer.json, sha256 19e77364...a82d); see CHARS_PER_TOKEN_NOTE in
build_corpus.py for the measured counts of the committed cases. The runner
records usage.prompt_tokens for every trial so a live run verifies them.
"""
import random
import sys

SUBJECTS = ("The night porter", "A delivery driver", "The site engineer", "Two visiting auditors",
            "The facilities manager", "A contractor from the north depot", "The safety officer",
            "An apprentice electrician", "The cleaning crew", "The building inspector")
VERBS = ("inspected", "repainted", "photographed", "measured", "swept", "relabelled",
         "rewired", "catalogued", "tested", "reorganised")
OBJECTS = ("the loading bay", "the east stairwell", "the boiler room", "the roof access hatch",
           "the bicycle racks", "the fire doors on level two", "the staff kitchen",
           "the archive shelving", "the emergency lighting", "the car park barrier")
CODAS = ("before the morning shift began", "while it rained steadily outside",
         "and logged the work in the paper ledger", "after the power was restored",
         "without finding anything unusual", "and reported a loose hinge",
         "despite the noise from the street", "as part of the quarterly routine",
         "and left a note for the day team", "under the flickering corridor lights")
WEATHER = ("The weather stayed mild.", "A cold wind came off the river.",
           "Fog lingered until noon.", "The afternoon was bright and dry.",
           "Light snow fell briefly.", "It was humid and still.")
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"


def _sentence(rng):
    return f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)} {rng.choice(CODAS)}."


def _paragraph(rng, day):
    body = " ".join(_sentence(rng) for _ in range(rng.randint(3, 5)))
    return f"Day {day}. {rng.choice(WEATHER)} {body}"


def _code(rng, used):
    while True:
        code = "".join(rng.choice(LETTERS) for _ in range(3)) + "-" + str(rng.randint(1000, 9999))
        if code not in used:
            used.add(code)
            return code


def _record(locker, code):
    return f"Maintenance log entry: locker {locker} was assigned the access code {code}."


def generate(seed, chars, depth, distractors=6):
    """Return (prompt, needle_code, distractor_codes, asked_locker).

    `chars` is the minimum filler length before records are added; `depth`
    in [0, 1] places the needle record that fraction of the way through the
    haystack paragraphs.
    """
    if not 0.0 <= depth <= 1.0 or chars < 1000 or distractors < 1:
        raise ValueError("depth in [0,1], chars >= 1000 and >= 1 distractor required")
    rng = random.Random(seed)
    used = set()
    lockers = rng.sample(range(100, 1000), distractors + 1)
    asked = lockers[0]
    needle = _code(rng, used)
    others = [(locker, _code(rng, used)) for locker in lockers[1:]]
    paragraphs, total, day = [], 0, 1
    while total < chars:
        paragraph = _paragraph(rng, day)
        paragraphs.append(paragraph)
        total += len(paragraph) + 2
        day += 1
    slots = len(paragraphs)
    if slots <= distractors:
        raise ValueError("haystack too short for the requested distractors")
    needle_at = int(round(depth * (slots - 1)))
    # Distractor records go in other paragraphs, never beside the needle.
    free = [i for i in range(slots) if i != needle_at]
    for (locker, code), at in zip(others, sorted(rng.sample(free, len(others)))):
        paragraphs[at] += " " + _record(locker, code)
    paragraphs[needle_at] += " " + _record(asked, needle)
    question = (f"\n\nQuestion: According to the log above, what is the access code for locker "
                f"{asked}? Reply with only the code.")
    prompt = "Read the following building log carefully.\n\n" + "\n\n".join(paragraphs) + question
    return prompt, needle, [code for _, code in others], asked


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    chars = int(sys.argv[2]) if len(sys.argv) > 2 else 13000
    depth = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    print(generate(seed, chars, depth)[0])
