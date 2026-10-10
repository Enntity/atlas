# SPDX-License-Identifier: AGPL-3.0-only
"""Freeze text cases using a local checkpoint tokenizer, with no network access."""
import argparse
import hashlib
import json
from pathlib import Path

from parity_protocol import validate_cases


def freeze(tokenizer, cases, add_special_tokens):
    result = []
    for case in cases:
        if set(case) != {"id", "prompt", "continuation"} or not all(isinstance(case[k], str) for k in case):
            raise ValueError("text cases require exactly string id/prompt/continuation")
        result.append({"id": case["id"],
                       "prompt_ids": tokenizer.encode(case["prompt"], add_special_tokens=add_special_tokens),
                       "continuation_ids": tokenizer.encode(case["continuation"], add_special_tokens=False)})
    return validate_cases(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint", "input", "out", "add-special-tokens"):
        parser.add_argument("--" + flag, required=True, **({"choices": ["yes", "no"]} if flag == "add-special-tokens" else {}))
    args = parser.parse_args()
    from transformers import AutoTokenizer
    import transformers
    checkpoint, destination = Path(args.checkpoint), Path(args.out)
    if not checkpoint.is_dir() or destination.exists() or Path(str(destination) + ".tokenizer.json").exists():
        raise ValueError("checkpoint must be local and output destinations must be new")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    cases = freeze(tokenizer, json.loads(Path(args.input).read_text()), args.add_special_tokens == "yes")
    sources = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
               "vocab.json", "merges.txt", "vocab.txt", "tokenizer.model", "chat_template.jinja"]
    hashes = {name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
              for name in sources if (checkpoint / name).is_file()}
    if not hashes:
        raise ValueError("cannot fingerprint local tokenizer sources")
    destination.write_text(json.dumps(cases, indent=2))
    Path(str(destination) + ".tokenizer.json").write_text(json.dumps({"tokenizer_sha256": hashes,
        "transformers_version": transformers.__version__, "prompt_add_special_tokens": args.add_special_tokens,
        "continuation_add_special_tokens": False, "text_cases_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
