#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Reject known native runtime faults and accidentally enabled MTP probes.

This bounded log gate complements correctness, memory and process-exit checks;
absence of these markers alone does not establish healthy execution.
"""
import argparse
import re
from pathlib import Path

ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
FAULT = re.compile(
    r'^(?:\d{4}-\d{2}-\d{2}[T ][\d:.]+Z?\s+)?ERROR(?:\s|:|$)|'
    r'unhealthy|panic|illegal memory|out of memory|'
    r'HIDDEN_TRACE|K5_LEDGER|CUDA_ERROR_|CudaError\(|NCCL error|device-side assert',
    re.IGNORECASE,
)


def fault_lines(lines):
    return [line for line in lines if FAULT.search(ANSI.sub('', line))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    args = parser.parse_args()
    matches = fault_lines(args.log.read_text(encoding='utf-8').splitlines())
    for line in matches:
        print(line)
    raise SystemExit(1 if matches else 0)


if __name__ == '__main__':
    main()
