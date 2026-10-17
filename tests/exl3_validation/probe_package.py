# SPDX-License-Identifier: AGPL-3.0-only
"""Assemble the evidence-root probe overlay for the runtime-mechanism probe.

Implements the registration's owed item "Evidence-root overlay package
assembly (sitecustomize lineage copy + hash manifest) wiring the frozen
harness modules; package hash recorded":

- the frozen harness trio is copied from the repo and verified against the
  hashes recorded in the probe registration (the SSOT for that freeze); a
  mismatch fails, because a changed file needs a re-freeze first;
- the lineage overlay is copied verbatim with its ``sitecustomize.py``
  renamed to ``lineage_sitecustomize.py`` so the probe's own
  ``sitecustomize.py`` can chain with it at import time;
- ``probe_wiring.py`` (the observer glue) and ``reference_norm_binding.py``
  (the frozen trio's imported dependency, bound here) join the package;
- ``package-binding.json`` records every file hash plus the registration and
  lineage bindings; its SHA256 is the recorded package hash.

Usage:
    probe_package.py --config package.json
"""
import argparse
import json
from pathlib import Path
import re
import shutil

from image_cycle import sha, read, require, text

CONFIG_SCHEMA = 'atlas-probe-package-config-v1'
PACKAGE_SCHEMA = 'atlas-probe-hook-package-v1'
FROZEN_TRIO = {'linear_launch.py', 'reference_router_capture.py',
               'reference_collective_trace.py'}
GLUE_MEMBERS = {'probe_wiring.py', 'reference_norm_binding.py'}
TRIO_ROW = r'^\| `tests/exl3_validation/([^`]+)` \| `([0-9a-f]{64})` \|$'


def frozen_trio(registration_path):
    """Parse the registration's harness-freeze table (its SSOT)."""
    rows = dict(re.findall(TRIO_ROW, Path(registration_path).read_text(), re.M))
    require(set(rows) == FROZEN_TRIO,
            'registration must freeze exactly the expected harness trio, found: '
            + str(sorted(rows)))
    return rows


def _copy_lineage(source, target):
    for path in sorted(Path(source).rglob('*')):
        if {'__pycache__', '.pytest_cache'} & set(path.parts) or path.is_dir():
            continue
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)


def assemble(config_path):
    config = read(config_path)
    require(set(config) == {'schema', 'out', 'evidence_root', 'registration',
                            'lineage_overlay'}
            and config['schema'] == CONFIG_SCHEMA,
            'explicit probe package configuration keys/schema required')
    out = Path(config['out'])
    evidence_root = Path(config['evidence_root'])
    require(out.is_absolute() and evidence_root.is_absolute()
            and out.is_relative_to(evidence_root),
            'package output must be an absolute path inside the evidence root')
    require(not out.exists(), 'package output must be a fresh directory')
    registration = config['registration']
    require(set(registration) == {'path', 'sha256'},
            'explicit registration binding required')
    reg_path = Path(registration['path'])
    require(reg_path.is_file() and sha(reg_path) == registration['sha256'],
            'registration binding differs')
    lineage = config['lineage_overlay']
    require(set(lineage) == {'path'}, 'explicit lineage overlay path required')
    lineage_path = Path(lineage['path'])
    require(lineage_path.is_absolute() and lineage_path.is_dir(),
            'lineage overlay directory missing')

    trio = frozen_trio(reg_path)
    here = Path(__file__).resolve().parent
    out.mkdir(parents=True)
    _copy_lineage(lineage_path, out)
    require((out / 'sitecustomize.py').is_file(),
            'lineage overlay must carry its sitecustomize.py')
    (out / 'sitecustomize.py').rename(out / 'lineage_sitecustomize.py')
    lineage_binding = out / 'package-binding.json'
    lineage_binding_sha = None
    if lineage_binding.is_file():
        lineage_binding_sha = sha(lineage_binding)
        lineage_binding.rename(out / 'lineage-package-binding.json')
    for name in sorted(FROZEN_TRIO | GLUE_MEMBERS):
        source = here / name
        require(source.is_file(), 'repo member missing: ' + name)
        if name in FROZEN_TRIO:
            require(sha(source) == trio[name],
                    'frozen harness differs from registration (re-freeze '
                    'required first): ' + name)
        shutil.copyfile(source, out / name)
    shutil.copyfile(here / 'probe_sitecustomize.py', out / 'sitecustomize.py')
    binding = {'schema': PACKAGE_SCHEMA,
               'registration': {'path': str(reg_path),
                                'sha256': registration['sha256'],
                                'frozen_trio': trio},
               'lineage': {'path': str(lineage_path),
                           'package_binding_sha256': lineage_binding_sha},
               'file_sha256': {}}
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name != 'package-binding.json':
            binding['file_sha256'][str(path.relative_to(out))] = sha(path)
    with (out / 'package-binding.json').open('x') as target:
        target.write(text(binding))
    return {'out': str(out), 'package_sha256': sha(out / 'package-binding.json'),
            'files': len(binding['file_sha256'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    print(text(assemble(args.config)), end='')


if __name__ == '__main__':
    main()
