# SPDX-License-Identifier: AGPL-3.0-only
"""Lifecycle ordering and failure containment around production collector seam."""
from contextlib import contextmanager
import json
import pytest
from c2_ab_driver import run_ab
from test_c2_ab_driver import FakeCollector, _entries, _plan


def test_lifecycle_wraps_all_repeats_and_exposes_rendered_commands(tmp_path):
    plan = _plan(tmp_path)
    events = []
    @contextmanager
    def lifecycle(**context):
        root = context['run_root']
        assert set(context['create_commands']) == {'0', '1'}
        assert (root / 'create-command-rank0.json').is_file()
        assert context['records_dir'].is_dir()
        events.append(('start', root.name))
        try:
            yield
        finally:
            events.append(('stop', root.name))
    run_ab(plan, FakeCollector(_entries(plan)), lifecycle=lifecycle)
    assert len(events) == 8
    assert all(events[i][0] == 'start' and events[i + 1][0] == 'stop'
               and events[i][1] == events[i + 1][1] for i in range(0, 8, 2))


def test_collector_failure_still_exits_lifecycle_and_preserves_attempt(tmp_path):
    plan = _plan(tmp_path)
    stopped = []
    @contextmanager
    def lifecycle(**context):
        try:
            yield
        finally:
            stopped.append(context['run_root'].name)
    with pytest.raises(RuntimeError, match='collector failed'):
        run_ab(plan, FakeCollector(_entries(plan), fail_call=0), lifecycle=lifecycle)
    assert stopped == ['arm-a-repeat-00']
    assert json.loads((plan['evidence_root'] / plan['attempt'] / 'failure.json').read_text())


def test_default_driver_cli_refuses_unlaunched_live_collection():
    from c2_ab_driver import main
    with pytest.raises(ValueError, match='explicit remote config'):
        main()
