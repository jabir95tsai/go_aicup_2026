"""Exercise the single-job workflow without allocating GPUs or invoking Slurm."""
import json
import os
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path('slurm/SE3_H_C_B2/train.slurm').resolve()


@pytest.mark.parametrize('gate,failed', [(True, False), (False, False), (True, True)])
def test_single_job_workflow(tmp_path, gate, failed):
    fake_python = tmp_path / 'python'
    fake_python.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
root = pathlib.Path(os.environ['TEST_OUTPUT'])
if args[0] == '-c':
    if 'continue_development' in args[1]:
        value = json.loads(pathlib.Path(args[2]).read_text())['continue_development']
        print('continue' if value else 'stop')
    else:
        print(root)
    raise SystemExit(0)
command = args[args.index('--config') + 2:]
with open(os.environ['TEST_LOG'], 'a') as handle:
    handle.write(json.dumps(command) + '\\n')
if command[:5] == ['evaluate', '--stage', 'dev', '--role', 'select']:
    result = root / 'reports/dev_select/result.json'
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps({'continue_development': os.environ['TEST_GATE'] == 'yes'}))
if (os.environ['TEST_FAIL'] == 'yes' and command[:5] ==
        ['extract', '--component', 'se3', '--scope', 'dev'] and command[6] == '3'):
    raise SystemExit(7)
''')
    fake_python.chmod(0o755)
    env = {**os.environ, 'PYTHON_BIN': str(fake_python), 'CONFIG': 'unused.json',
           'CUDA_VISIBLE_DEVICES': ','.join(map(str, range(8))), 'SLURM_JOB_ID': '123',
           'TEST_OUTPUT': str(tmp_path / 'run'), 'TEST_LOG': str(tmp_path / 'calls.jsonl'),
           'TEST_GATE': 'yes' if gate else 'no', 'TEST_FAIL': 'yes' if failed else 'no'}
    result = subprocess.run(['bash', str(SCRIPT)], env=env, text=True, capture_output=True, timeout=30)
    calls = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    assert calls[0] == ['prepare']
    dev = [c for c in calls if c[:5] == ['extract', '--component', 'se3', '--scope', 'dev']]
    assert sorted(int(c[6]) for c in dev) == list(range(8))
    if failed:
        assert result.returncode != 0 and 'Feature extraction failed' in result.stderr
        assert not any(c[0] == 'train' for c in calls)
    elif not gate:
        assert result.returncode == 0 and 'Development gate stopped' in result.stdout
        assert not any(c[:3] == ['train', '--stage', 'final'] for c in calls)
    else:
        assert result.returncode == 0, result.stderr
        assert calls[-1] == ['evaluate', '--stage', 'final', '--role', 'newgames']
        assert [c for c in calls if c[0] == 'train'] == [
            ['train', '--stage', stage, '--arm', arm]
            for stage in ('dev', 'final') for arm in ('a', 'se3')]
        for component, scope in [('se3', 'fit'), ('h', 'newgames'), ('c', 'newgames')]:
            shards = [c for c in calls if c[:5] == ['extract', '--component', component, '--scope', scope]]
            assert sorted(int(c[6]) for c in shards) == list(range(8))
