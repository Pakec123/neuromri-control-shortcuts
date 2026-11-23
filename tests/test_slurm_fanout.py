"""Resource regression: each concurrent srun gets one GPU and finite memory."""
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


def test_every_background_srun_has_memory_and_one_gpu():
    found = []
    for path in Path('slurm').glob('*'):
        if not path.is_file():
            continue
        text = re.sub(r'\\\n', ' ', path.read_text())
        for line in text.splitlines():
            if re.search(r'^\s*srun\s', line) and re.search(r'&\s*$', line):
                found.append(path.name)
                assert '--exclusive' in line, path
                assert re.search(r'--gres(?:=|\s+)gpu:1(?:\s|$)', line), path
                assert re.search(r'--mem(?:=|\s+)"?\$GPU_STEP_MEM_MB"?(?:\s|$)', line), path
                assert 'source slurm/four_step_memory.sh' in text, path
    assert set(found) == {'four_gpu.sbatch',
                          'extract.sbatch', 'external_extract.sbatch'}


@pytest.mark.skipif(not shutil.which('bash'), reason='Bash unavailable')
@pytest.mark.parametrize('memory,override,success', [('131072', '', True), ('32768', '', True),
    ('131072', '32769', False), ('0', '', False), ('', '', False), ('128G', '', False), ('131072', '0', False)])
def test_step_memory_allocation(memory, override, success):
    env = dict(os.environ, SLURM_MEM_PER_NODE=memory, GPU_STEP_MEM_MB=override)
    result = subprocess.run([shutil.which('bash'), '-c', 'source slurm/four_step_memory.sh; echo "$GPU_STEP_MEM_MB"'],
                            env=env, capture_output=True, text=True)
    assert (result.returncode == 0) == success, result.stderr
    if success:
        assert int(result.stdout) == int(memory) // 4
