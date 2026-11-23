"""Execute the unchanged P1 aggregate commands locally against a passing gate."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.p1_postprocessing import load_gate, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['aggregate', 'stats'], required=True)
    args = parser.parse_args()
    gate_path = args.gate.resolve()
    gate = load_gate(gate_path)
    output = args.output.resolve()
    if output.is_relative_to(Path(gate['input_root'])):
        raise ValueError('Output must be outside read-only input export')
    packages = {p: importlib.metadata.version(p) for p in
                ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'PyYAML']}
    pins = dict(line.strip().split('==') for line in (ROOT/'env/requirements.lock').read_text().splitlines()
                if '==' in line and not line.startswith('#'))
    if any(value != pins[name] for name, value in packages.items()):
        raise ValueError('Numerical reanalysis packages differ from frozen pins: ' + str(packages))
    from tools.p1_analysis_commands import commands
    script = commands(Path(gate['input_root']), output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'commands.sh').write_text(script, encoding='utf-8')
    started = datetime.now(timezone.utc).isoformat()
    tick = time.perf_counter()
    executed = []
    with (output/(args.phase+'.log')).open('w', encoding='utf-8') as log:
        for line in script.splitlines():
            if not line.startswith('python '):
                continue
            argv = shlex.split(line)[1:]
            is_stats = argv[1] == 'neuromri.p1_stats'
            if is_stats != (args.phase == 'stats'):
                continue
            if not is_stats:
                argv += ['--source-integrity-gate', str(gate_path)]
            sys.argv = [argv[1], *argv[2:]]
            with redirect_stdout(log):
                if is_stats:
                    from neuromri.p1_stats import main as run
                else:
                    from neuromri.aggregate import main as run
                run()
            executed.append(argv)
            print(f'{args.phase}: completed command {len(executed)}', flush=True)
    result = dict(phase=args.phase, started_utc=started, ended_utc=datetime.now(timezone.utc).isoformat(),
                  seconds=time.perf_counter()-tick, packages=packages, python=sys.version,
                  executable=sys.executable, platform=platform.platform(), source_gate_sha256=sha(gate_path),
                  commands=executed, outputs={str(p.relative_to(output)):sha(p) for p in output.rglob('*')
                                             if p.is_file() and p.name != 'runtime_'+args.phase+'.json'})
    (output/('runtime_'+args.phase+'.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
