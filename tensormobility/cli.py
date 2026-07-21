"""Minimal TensorMobility CLI (console script `tm`).

    tm demo list            # workflows + pipelines from workflow.yml
    tm demo run <case>      # run a case script from cases/
    tm version
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = {p.stem.replace('run_', ''): p
         for p in (ROOT / 'cases').glob('run_*.py')} \
    if (ROOT / 'cases').exists() else {}


def _load_workflows():
    import yaml
    wf = ROOT / 'workflow.yml'
    if not wf.exists():
        return {}
    return yaml.safe_load(wf.read_text(encoding='utf-8'))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ('-h', '--help'):
        print(__doc__)
        return 0
    if args[0] == 'version':
        from tensormobility import __version__
        print(__version__)
        return 0
    if args[0] == 'demo':
        if len(args) < 2 or args[1] == 'list':
            d = _load_workflows()
            for wid, w in (d.get('workflows') or {}).items():
                print(f"{wid}  {w.get('name'):<10} {w.get('summary','')}")
            print('pipelines:', ', '.join((d.get('pipelines') or {})))
            print('cases:', ', '.join(sorted(CASES)) or '(none packaged)')
            return 0
        if args[1] == 'run' and len(args) >= 3:
            case = args[2]
            if case not in CASES:
                print(f'unknown case {case!r}; options: {sorted(CASES)}')
                return 2
            return subprocess.call([sys.executable, str(CASES[case])])
    print(__doc__)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
