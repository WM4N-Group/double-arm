"""Write a small atomic stage report; no GPU imports or credentials."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True, type=Path)
parser.add_argument('--stage', required=True)
parser.add_argument('--state', choices=['running', 'passed', 'failed'], required=True)
parser.add_argument('--exit-code', type=int, default=0)
args = parser.parse_args()
data = {'stage': args.stage, 'state': args.state, 'exit_code': args.exit_code,
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'pid': os.getppid(), 'policy_training': 'not_run'}
args.file.parent.mkdir(parents=True, exist_ok=True)
tmp = args.file.with_suffix('.json.tmp')
tmp.write_text(json.dumps(data, indent=2))
tmp.replace(args.file)
print(json.dumps(data), flush=True)
