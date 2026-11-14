#!/usr/bin/env python3
"""SERVER ONLY: distinguish a clean generator exit from usable successful data."""
import argparse
import json
from pathlib import Path


def classify(exit_status, successes, required):
    if exit_status != 0:
        return 'interrupted_or_failed'
    return 'passed' if successes >= required else 'insufficient_successful_demos'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--exit-status', type=int, required=True)
    p.add_argument('--required', type=int, required=True)
    a = p.parse_args()
    if a.required < 1: p.error('--required must be positive')
    paths = list(a.run_dir.glob('generated/*/important_stats.json'))
    report = {'generator_exit_status': a.exit_status, 'required_successes': a.required,
              'training': 'not_started', 'evaluation': 'not_started'}
    successes = 0
    if len(paths) == 1:
        stats = json.loads(paths[0].read_text())
        report.update(stats_file=str(paths[0]), num_attempts=stats.get('num_attempts'),
                      num_success=stats.get('num_success', 0), num_failures=stats.get('num_failures'))
        dataset = paths[0].parent / 'demo.hdf5'
        if dataset.is_file():
            import h5py
            with h5py.File(dataset, 'r') as f:
                demos = [k for k in f['data'] if k.startswith('demo_')]
                valid = [k for k in demos if all(n in f['data'][k] for n in ('actions', 'states', 'obs'))
                         and len(f['data'][k]['actions']) > 0
                         and len(f['data'][k]['actions']) == len(f['data'][k]['states'])]
                successes = min(int(stats.get('num_success', 0)), len(valid))
            report.update(dataset=str(dataset), verified_successful_episodes=successes)
    else:
        report['error'] = f'Expected one final stats file, found {len(paths)}'
    report['state'] = classify(a.exit_status, successes, a.required)
    path = a.run_dir / 'generation-result.json'
    tmp = path.with_suffix('.json.tmp'); tmp.write_text(json.dumps(report, indent=2)); tmp.replace(path)
    print(json.dumps(report, indent=2), flush=True)
    if report['state'] != 'passed': raise SystemExit(6)


if __name__ == '__main__': main()
