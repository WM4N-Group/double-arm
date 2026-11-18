#!/usr/bin/env python3
"""Freeze a shard-balanced expert command replay panel, without launching a simulator."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re

RUNS = Path('/mnt/yuanyue/double-arm/runs')
PANEL_NAMESPACE = 'tidy-table-d0-physical-expert-pilot-v1'


def panel(report, audit_hash, count):
    if (report.get('mode') != 'immutable_finalized_official_raw_pool_CPU_audit'
            or report.get('source_grasping_mode') != 'assisted'
            or report.get('dataset_modified') is not False or report.get('split_created') is not False):
        raise ValueError('Need a finalized, unmodified official assisted/raw audit')
    entries = report['episodes']
    if len(entries) != report['eligible_unique_episodes']:
        raise ValueError('Incomplete population')
    grouped = defaultdict(list)
    seen = set()
    for item in entries:
        source = Path(item['source'])
        if (not source.is_relative_to(RUNS) or '..' in source.parts or source.name != 'demo.hdf5'
                or source.parent.parent.name != 'generated' or item.get('source_grasping_mode') != 'assisted'
                or type(item['frames']) is not int or item['frames'] < 1):
            raise ValueError('Expected finalized generated assisted episodes')
        shard = str(source.parents[2]); fingerprint = item['fingerprint']
        if shard not in report['shards'] or not re.fullmatch('[a-f0-9]{64}', fingerprint) or fingerprint in seen:
            raise ValueError('Unverified shard or duplicate/invalid episode fingerprint')
        seen.add(fingerprint)
        grouped[shard].append(item)
    if not 1 <= count <= len(grouped):
        raise ValueError('One episode per distinct shard, no replacement')

    def score(scope, key):
        return hashlib.sha256(f'{PANEL_NAMESPACE}:{audit_hash}:{scope}:{key}'.encode()).hexdigest()

    shards = sorted(grouped, key=lambda shard: (score('shard', shard), shard))[:count]
    return [{'shard': shard, **min(grouped[shard], key=lambda item: (
        score('episode', item['fingerprint']), item['fingerprint']))} for shard in shards]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not args.output.resolve().is_relative_to(RUNS)
            or not args.audit.resolve().is_relative_to(RUNS)):
        raise ValueError('Explicit server CPU and project run directories only')
    audit = args.audit.read_bytes()
    digest = hashlib.sha256(audit).hexdigest()
    report = json.loads(audit)
    selected = panel(report, digest, 8)
    plan = {
        'state': 'FROZEN_CPU_PANEL_NOT_LAUNCHED', 'namespace': PANEL_NAMESPACE,
        'audit': str(args.audit.resolve()), 'audit_sha256': digest,
        'population_episodes': report['eligible_unique_episodes'],
        'selection': 'First 8 SHA-ranked finalized source shards, then first SHA-ranked episode per shard',
        'scope': 'Shard-balanced feasibility panel, NOT an episode-weighted success-rate estimate, training split or heldout policy evaluation',
        'no_post_result_substitution': True, 'training': False, 'gpu_jobs_started': 0, 'policy_calls': 0,
        'future_replay': {'start_source_frame': 0, 'one_state_load': True, 'refresh_steps': 20,
                          'actual_grasping_mode': 'physical', 'unchanged_native_actions': True,
                          'audit_no_assistance': True, 'no_planner_or_policy': True,
                          'phase_outcomes': ['contact', 'lift', 'retention_through_transport', 'release', 'stable_inside_after_release'],
                          'report_separately': ['setup_failure', 'aborted', 'no_contact', 'lost_after_lift', 'no_release', 'completed']},
        'episodes': selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(plan, ensure_ascii=False, indent=2) + '\n').encode()
    with args.output.open('xb') as stream:
        stream.write(payload)
    print(args.output, hashlib.sha256(payload).hexdigest())
    for ep in selected:
        print(Path(ep['shard']).name, ep.get('seed'), ep['source_demo'], ep['frames'])


if __name__ == '__main__':
    main()
