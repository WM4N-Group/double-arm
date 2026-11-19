#!/usr/bin/env python3
"""Attribute frozen X7 rear-return minima; simulator IDs are diagnostics only."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pybullet as p

from audit_x7_chassis_range_proxy import gaps_from_fractions
from audit_x7_dock_sensor import BIAS, GUARD_SCAN_M, PERIOD, WAVE
from audit_x7_handle_rear_budget import (
    ArmIK, FROZEN_CHECKPOINT, GRIP_Z, HANDLE_HALF_SPAN, PARK_Y, PANEL,
    HandleBudgetWorld, digest, run_case,
)
from audit_x7_payload_range_gate import MODEL_ERROR_M
from audit_x7_upper_rear_coverage import RAY_BATCH


class ReturnAuditWorld(HandleBudgetWorld):
    audit_window = 240  # Physics substeps from the brake latch.

    def __init__(self, *args):
        self.diagnostics = {}
        self.capture = None
        super().__init__(*args)

    def acquire_packet(self, k, truth_rear):
        if self.latch_k is None or k >= self.latch_k + self.audit_window:
            return super().acquire_packet(k, truth_rear)
        self.capture = []
        try:
            raw, pose = super().acquire_packet(k, truth_rear)
            names = ('fixed_cad', 'main_box', 'handles') if self.handles else ('fixed_cad', 'main_box')
            if len(self.capture) != len(names):
                raise RuntimeError('Unexpected rear-channel order')
            wave = float(np.sin(.77 * (k // PERIOD) + 2.24) * WAVE + BIAS)
            channels = dict(zip(names, self.capture))
            for key, channel in channels.items():
                geometric = channel['metric_minimum_m']
                noisy = None if geometric is None else min(geometric + wave, GUARD_SCAN_M)
                channel['noise_and_cap_m'] = noisy
                channel['calibrated_m'] = None if noisy is None else noisy - (0. if key == 'fixed_cad' else MODEL_ERROR_M)
            if raw[2] >= 0 and not np.isclose(min(c['calibrated_m'] for c in channels.values()), raw[2], atol=1e-12, rtol=0):
                raise RuntimeError('Diagnostic fusion differs from control packet')
            self.diagnostics[k] = dict(channels=channels, sampled_odometry=pose.tolist(), fused_raw_m=float(raw[2]))
            return raw, pose
        finally:
            self.capture = None

    def rear_fractions(self, a, b):
        if self.capture is None:
            return super().rear_fractions(a, b)
        # Match the production batches/fractions exactly. Archive identity
        # separately; not even self returns can change the controller input.
        entries = []
        minima = {}
        for k in range(0, len(a), RAY_BATCH):
            start, end = a[k:k + RAY_BATCH], b[k:k + RAY_BATCH]
            batch = p.rayTestBatch(start.tolist(), end.tolist(), numThreads=1)
            if len(batch) != len(start):
                entries.extend([np.nan] * len(start))
                continue
            entries.extend(row[2] for row in batch)
            for i, (body, link, frac, hit, _) in enumerate(batch):
                if body < 0 or not 0. <= frac < 1.:
                    continue
                key = int(body), int(link)
                if key not in minima or frac < minima[key]['fraction']:
                    minima[key] = dict(fraction=float(frac), ray_index=k + i, origin_m=start[i].tolist(), hit_m=list(hit))
        f = np.asarray(entries)
        records = []
        for (body, link), row in minima.items():
            kind = ('robot' if body == self.robot else 'payload' if body == self.tote else
                    'actor' if body == self.obstacle else 'floor' if body == self.floor else
                    f'table_{self.tables.index(body)}' if body in self.tables else 'unclassified')
            link_name = p.getJointInfo(body, link)[12].decode() if link >= 0 else None
            records.append(row | dict(body=body, link=link, link_name=link_name, kind=kind,
                                      range_m=float(gaps_from_fractions(row['fraction']))))
        known = np.isfinite(f).all() and (f >= 0.).all() and (f <= 1.).all()
        self.capture.append(dict(metric_minimum_m=float(gaps_from_fractions(f).min()) if known else None,
                                 hits_by_body_link=sorted(records, key=lambda row: row['range_m'])))
        return f


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pilot', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--cases', nargs='+', choices=PANEL, default=['clear-with-handles'])
    args = ap.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES') != '' or len(set(args.cases)) != len(args.cases):
        ap.error('remote CPU only; distinct cases')
    model = args.pilot / 'anchored-hybrid.npz'
    if digest(model) != FROZEN_CHECKPOINT:
        raise ValueError('Unexpected frozen checkpoint')
    args.out.mkdir(parents=True)
    with np.load(model, allow_pickle=False) as stored:
        weights = {k: stored[k] for k in stored.files}
    asset = args.pilot / 'model/x7_proxy_gripper.urdf'
    arms = [ArmIK(asset, i, np.array([s * HANDLE_HALF_SPAN, PARK_Y + .015, GRIP_Z + .105]))
            for i, s in enumerate((-1, 1))]
    p.connect(p.DIRECT)
    try:
        for name in args.cases:
            worlds = []

            def make_world(*a):
                world = ReturnAuditWorld(*a)
                worlds.append(world)
                return world

            result = run_case(weights, asset, arms, name, args.out, world_type=make_world)
            report = dict(checkpoint_sha256=FROZEN_CHECKPOINT, result=result, packets=worlds[0].diagnostics,
                          diagnostic_ids_used_for_control=False)
            (args.out / (name + '.json')).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(name, result['rear_events'], flush=True)
    finally:
        p.disconnect()


if __name__ == '__main__':
    main()
