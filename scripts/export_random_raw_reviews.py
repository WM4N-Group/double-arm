#!/usr/bin/env python3
"""SERVER CPU ONLY: random, non-curated review of recorded official raw RGB.

Never launches a simulator/model or changes data. Selection is committed before
reading video pixels; a failure stops export instead of drawing a replacement.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess

ROOT = Path('/mnt/yuanyue/double-arm')
CAMERAS = ('head_image', 'left_wrist_image', 'right_wrist_image')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def select_entries(report, count, seed):
    if (report.get('mode') != 'immutable_finalized_official_raw_pool_CPU_audit'
            or report.get('source_grasping_mode') != 'assisted'):
        raise ValueError('Finalized official assisted audit required')
    entries = report['episodes']
    if len(entries) != report['eligible_unique_episodes']:
        raise ValueError('Audit population count mismatch')
    if len({e['fingerprint'] for e in entries}) != len(entries):
        raise ValueError('Duplicate fingerprints in sampling population')
    if not 1 <= count <= len(entries):
        raise ValueError('Sample count outside population')
    if any(e.get('source_grasping_mode') != 'assisted' for e in entries):
        raise ValueError('Mixed or unknown source modes')
    # Canonical ordering makes the sample reproducible even if JSON order changes.
    population = sorted(entries, key=lambda e: e['fingerprint'])
    return random.Random(seed).sample(population, count)


def recorded_fps(env):
    value = float(env['env_kwargs']['env']['action_frequency'])
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid recorded action frequency')
    return value


def export_one(entry, out, number, total, audit_sha, selection_sha):
    import h5py
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from momagen_pi0_schema import native_actions, rgb_image

    source = Path(entry['source']).resolve()
    if not source.is_relative_to(ROOT / 'runs') or source.name != 'demo.hdf5':
        raise ValueError('Expected remote finalized source')
    verdict = json.loads((source.parents[2] / 'generation-result.json').read_text())
    if (verdict.get('state') != 'passed' or verdict.get('generator_exit_status') != 0
            or Path(verdict['dataset']).resolve() != source):
        raise ValueError('Finalized source gate failed')
    out.mkdir()
    video = out / 'demo.mp4'
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    except OSError:
        font = ImageFont.load_default()
    rgb_hashes = {name: hashlib.sha256() for name in CAMERAS}
    with h5py.File(source, 'r') as file:
        env = json.loads(file['data'].attrs['env_args'])
        if (env['env_name'] != 'r1_tidy_table_D0'
                or env['env_kwargs']['robots'][0]['grasping_mode'] != 'assisted'):
            raise ValueError('Unexpected environment or source grasp mode')
        fps = recorded_fps(env)
        g = file['data'][entry['source_demo']]
        actions = native_actions(g['actions'][()])
        first_state = g['obs/prop_state'][0]
        fingerprint = hashlib.sha256(actions.tobytes() + first_state.tobytes()).hexdigest()
        n = len(actions)
        if (fingerprint != entry['fingerprint'] or n != entry['frames']
                or bool(g.attrs.get('partial', False))):
            raise ValueError('Episode identity/length/completeness changed')
        datasets = {k: g['obs'][entry['cameras'][k]] for k in CAMERAS}
        if any(d.shape != (n, 256, 256, 4) or d.dtype != np.uint8 for d in datasets.values()):
            raise ValueError('Unexpected recorded RGB layout')
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-threads', '2',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '960x408', '-r', str(fps),
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2',
               '-preset', 'fast', '-crf', '25', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(video)]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
            try:
                for start in range(0, n, 16):
                    batch = {k: d[start:start + 16] for k, d in datasets.items()}
                    for k, values in batch.items():
                        rgb_hashes[k].update(values.tobytes())
                    for offset in range(len(batch[CAMERAS[0]])):
                        index = start + offset
                        canvas = Image.new('RGB', (960, 408), '#151515')
                        draw = ImageDraw.Draw(canvas)
                        draw.text((8, 5), f'RANDOM RAW SAMPLE {number}/{total} | OFFICIAL SOURCE | ASSISTED | NOT LEARNED PI0', font=font, fill='yellow')
                        draw.text((8, 26), f'Seed {entry["seed"]} | {entry["source_demo"]} | frame {index + 1}/{n} | {index / fps:.2f}s SIM TIME', font=font, fill='white')
                        for col, key in enumerate(CAMERAS):
                            image = Image.fromarray(rgb_image(batch[key][offset])).resize((320, 320))
                            canvas.paste(image, (col * 320, 66))
                            draw.text((col * 320 + 8, 47), key.upper(), font=font, fill='white')
                        draw.text((8, 389), 'RECORDED DATA RGB (256px, enlarged) | NO SIM REPLAY | NOT PHYSICAL GRASP VALIDATION', font=font, fill='yellow')
                        encoder.stdin.write(canvas.tobytes())
                    print(f'SAMPLE {number} RGB {min(start + 16, n)}/{n}', flush=True) if start % 256 == 0 else None
            finally:
                encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('Video encoder failed')
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,nb_frames,r_frame_rate:format=duration,size', '-of', 'json', str(video)]))
    stream = probe['streams'][0]
    if (int(stream['nb_frames']) != n or abs(float(probe['format']['duration']) - n / fps) > .06):
        raise ValueError('Exported frame count/duration mismatch')
    subprocess.run(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(video), '-f', 'null', '-'], check=True)
    sheet = Image.new('RGB', (960, 204 * 3))
    seconds = [((n - 1) / fps) * f for f in (0, .2, .4, .6, .8, 1)]
    for i, second in enumerate(seconds):
        png = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '2', '-ss', str(second),
            '-i', str(video), '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-threads', '1', '-'])
        frame = Image.open(io.BytesIO(png)).convert('RGB').resize((480, 204))
        sheet.paste(frame, ((i % 2) * 480, (i // 2) * 204))
    sheet.save(out / 'contact-sheet.jpg', quality=90)
    review = {'role': 'random_OFFICIAL_ASSISTED_recorded_data_NOT_learned_policy',
        'source': str(source), 'source_demo': entry['source_demo'], 'seed': entry['seed'],
        'episode_fingerprint': fingerprint, 'audit_sha256': audit_sha,
        'selection_sha256': selection_sha, 'source_grasping_mode': 'assisted',
        'frames': n, 'fps': fps, 'duration_sim_seconds': n / fps,
        'rgb_source_shapes': {k: [n, 256, 256, 4] for k in CAMERAS},
        'recorded_rgba_sha256': {k: h.hexdigest() for k, h in rgb_hashes.items()},
        'video_sha256': sha256(video), 'video_bytes': video.stat().st_size,
        'contact_sheet_sha256': sha256(out / 'contact-sheet.jpg'),
        'server_full_decode_passed': True, 'all_source_frames_exported': True,
        'gpu_initialized': False, 'simulator_launched': False, 'dataset_modified': False,
        'policy_inference': False, 'physical_grasp_validated': False,
        'selection_not_conditioned_on_visual_quality': True}
    (out / 'review.json').write_text(json.dumps(review, indent=2))
    return review


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--count', type=int, default=5)
    p.add_argument('--seed', type=int)
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Explicit CPU-only environment required')
    if not args.output.resolve().is_relative_to(ROOT / 'runs') or args.output.exists():
        raise ValueError('New remote run directory required')
    report = json.loads(args.audit.read_bytes())
    seed = args.seed if args.seed is not None else random.SystemRandom().getrandbits(64)
    entries = select_entries(report, args.count, seed)
    args.output.mkdir(parents=True)
    selection = {'created_at': datetime.now(timezone.utc).isoformat(),
        'sampling': 'uniform without replacement; canonical fingerprint order; no visual filtering or redraw',
        'seed': seed, 'audit': str(args.audit), 'audit_sha256': sha256(args.audit),
        'population': report['eligible_unique_episodes'], 'count': args.count,
        'scope': 'finalized eligible successful RAW sources only; excludes failed attempts and unfinished shards',
        'episodes': entries}
    selected = args.output / 'selection.json'
    selected.write_text(json.dumps(selection, indent=2))
    print('SELECTION_COMMITTED', str(selected), seed, flush=True)
    try:
        reviews = [export_one(e, args.output / f'sample-{i:02d}', i, len(entries),
                    selection['audit_sha256'], sha256(selected)) for i, e in enumerate(entries, 1)]
    except BaseException as error:
        (args.output / 'status.json').write_text(json.dumps({'state': 'FAILED_NO_RESAMPLING', 'error': str(error)}))
        raise
    (args.output / 'status.json').write_text(json.dumps({'state': 'completed', 'count': len(reviews),
        'total_video_bytes': sum(r['video_bytes'] for r in reviews)}, indent=2))
    print('ALL_REVIEWS_COMPLETE', len(reviews), flush=True)


if __name__ == '__main__':
    main()
