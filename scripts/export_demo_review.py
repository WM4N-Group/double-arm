"""SERVER ONLY: validate a successful generated demo and export a small review pack.

The pack contains rendered video/thumbnail/metrics only; never robot assets,
license keys, source HDF5 datasets, or model weights.
"""
import argparse
import io
import hashlib
import json
from pathlib import Path
import subprocess


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def grasp_flag(code):
    # Pinned OmniGibson IsGraspingState: FALSE=-1, UNKNOWN=0, TRUE=1.
    # Do not use bool(code): bool(-1) would incorrectly mean grasped.
    return {-1: False, 0: None, 1: True}[int(code)]


def read_motion_metrics(group):
    import numpy as np
    result = {}
    if 'obs/base_qpos' in group:
        q = group['obs/base_qpos'][()]
        if q.ndim == 2 and q.shape[1] == 3 and len(q):
            yaw = float(np.arctan2(np.sin(q[-1, 2] - q[0, 2]), np.cos(q[-1, 2] - q[0, 2])))
            result.update(base_displacement_m=float(np.linalg.norm(q[-1, :2] - q[0, :2])),
                          base_yaw_change_deg=float(np.degrees(yaw)))
    for arm in ('left', 'right'):
        key = f'obs/grasp_{arm}'
        if key in group:
            code = int(np.asarray(group[key][-1]).reshape(-1)[0])
            result[f'{arm}_grasp_final_code'] = code
            result[f'{arm}_grasp_final'] = grasp_flag(code)
    if 'datagen_info/object_poses' in group:
        for name, values in group['datagen_info/object_poses'].items():
            if 'coffee_cup' in name and values.shape[-2:] == (4, 4):
                result['cup_height_change_m'] = float(values[-1, 2, 3] - values[0, 2, 3])
    return result


def read_video(path):
    data = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,codec_name,nb_frames,r_frame_rate:format=duration,size',
        '-of', 'json', str(path),
    ]))
    stream = data['streams'][0]
    stream['duration'] = float(data['format']['duration'])
    stream['bytes'] = int(data['format']['size'])
    if stream['duration'] <= 0 or stream['width'] < 128 or stream['height'] < 128:
        raise ValueError(f'Invalid review video: {path}')
    return stream


def successful_video(stats_path):
    stats = json.loads(stats_path.read_text())
    if stats.get('num_success', 0) < 1:
        raise ValueError('No successful generated trajectory recorded')
    logs = stats.get('all_episode_logs', {})
    statuses = logs.get('err_status', [])
    outcomes = logs.get('task_success', [])
    valid_index = 0
    for status, success in zip(statuses, outcomes):
        if status == 'problematic':
            continue
        video = stats_path.parent / 'videos' / f'{valid_index:04d}.mp4'
        if success and video.is_file():
            return stats, video, valid_index
        valid_index += 1
    if stats.get('num_attempts') == 1 and stats.get('num_success') == 1:
        video = stats_path.parent / 'videos/0000.mp4'
        if video.is_file():
            return stats, video, 0
    raise ValueError('Cannot match success statistics to an actual attempt video')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source-video', type=Path)
    args = parser.parse_args()
    candidates = []
    for path in args.run_dir.rglob('important_stats.json'):
        if json.loads(path.read_text()).get('num_success', 0) > 0:
            candidates.append(path)
    if len(candidates) != 1:
        raise SystemExit(f'Expected one successful stats file, got {len(candidates)}')
    stats_path = candidates[0]
    stats, source, attempt = successful_video(stats_path)
    hdf5_files = list(stats_path.parent.glob('*.hdf5'))
    import h5py
    datasets = []
    motion_metrics = {}
    for path in hdf5_files:
        if 'failed' in path.stem:
            continue
        with h5py.File(path, 'r') as file:
            if 'data' not in file:
                continue
            demos = []
            for name, group in file['data'].items():
                if not name.startswith('demo_'):
                    continue
                actions = group.get('actions', group.get('action'))
                if actions is not None and len(actions) > 0:
                    demos.append({'name': name, 'steps': len(actions)})
                    if not motion_metrics:
                        motion_metrics = read_motion_metrics(group)
            if demos:
                datasets.append({'path': str(path), 'demos': demos})
    if not datasets:
        raise SystemExit('No non-empty generated action dataset found')
    rendering = {'rendering_mode': 'live_generated_attempt', 'camera_layout': ['third_person_follow']}
    original_video = source
    if args.source_video:
        manifest_path = args.source_video.with_suffix('.json')
        rendering = json.loads(manifest_path.read_text())
        if rendering.get('rendering_mode') != 'generated_state_replay':
            raise ValueError('Expected a generated-state replay manifest')
        matched = [item for item in datasets if Path(item['path']).resolve() == Path(rendering['dataset']).resolve()]
        if not matched or rendering['num_states'] != sum(item['steps'] for item in matched[0]['demos']):
            raise ValueError('Replay provenance differs from the successful generated dataset')
        if sha256_file(rendering['dataset']) != rendering['dataset_sha256']:
            raise ValueError('Replay dataset hash no longer matches')
        source = args.source_video
    out = args.output or args.run_dir / 'review' 
    out.mkdir(parents=True, exist_ok=False)
    dest = out / 'demo.mp4'
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-i', str(source),
        '-vf', "scale='min(1600,iw)':-2", '-c:v', 'libx264', '-preset', 'medium',
        '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', str(dest),
    ], check=True)
    info = read_video(dest)
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(dest), '-f', 'null', '-'], check=True)
    from PIL import Image, ImageDraw
    thumbs = []
    for fraction in (0.02, 0.20, 0.40, 0.60, 0.80, 0.98):
        second = max(0, min(info['duration'] * fraction, info['duration'] - 0.05))
        raw = subprocess.check_output([
            'ffmpeg', '-v', 'error', '-ss', str(second), '-i', str(dest),
            '-frames:v', '1', '-vf', 'scale=640:-2', '-f', 'image2pipe', '-vcodec', 'png', '-',
        ])
        image = Image.open(io.BytesIO(raw)).convert('RGB')
        ImageDraw.Draw(image).text((8, 8), f'{second:.1f}s', fill='yellow', stroke_width=1, stroke_fill='black')
        thumbs.append(image)
    width, height = thumbs[0].size
    sheet = Image.new('RGB', (3 * width, 2 * (height + 20)), '#161616')
    for index, image in enumerate(thumbs):
        sheet.paste(image, ((index % 3) * width, (index // 3) * (height + 20)))
    sheet.save(out / 'contact-sheet.jpg', quality=88)
    report = {
        'task': 'r1_pick_cup_D0', 'method': 'MoMaGen demonstration generation',
        'source': 'new generated trajectory, not the original source demonstration',
        'policy_training': 'not_run', 'camera_layout': rendering['camera_layout'],
        'rendering_mode': rendering['rendering_mode'], 'native_attempt_video': str(original_video),
        'stats_path': str(stats_path), 'source_video': str(source), 'attempt_video_index': attempt,
        'num_success': stats['num_success'], 'num_attempts': stats['num_attempts'],
        'num_problematic': stats.get('num_problematic', 0), 'datasets': datasets,
        'video': info, 'observed_motion': motion_metrics,
        'render_provenance': rendering if args.source_video else None,
    }
    (out / 'review.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'review_dir': str(out), **report}, indent=2), flush=True)


if __name__ == '__main__':
    main()
