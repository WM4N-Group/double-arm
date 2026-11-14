"""SERVER ONLY: validate a successful generated demo and export a small review pack.

The pack contains rendered video/thumbnail/metrics only; never robot assets,
license keys, source HDF5 datasets, or model weights.
"""
import argparse
import io
import json
from pathlib import Path
import subprocess


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
    for path in hdf5_files:
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
            if demos:
                datasets.append({'path': str(path), 'demos': demos})
    if not datasets:
        raise SystemExit('No non-empty generated action dataset found')
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
        'source': 'new generated attempt, not source demonstration replay',
        'policy_training': 'not_run', 'camera_layout': ['third_person', 'ego'],
        'stats_path': str(stats_path), 'source_video': str(source), 'attempt_video_index': attempt,
        'num_success': stats['num_success'], 'num_attempts': stats['num_attempts'],
        'num_problematic': stats.get('num_problematic', 0), 'datasets': datasets,
        'video': info,
    }
    (out / 'review.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'review_dir': str(out), **report}, indent=2), flush=True)


if __name__ == '__main__':
    main()
