"""SERVER ONLY: inspect a generated trajectory from new cameras, not a policy run."""
import argparse
import json
import os
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get('ACCEPT_BEHAVIOR_DATASET_TERMS') != 'YES' or os.environ.get('ACCEPT_NVIDIA_EULA') != 'YES':
        raise SystemExit('Explicit simulator and asset license confirmation required')
    import h5py
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    import omnigibson as og
    from robomimic.envs.env_omnigibson import EnvOmniGibson
    with h5py.File(args.dataset, 'r') as file:
        meta = json.loads(file['data'].attrs['env_args'])
        demos = [key for key in file['data'] if key.startswith('demo_')]
        if len(demos) != 1:
            raise ValueError('Inspection renderer currently requires one generated episode')
        states = file[f'data/{demos[0]}/states'][()]
        action_count = len(file[f'data/{demos[0]}/actions'])
    if len(states) != action_count or not len(states):
        raise ValueError('Recorded state/action lengths differ or are empty')
    stats_path = args.dataset.parent / 'important_stats.json'
    if json.loads(stats_path.read_text()).get('num_success', 0) < 1:
        raise ValueError('Only render a trajectory with verified generation success')
    # Only visualization is performed; no MP solver, actions or policy is used.
    meta['env_kwargs']['init_curobo'] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    failed = False
    writer = None
    try:
        env = EnvOmniGibson(env_name=meta['env_name'], **meta['env_kwargs'])
        from pxr import Gf
        matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(3.0, -3.5, 2.5), Gf.Vec3d(0.5, 0.0, 0.8), Gf.Vec3d(0, 0, 1)).GetInverse()
        quat = matrix.ExtractRotationQuat()
        imaginary = quat.GetImaginary()
        camera_pos = torch.tensor([3.0, -3.5, 2.5])
        camera_quat = torch.tensor([imaginary[0], imaginary[1], imaginary[2], quat.GetReal()], dtype=torch.float32)
        actual_size = int(og.sim.dump_state(serialized=True).numel())
        if actual_size != states.shape[1]:
            raise ValueError(f'Scene state layout differs: {actual_size} vs {states.shape[1]}')
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
        except OSError:
            font = ImageFont.load_default()
        writer = imageio.get_writer(str(args.output), fps=20, codec='libx264', quality=8, macro_block_size=2)
        wrist_key = None
        for index, state in enumerate(states):
            og.sim.load_state(torch.as_tensor(state, dtype=torch.float32), serialized=True)
            og.sim.viewer_camera.set_position_orientation(position=camera_pos, orientation=camera_quat)
            for _ in range(4 if index == 0 else 1):
                og.sim.render()
            obs, _ = env.env.get_obs()
            if wrist_key is None:
                candidates = [key for key in obs if key.endswith(':left_eef_link:Camera:0::rgb')]
                if not candidates:
                    candidates = [key for key in obs if key.endswith(':eyes:Camera:0::rgb')]
                if not candidates:
                    raise ValueError('No wrist or ego RGB sensor found')
                wrist_key = candidates[0]
                print('INSPECTION_RGB_SENSOR', wrist_key, flush=True)
            global_rgb = og.sim.viewer_camera._get_obs()[0]['rgb'].cpu().numpy()[:, :, :3]
            wrist_rgb = obs[wrist_key].cpu().numpy()[:, :, :3]
            panels = []
            for image, label in ((global_rgb, 'GLOBAL / recorded trajectory'),
                                 (wrist_rgb, 'LEFT WRIST' if 'left_eef_link' in wrist_key else 'EGO')):
                height = 480
                width = max(2, 2 * round(image.shape[1] * height / image.shape[0] / 2))
                panel = Image.fromarray(image).resize((width, height))
                framed = Image.new('RGB', (width, height + 28), '#171717')
                framed.paste(panel, (0, 28))
                ImageDraw.Draw(framed).text((8, 4), label, font=font, fill='white')
                panels.append(np.asarray(framed))
            writer.append_data(np.concatenate(panels, axis=1))
            if index % 50 == 0 or index == len(states) - 1:
                print(f'RENDER_PROGRESS {index + 1}/{len(states)}', flush=True)
        writer.close()
        writer = None
        from export_demo_review import sha256_file
        digest = sha256_file(args.dataset)
        manifest = {
            'dataset': str(args.dataset.resolve()), 'dataset_sha256': digest,
            'demo_keys': demos, 'num_states': len(states),
            'rendering_mode': 'generated_state_replay', 'policy_training': 'not_run',
            'camera_layout': ['global_fixed', 'left_wrist' if 'left_eef_link' in wrist_key else 'ego'],
            'frame_size': [sum(panel.shape[1] for panel in panels), panels[0].shape[0]],
        }
        args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2))
        print('INSPECTION_RENDER_COMPLETE', json.dumps(manifest), flush=True)
    except BaseException:
        failed = True
        traceback.print_exc()
    finally:
        if writer is not None:
            writer.close()
        try:
            og.shutdown()
        except SystemExit:
            pass
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
