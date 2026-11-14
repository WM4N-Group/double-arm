"""Review provenance tests with tiny fake videos; no GPU/media downloads."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('review_export', ROOT / 'scripts/export_demo_review.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class ReviewExportTests(unittest.TestCase):
    def fixture(self, root, stats, videos):
        path = root / 'important_stats.json'
        path.write_text(json.dumps(stats))
        (root / 'videos').mkdir()
        for name in videos:
            (root / 'videos' / name).write_bytes(b'tiny synthetic selection fixture')
        return path

    def test_selects_success_not_failed_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.fixture(root, {
                'num_success': 1, 'num_attempts': 2,
                'all_episode_logs': {'err_status': ['BaseMPFailed', 'None'], 'task_success': [False, True]},
            }, ['0000.mp4', '0001.mp4'])
            _, video, index = review.successful_video(path)
            self.assertEqual(video.name, '0001.mp4')
            self.assertEqual(index, 1)

    def test_problematic_rollout_does_not_shift_video_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.fixture(root, {
                'num_success': 1, 'num_attempts': 1, 'num_problematic': 1,
                'all_episode_logs': {'err_status': ['problematic', 'None'], 'task_success': [False, True]},
            }, ['0000.mp4'])
            _, video, index = review.successful_video(path)
            self.assertEqual(video.name, '0000.mp4')
            self.assertEqual(index, 0)

    def test_rejects_failed_run_even_if_video_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp), {'num_success': 0, 'num_attempts': 1}, ['0000.mp4'])
            with self.assertRaises(ValueError):
                review.successful_video(path)

    def test_rejects_missing_success_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp), {'num_success': 1, 'num_attempts': 1}, [])
            with self.assertRaises(ValueError):
                review.successful_video(path)

    def test_grasp_enum_is_not_interpreted_as_generic_truthiness(self):
        self.assertIs(review.grasp_flag(-1), False)
        self.assertIsNone(review.grasp_flag(0))
        self.assertIs(review.grasp_flag(1), True)
        with self.assertRaises(KeyError):
            review.grasp_flag(2)

    def test_ffprobe_metadata_is_checked(self):
        output = {'streams': [{'width': 1600, 'height': 576, 'codec_name': 'h264'}],
                  'format': {'duration': '12.0', 'size': '1000'}}
        with patch.object(review.subprocess, 'check_output', return_value=json.dumps(output).encode()):
            self.assertEqual(review.read_video(Path('fixture.mp4'))['duration'], 12.0)
        output['format']['duration'] = '0'
        with patch.object(review.subprocess, 'check_output', return_value=json.dumps(output).encode()):
            with self.assertRaises(ValueError):
                review.read_video(Path('fixture.mp4'))

    def test_video_patch_applies_and_reverses_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            source = root / 'robomimic/envs/env_omnigibson.py'
            source.parent.mkdir(parents=True)
            source.write_text('\n' * 921 +
                '            # eef_right_img = obs[f"{robot_name}::{robot_name}:right_eef_link:Camera:0::rgb"]\n'
                '            concatenated_img = hori_concatenate_image([ego_img, viewer_img])\n'
                '            return concatenated_img\n'
                '            # video_writer.append_data(concatenated_img)\n')
            patch_path = ROOT / 'patches/robomimic/001-inspection-video-layout.patch'
            subprocess.run(['git', '-C', str(root), 'apply', '--check', str(patch_path)], check=True)
            subprocess.run(['git', '-C', str(root), 'apply', str(patch_path)], check=True)
            self.assertIn('[viewer_img, ego_img]', source.read_text())
            subprocess.run(['git', '-C', str(root), 'apply', '--reverse', '--check', str(patch_path)], check=True)
