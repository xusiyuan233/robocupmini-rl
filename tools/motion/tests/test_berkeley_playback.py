"""Playback launcher works independently of the RL Conda environment."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'tools/motion/play_berkeley_reference.sh'


class BerkeleyPlaybackTests(unittest.TestCase):
    def run_script(self, *args, env=None):
        return subprocess.run(['/bin/bash', str(SCRIPT), *args], cwd='/tmp',
                              env=env, capture_output=True, text=True, timeout=20)

    def test_shell_syntax(self):
        p = subprocess.run(['/bin/bash', '-n', str(SCRIPT)], capture_output=True)
        self.assertEqual(p.returncode, 0)

    def test_help_does_not_need_display_or_conda(self):
        p = self.run_script('--help', env={'PATH': '/usr/bin:/bin'})
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('--check', p.stdout)

    def test_invalid_argument_is_rejected(self):
        self.assertEqual(self.run_script('--invalid').returncode, 2)
        self.assertEqual(self.run_script('--check', '--invalid').returncode, 2)

    def test_full_decode_check_without_display_or_conda(self):
        p = self.run_script('--check', env={'PATH': '/usr/bin:/bin'})
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('整段视频解码成功', p.stdout)
        self.assertIn('/usr/bin/ffplay', p.stdout)

    def test_default_launch_prefers_ffplay_not_xdg_open(self):
        # Stub only the player; file/index checks are real. Never launch a GUI.
        with tempfile.TemporaryDirectory() as temp:
            player = Path(temp)/'ffplay'
            player.write_text('#!/bin/sh\nprintf "TEST_PLAYER_ARG=%s\\n" "$@"\n')
            player.chmod(0o755)
            p = self.run_script(env={'PATH': temp+':/usr/bin:/bin', 'DISPLAY': ':stub'})
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn('TEST_PLAYER_ARG=-autoexit', p.stdout)
            self.assertIn('TEST_PLAYER_ARG=-threads', p.stdout)
            self.assertIn('_review/berkeley_to_robotcupmini_physics.mp4', p.stdout)
            self.assertNotIn('Traceback', p.stderr)

    def test_missing_desktop_gets_useful_error(self):
        p = self.run_script(env={'PATH': '/usr/bin:/bin'})
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn('没有图形显示会话', p.stderr)


if __name__ == '__main__':
    unittest.main()
