"""Predictive style costs are opt-in; independent acceptance gates stay fixed."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools/motion'))
from optimize_reference_sampling import parse_args


class CoordinationCorridorTests(unittest.TestCase):
    def parse(self, *extra):
        with tempfile.TemporaryDirectory() as folder:
            argv = ['optimizer', '--motion', str(Path(folder)/'source.npz'),
                    '--warm', str(Path(folder)/'guess.npz'), '--output', str(Path(folder)/'new'), *extra]
            with patch.object(sys, 'argv', argv):
                return parse_args()

    def test_existing_runs_have_no_new_peak_coordination_cost(self):
        a = self.parse()
        for name in ('lateral_corridor_weight', 'foot_lateral_safety_weight',
                     'foot_width_safety_weight', 'heading_safety_weight'):
            self.assertEqual(getattr(a, name), 0.)

    def test_explicit_refinement_contract(self):
        a = self.parse('--lateral-corridor-weight', '1e10', '--lateral-corridor-error', '.004',
                       '--foot-lateral-safety-weight', '1e10', '--foot-width-safety-weight', '1e10',
                       '--heading-safety-weight', '1e9', '--heading-safety-deg', '2.5')
        self.assertEqual(a.lateral_corridor_weight, 1e10)
        self.assertEqual(a.heading_safety_deg, 2.5)
        self.assertEqual(a.foot_lateral_safety_error, .008)

    def test_neutral_waist_is_opt_in_motor_target_only(self):
        self.assertFalse(self.parse().neutral_waist_target)
        self.assertTrue(self.parse('--neutral-waist-target').neutral_waist_target)

    def test_torso_tracking_weight_range(self):
        self.assertEqual(self.parse('--torso-pose-weight', '5e7').torso_pose_weight, 5e7)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse('--torso-pose-weight', 'inf')

    def test_invalid_corridors_fail_closed(self):
        for flag, value in [('--lateral-corridor-weight', 'nan'),
                            ('--heading-safety-deg', 'nan'),
                            ('--foot-width-safety-weight', '-1'),
                            ('--foot-lateral-safety-error', '0')]:
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse(flag, value)


if __name__ == '__main__':
    unittest.main()
