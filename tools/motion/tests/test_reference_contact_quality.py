"""CPU-only tests for contact/clearance gates (no IsaacLab, GPU or RL training)."""
import sys
import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference_contact_quality import (
    contact_geometry_report, interior_mask, source_swing_clearance,
)


class ContactQualityTests(unittest.TestCase):
    def test_candidate_refinement_refuses_to_overwrite(self):
        # Import must not start a simulation or parse terminal arguments.
        from refine_human_dynamics_candidate import refine
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            marker = output / 'user_file.txt'
            marker.write_text('preserve')
            with self.assertRaises(FileExistsError):
                refine(SimpleNamespace(output=output))
            self.assertEqual(marker.read_text(), 'preserve')

    def test_transition_guard_does_not_wrap(self):
        mask = np.ones((20, 2), dtype=bool)
        result = interior_mask(mask, 3)
        self.assertFalse(result[:3].any())
        self.assertFalse(result[-3:].any())
        self.assertTrue(result[3:-3].all())

    def test_short_intervals_not_counted_as_interior(self):
        mask = np.zeros((30, 2), dtype=bool)
        mask[4:9, 0] = True
        self.assertFalse(interior_mask(mask, 3).any())

    def test_clearance_follows_measured_lift_and_keeps_support_zero(self):
        t = np.arange(100) * .02
        stance = np.ones(100, dtype=bool)
        stance[20:80] = False
        h = np.full(100, .08)
        h[35:65] += .05
        clear = source_swing_clearance(t, h, stance, .6)
        np.testing.assert_array_equal(clear[stance], 0.)
        self.assertGreater(clear[50], .03)
        self.assertGreater(clear[32], .014)
        self.assertEqual(clear[20], 0.)
        self.assertEqual(clear[79], 0.)
        self.assertTrue(np.isfinite(clear).all())

    def test_mid_clip_swing_has_no_invented_start_contact(self):
        t = np.arange(80) * .02
        stance = np.zeros(80, dtype=bool)
        stance[50:] = True
        clear = source_swing_clearance(t, np.full(80, .08), stance, .6)
        self.assertGreater(clear[0], .015)

    def test_reject_invalid_clearance_inputs(self):
        t = np.arange(20) * .02
        stance = np.ones(20, dtype=bool)
        for scale in [-1, 0, np.nan]:
            with self.assertRaises(ValueError):
                source_swing_clearance(t, t, stance, scale)
        with self.assertRaises(ValueError):
            source_swing_clearance(t[::-1], t, stance, .6)
        with self.assertRaises(ValueError):
            source_swing_clearance(t, t, ~stance, .6)

    def fixture(self):
        contact = np.ones((100, 2), dtype=bool)
        contact[20:80, 0] = False
        z = np.zeros((100, 2))
        z[20:80, 0] = .02
        v = np.zeros((100, 2, 3))
        v[20:80, 0, 0] = .2
        return contact, z, v

    def test_reject_moving_grounded_swing(self):
        c, z, v = self.fixture()
        z[:, 0] = 0
        report = contact_geometry_report(z, v, c, .02)
        self.assertFalse(report['passed'])
        self.assertEqual(report['moving_swing_ground_samples'], 50)

    def test_accept_geometry_is_not_dynamics_certificate(self):
        c, z, v = self.fixture()
        report = contact_geometry_report(z, v, c, .02)
        self.assertTrue(report['passed'])
        self.assertIn('not a dynamics', report['note'])

    def test_stance_sliding_is_a_failure(self):
        c, z, v = self.fixture()
        v[:, 1, 0] = .03
        report = contact_geometry_report(z, v, c, .02)
        self.assertFalse(report['passed'])
        self.assertGreater(report['stance_sliding_samples'], 0)

    def test_no_swing_or_nan_is_not_approval(self):
        c, z, v = self.fixture()
        c[:] = True
        v[:] = 0
        self.assertFalse(contact_geometry_report(z, v, c, .02)['passed'])
        z[0, 0] = np.nan
        with self.assertRaises(ValueError):
            contact_geometry_report(z, v, c, .02)


if __name__ == '__main__':
    unittest.main()
