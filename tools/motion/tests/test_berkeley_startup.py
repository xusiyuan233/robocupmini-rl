"""Startup warm starts must bind new references, not copy old live states."""
import sys
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools/motion'))
from prepare_initial_support_control import prepare


class StartupSafetyTests(unittest.TestCase):
    def test_outside_project_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as folder:
            a = SimpleNamespace(motion=Path(folder)/'motion.npz', run=None,
                                warm=ROOT/'does_not_exist.npz', output=ROOT/'new_seed.npz')
            with self.assertRaisesRegex(ValueError, 'Outside user project'):
                prepare(a)

    def test_wrong_reference_warm_is_rejected(self):
        import numpy as np
        motion = ROOT/'data/motions/robotcupmini/retargeted/berkeley_v6_ik/candidate_motion.npz'
        # Create a temporary numerical guess only inside this user's project.
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs/berkeley_adaptation_20261120') as folder:
            warm = Path(folder)/'warm.npz'
            np.savez_compressed(warm, controls=np.zeros((420, 13)), reference_sha256=np.array('wrong'))
            a = SimpleNamespace(motion=motion, run=None, warm=warm, output=Path(folder)/'out.npz')
            with self.assertRaisesRegex(ValueError, 'Warm reference mismatch'):
                prepare(a)
            self.assertFalse(a.output.exists())

class ExactPlanningGeometryTests(unittest.TestCase):
    def test_convex_hull_height_matches_transformed_mesh(self):
        import numpy as np
        from scipy.spatial import ConvexHull
        from scipy.spatial.transform import Rotation
        from optimize_reference_sampling import exact_mesh_min_z
        rng = np.random.default_rng(42)
        points = rng.normal(size=(100, 3)) * [.08, .04, .015]
        hull = points[ConvexHull(points).vertices]
        rotations = Rotation.random(28, random_state=rng)
        quaternions = rotations.as_quat()[:, [3, 0, 1, 2]].reshape(4, 7, 4)
        positions = rng.normal(size=(4, 7, 3))
        actual = exact_mesh_min_z(positions, quaternions, hull)
        expected = ((np.einsum('bij,vj->bvi', rotations.as_matrix(), points) +
                     positions.reshape(28, 3)[:, None])[:, :, 2].min(axis=1).reshape(4, 7))
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0.)


if __name__ == '__main__':
    unittest.main()
