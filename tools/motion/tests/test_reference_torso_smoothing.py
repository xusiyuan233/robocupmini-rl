"""The torso planning feature is exact and never changes its input state."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools/motion'))
from reference_torso_smoothing import torso_rotation_vector


class TorsoPlanningFeatureTests(unittest.TestCase):
    def test_matches_full_rotation_composition_without_modifying_states(self):
        rng = np.random.default_rng(121)
        q = np.zeros((7, 9, 20))
        root = Rotation.random(63, random_state=rng)
        q[..., 3:7] = root.as_quat()[:, [3, 0, 1, 2]].reshape(7, 9, 4)
        q[..., 19] = rng.normal(size=(7, 9))
        before = q.copy()
        expected = (root*Rotation.from_euler('z', q[..., 19].ravel())).as_rotvec().reshape(7, 9, 3)
        np.testing.assert_allclose(torso_rotation_vector(q, 19), expected, atol=1e-12)
        np.testing.assert_array_equal(q, before)

    def test_identity_is_finite(self):
        q = np.zeros(20); q[3] = 1.
        np.testing.assert_array_equal(torso_rotation_vector(q, 19), np.zeros(3))

    def test_quaternion_sign_invariance(self):
        q = np.zeros((3, 20)); q[:, 3:7] = Rotation.from_euler('xyz', [[.1,.2,.3],[.2,.1,-.3],[-.2, .3, .4]]).as_quat()[:, [3,0,1,2]]
        q[:, 19] = .05
        a = torso_rotation_vector(q, 19)
        q[:, 3:7] *= -1
        np.testing.assert_allclose(torso_rotation_vector(q, 19), a, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
