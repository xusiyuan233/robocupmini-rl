"""Regression checks for robot-source provenance and unchanged control contracts."""
import sys,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools/motion'))
from collect_berkeley_teacher import build_teacher
from optimize_reference_sampling import parse_args

class BerkeleySourceTests(unittest.TestCase):
    def test_native_teacher_model_contract(self):
        m,xml,omitted=build_teacher()
        self.assertEqual(m.nu,12);self.assertAlmostEqual(m.opt.timestep,.002)
        self.assertAlmostEqual(float(m.body_mass.sum()),16.0567651,places=6)
        self.assertEqual(len(omitted),26)
        self.assertEqual(sum(b.find('inertial') is not None for b in xml.findall('.//body')),13)
        self.assertTrue(all(g.get('mesh') is None for g in xml.findall('.//geom')))
    def test_old_optimizer_defaults_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            args=['tool','--motion',str(Path(td)/'motion.npz'),'--warm',str(Path(td)/'warm.npz'),'--output',str(Path(td)/'new')]
            with patch.object(sys,'argv',args):a=parse_args()
            self.assertEqual(a.root_forward_weight,250.)
            self.assertEqual(a.root_forward_velocity_weight,0.)
            self.assertEqual(a.foot_forward_weight,1200.)
    def test_robot_initializer_is_not_a_demo(self):
        p=ROOT/'data/motions/robotcupmini/retargeted/berkeley_v2_initializer/candidate_motion.npz'
        with np.load(p,allow_pickle=False) as a:
            self.assertFalse(bool(a['valid_for_training']))
            self.assertFalse(bool(a['periodic_reference']))
            self.assertEqual(a['joint_positions_rad'].shape,(420,13))
            self.assertTrue(np.allclose(a['joint_positions_rad'][:,-1],0.))
            self.assertNotIn('human_positions_m',a.files)
            self.assertEqual(a['source_sole_clearance_m'].shape,(420,2))
            self.assertTrue(np.all(a['source_sole_clearance_m']>=0.))
            self.assertTrue(np.all(a['source_sole_clearance_m']<=.03))
    def test_source_rollout_has_native_state_and_policy_provenance(self):
        p=ROOT/'data/motions/robotcupmini/raw/berkeley_humanoid/teacher_040_heading/teacher_rollout.npz'
        with np.load(p,allow_pickle=False) as a:
            self.assertFalse(bool(a['valid_for_training']))
            self.assertEqual(a['qpos'].shape,(701,19))
            self.assertEqual(a['qvel'].shape,(701,18))
            self.assertEqual(a['command'].shape,(701,3))
            self.assertEqual(len(str(a['source_policy_sha256'])),64)
            self.assertTrue(np.allclose(np.diff(a['time_s']),.02))
            self.assertTrue(np.all(np.isfinite(a['qvel'])))
if __name__=='__main__':unittest.main()
