"""CPU checks: don't remove path drift, fake contacts or self-certify dynamics."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference_sway_quality import walking_centerline, compare
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'data/motions/robotcupmini/retargeted/walk1_v13_physics_accepted/candidate_motion.npz'

class SwayQualityTests(unittest.TestCase):
    def test_straight_path_drift_preserved(self):
        t=np.arange(600)*.02; y=.1+.005*t
        np.testing.assert_allclose(walking_centerline(y,.02,2),y,atol=1e-12)

    def test_step_oscillation_not_copied_into_path(self):
        t=np.arange(1500)*.02
        y=.005*t+.04*np.sin(2*np.pi*t/3)
        c=walking_centerline(y,.02,2)
        self.assertLess(np.ptp((c-.005*t)[200:-200]),.001)

    def test_small_arrays_and_nonfinite_rejected(self):
        for a in [np.array([1.,2.]),np.array([1.,np.nan,2.]),np.zeros((4,2))]:
            with self.assertRaises(ValueError): walking_centerline(a,.02,2)
        for dt,s in [(0.,2.),(.02,0.),(.02,float('nan'))]:
            with self.assertRaises(ValueError): walking_centerline(np.zeros(10),dt,s)

    def test_optimizer_limit_cost_is_planning_only_and_optional(self):
        from unittest.mock import patch
        from optimize_reference_sampling import parse_args
        argv=['optimizer','--motion','unused.npz','--warm','unused.npz',
              '--output','/tmp/robocup_sway_unused_output_for_cli_test']
        with patch('sys.argv',argv):
            args=parse_args()
        self.assertEqual(args.joint_limit_weight,3000.)
        self.assertEqual(args.joint_limit_margin,0.)
        with patch('sys.argv',argv+['--joint-limit-weight','100000','--joint-limit-margin','.01']):
            args=parse_args()
        self.assertEqual(args.joint_limit_weight,100000.)
        self.assertEqual(args.joint_stop,.004)  # solver/physical limit unchanged

    def test_prior_sway_does_not_pass_new_gate(self):
        report=compare(BASE,BASE)
        self.assertFalse(report['dynamics_approved_by_this_report'])
        self.assertFalse(report['checks']['at_least_35_percent_less_lateral_range'])
        self.assertFalse(report['checks']['detrended_lateral_range_at_most_10cm'])

if __name__=='__main__': unittest.main()
