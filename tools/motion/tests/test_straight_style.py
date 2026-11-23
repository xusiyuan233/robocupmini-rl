"""Regression checks: no rewarding a filtered display or a drifting straight clip."""
import sys,unittest,tempfile,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_straight_style import judge,TARGETS
from prepare_source_initializer import prepare

class StraightStyleTests(unittest.TestCase):
    def metrics(self):
        return dict(torso_yaw_correlation_to_source=.9,torso_roll_correlation_to_source=.9,torso_yaw_rmse_to_source_deg=2.,duration_s=8.5,detrended_pelvis_sway_range_m=.05,
                    foot_width_p05_median_p95_m=[.145,.165,.18],
                    torso=dict(highfreq_rms_deg=.1,peak_rate_deg_s=30.,acceleration_p95_deg_s2=200.),
                    swing_side_excursion_p95_m=.01)
    def test_prediction_reference_is_at_future_state_time(self):
        import numpy as np
        from optimize_reference_sampling import prediction_frames
        i,j,w=prediction_frames(7,20,20,100)
        self.assertEqual(i[0],7);self.assertAlmostEqual(w[0],.05)
        self.assertEqual(i[-1],8);self.assertEqual(w[-1],0.)
        np.testing.assert_allclose(i+w,7+np.arange(1,21)/20)

    def test_old_sway_is_not_acceptable(self):
        a=self.metrics();a['detrended_pelvis_sway_range_m']=.09668
        self.assertFalse(judge(a,a,0.,True)['lateral_sway'])
    def test_improvement_cannot_override_bad_physics(self):
        a=self.metrics();old=dict(detrended_pelvis_sway_range_m=.09668)
        self.assertFalse(judge(a,old,0.,False)['physics'])
    def test_detrending_does_not_hide_drifting_path(self):
        a=self.metrics();old=dict(detrended_pelvis_sway_range_m=.09668)
        self.assertFalse(judge(a,old,.01,True)['no_lateral_path_drift'])
    def test_foot_spacing_and_upper_body_still_checked(self):
        a=self.metrics();old=dict(detrended_pelvis_sway_range_m=.09668)
        self.assertTrue(all(judge(a,old,.001,True).values()))
        a['foot_width_p05_median_p95_m'][1]=.12
        a['torso']['highfreq_rms_deg']=.5
        c=judge(a,old,0.,True);self.assertFalse(c['foot_spacing']);self.assertFalse(c['torso_jitter'])
    def test_source_initializer_rejects_other_user_project_before_reading(self):
        with self.assertRaises(ValueError):prepare(Path('/tmp/lilin/source.bvh'),0,10,1,Path('/tmp/output'))
    def test_verified_loader_requires_bound_passing_quality(self):
        from verified_style_data import load_verified_motion,ROOT
        base=json.loads((ROOT/'data/motions/robotcupmini/validated/coordination_20261119/selected_coordinated_motion.json').read_text())
        with tempfile.TemporaryDirectory(prefix='quality_test_',dir=ROOT/'data/motions/robotcupmini/validated/lateral_correction_20261120') as td:
            folder=Path(td);q=folder/'quality.json';m=folder/'manifest.json'
            report=dict(status='STRAIGHT_STYLE_TARGETS_MET',checks={'lateral_sway':False},physics_passed=True,
                        physics_certificate_sha256=base['certificate']['sha256'])
            q.write_text(json.dumps(report))
            base['style_quality_report']=dict(path=str(q.relative_to(ROOT)),sha256=hashlib.sha256(q.read_bytes()).hexdigest())
            m.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError,'quality/physics'):load_verified_motion(m)
            report['checks']['lateral_sway']=True;report['physics_certificate_sha256']='0'*64
            q.write_text(json.dumps(report));base['style_quality_report']['sha256']=hashlib.sha256(q.read_bytes()).hexdigest()
            m.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError,'quality/physics'):load_verified_motion(m)
            q.write_text('changed')
            with self.assertRaisesRegex(ValueError,'report changed'):load_verified_motion(m)

    def test_threshold_not_relaxed(self):
        self.assertEqual(TARGETS['maximum_sway_peak_to_peak_m'],.055)
        self.assertEqual(TARGETS['minimum_foot_width_median_m'],.15)
if __name__=='__main__':unittest.main()
