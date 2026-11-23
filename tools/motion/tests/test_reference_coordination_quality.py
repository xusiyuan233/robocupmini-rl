"""Motion quality cannot be inferred from surviving a physics rollout."""
import sys,unittest,tempfile,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference_coordination_quality import rotational_metrics,measure_motion,from_audit,audit_quality
from retime_reference_candidate import retime
ROOT=Path(__file__).resolve().parents[3]

class CoordinationQualityTests(unittest.TestCase):
    def test_low_frequency_roll_not_mistaken_for_jitter(self):
        t=np.arange(600)*.02;r=Rotation.from_euler('x',.05*np.sin(2*np.pi*.5*t))
        m=rotational_metrics(r,.02)
        self.assertLess(m['highfreq_rms_deg'],.03)
        noise=Rotation.from_euler('x',.05*np.sin(2*np.pi*.5*t)+.015*np.sin(2*np.pi*8*t))
        self.assertGreater(rotational_metrics(noise,.02)['highfreq_rms_deg'],.3)

    def test_prior_motion_fails_torso_and_spacing_targets(self):
        a,d=from_audit(ROOT/'data/motions/robotcupmini/validated/sway_correction_20261119/v14_final_audit')
        m=measure_motion(a)
        self.assertTrue(d['dynamics_certified_in_this_mujoco_contract'])
        self.assertGreater(m['torso']['highfreq_rms_deg'],.5)
        self.assertLess(m['foot_width_p05_median_p95_m'][1],.15)
        self.assertLess(m['torso_roll_correlation_to_source'],0.)

    def test_report_binds_the_independently_replayed_physics(self):
        audit=ROOT/'data/motions/robotcupmini/validated/coordination_20261119/v16_smooth_audit'
        with tempfile.TemporaryDirectory(prefix='robocup_quality_') as td:
            output=Path(td)/'quality.json';r=audit_quality(audit,output)
            self.assertTrue(r['physics_passed'])
            self.assertEqual(r['physics_certificate_sha256'],hashlib.sha256((audit/'decision.json').read_bytes()).hexdigest())
            self.assertEqual(r['physics_recording_sha256'],hashlib.sha256((audit/'physics_replay.npz').read_bytes()).hexdigest())
            self.assertEqual(r['status'],'COORDINATION_CHECKS_FAILED')
            self.assertEqual(r['failed_checks'],['lateral_sway'])
            self.assertFalse(r['human_equivalence_certified'])
            self.assertFalse(r['visual_review_performed_by_this_script'])
            self.assertEqual(json.loads(output.read_text())['checks'],r['checks'])

    def test_whole_clip_time_plot_can_render_without_changing_states(self):
        from render_coordination_review import plot_time_review
        audit=ROOT/'data/motions/robotcupmini/validated/coordination_20261119/v16_smooth_audit'
        a,_=from_audit(audit);before={k:v.copy() for k,v in a.items()}
        with tempfile.TemporaryDirectory(prefix='robocup_quality_') as td:
            plot_time_review(a,audit,Path(td))
            self.assertGreater((Path(td)/'whole_clip_time_review.png').stat().st_size,10000)
        for k,v in before.items():np.testing.assert_array_equal(a[k],v)

    def test_certified_physics_must_not_be_retimed(self):
        source=ROOT/'data/motions/robotcupmini/retargeted/walk1_v14_reduced_sway_accepted/candidate_motion.npz'
        with tempfile.TemporaryDirectory(prefix='robocup_quality_') as td:
            output=Path(td)/'not_created'
            with self.assertRaises(ValueError):retime(source,output,.75)
            self.assertFalse(output.exists())

    def test_source_retiming_preserves_source_timestamps_not_approval(self):
        source=ROOT/'data/motions/robotcupmini/retargeted/walk1_candidate_v5_human/candidate_motion.npz'
        with tempfile.TemporaryDirectory(prefix='robocup_quality_') as td:
            output=Path(td)/'candidate';retime(source,output,.75)
            with np.load(output/'candidate_motion.npz') as a:
                self.assertFalse(bool(a['valid_for_training']))
                self.assertEqual(len(a['time_s']),450)
                np.testing.assert_allclose(a['source_time_s'],135.5+a['time_s']/1.5,atol=1e-9)

if __name__=='__main__':unittest.main()
