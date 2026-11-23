"""CPU-only safety/provenance tests for dynamics acceptance tooling."""
from pathlib import Path
import sys
import unittest
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from physical_reference_model import build_physical_model,check_transition,mj
from audit_physical_reference import replay,GATES
from play_retarget_motion import load_motion
ROOT=Path(__file__).resolve().parents[3]
REFERENCE=ROOT/'data/motions/robotcupmini/retargeted/walk1_candidate_v6_dynamics_draft/candidate_motion.npz'

class PhysicalReferenceTests(unittest.TestCase):
    def names(self):
        return load_motion(REFERENCE)[1]

    def test_import_does_not_start_optimization_or_parse_cli(self):
        import optimize_physical_reference
        self.assertTrue(callable(optimize_physical_reference.run))

    def test_free_root_effort_limits_and_training_pd(self):
        m,ids,adr,dofs,feet,vertices,anchors,kp,kd,effort=build_physical_model(self.names())
        self.assertEqual(m.nq,20);self.assertEqual(m.nv,19);self.assertEqual(m.nu,13)
        self.assertEqual(m.jnt_type[0],mj.mjtJoint.mjJNT_FREE)
        np.testing.assert_allclose(m.actuator_forcerange,np.c_[-effort,effort])
        np.testing.assert_allclose(kd,.05*kp)
        self.assertAlmostEqual(m.body_mass.sum(),32.6565507)
        self.assertEqual(np.max(m.dof_armature),0.)
        np.testing.assert_allclose(m.opt.gravity,[0,0,-9.81])
        self.assertEqual(m.neq,0) # no root pinning/weld/hidden constraints
        self.assertEqual(len(vertices),2)

    def test_contact_calibration_is_explicit_and_bounded(self):
        names=self.names();m,*_=build_physical_model(names,.008)
        np.testing.assert_allclose(m.geom_solref[:,0],.008)
        with self.assertRaises(ValueError):build_physical_model(names,.0001)
        with self.assertRaises(ValueError):build_physical_model(names,1.)

    def test_instrumentation_and_joint_stop_do_not_change_inertias_or_limits(self):
        names=self.names()
        base,*_=build_physical_model(names,.006,joint_limit_time_constant=.004)
        instrumented,*_=build_physical_model(names,.006,sensors=True,joint_limit_time_constant=.004)
        np.testing.assert_allclose(base.body_mass,instrumented.body_mass)
        np.testing.assert_allclose(base.body_inertia,instrumented.body_inertia)
        np.testing.assert_allclose(base.jnt_range,instrumented.jnt_range)
        np.testing.assert_allclose(base.actuator_forcerange,instrumented.actuator_forcerange)
        self.assertEqual(instrumented.nsensordata,26)
        np.testing.assert_allclose(instrumented.jnt_solref[:,0],.004)
        with self.assertRaises(ValueError):
            build_physical_model(names,joint_limit_time_constant=.0001)

    def test_sampling_optimizer_import_does_not_run_it(self):
        import optimize_reference_sampling
        import make_physical_teacher_seed
        self.assertTrue(callable(optimize_reference_sampling.run))
        self.assertTrue(callable(make_physical_teacher_seed.generate))

    def test_conservative_contact_proxies(self):
        from physical_reference_model import oriented_box_min_z,contact_speed_bound
        from scipy.spatial.transform import Rotation
        rng=np.random.default_rng(42)
        rots=Rotation.random(12,random_state=rng)
        quat=rots.as_quat()[:,[3,0,1,2]]
        low=np.array([-.1,-.04,0]);high=np.array([.1,.04,.04])
        corners=np.array([[x,y,z] for x in [low[0],high[0]] for y in [low[1],high[1]] for z in [0,.04]])
        positions=np.zeros((12,3));positions[:,2]=.5
        predicted=oriented_box_min_z(positions,quat,low,high)
        exact=np.einsum('nij,vj->nvi',rots.as_matrix(),corners)[:,:,2].min(1)+.5
        np.testing.assert_allclose(predicted,exact,atol=1e-12)
        linear=rng.normal(size=(12,3));angular=rng.normal(size=(12,3))
        radius=.14
        offsets=rng.normal(size=(12,3));offsets*=radius/np.linalg.norm(offsets,axis=1,keepdims=True)
        real=np.linalg.norm((linear+np.cross(angular,offsets))[:,:2],axis=1)
        self.assertTrue(np.all(real<=contact_speed_bound(linear,angular,radius)+1e-12))

    def test_verified_loader_refuses_failed_certificate_before_reading_data(self):
        from verified_style_data import load_verified_motion
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'manifest.json';p.write_text('{"status":"NOT_APPROVED"}')
            with self.assertRaises(ValueError):load_verified_motion(p)

    def test_transition_rejects_numerical_reset_or_nan(self):
        initial=np.zeros(40);initial[0]=1.
        final=initial.copy();final[0]+=.02;check_transition(initial,final,.02)
        final[0]=0.
        with self.assertRaises(FloatingPointError):check_transition(initial,final,.02)
        final=initial.copy();final[0]+=.02;final[-1]=np.nan
        with self.assertRaises(FloatingPointError):check_transition(initial,final,.02)

    def test_audit_refuses_to_overwrite_or_access_other_project(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);marker=p/'keep.txt';marker.write_text('keep')
            with self.assertRaises(FileExistsError):replay(p/'nonexistent',p)
            self.assertEqual(marker.read_text(),'keep')
            with self.assertRaises(ValueError):replay(p/'lilin',p/'new_output')

    def test_acceptance_does_not_raise_speed_or_effort_limit(self):
        self.assertEqual(GATES['joint_speed_limit_rad_s'],5.)
        self.assertLessEqual(GATES['maximum_effort_ratio'],1.00001)
        self.assertGreaterEqual(GATES['minimum_duration_s'],8.)
        self.assertLessEqual(GATES['maximum_mesh_penetration_m'],.002)

if __name__=='__main__':unittest.main()
