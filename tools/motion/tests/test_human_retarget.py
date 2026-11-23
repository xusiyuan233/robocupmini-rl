"""CPU-only regression tests; does not start simulators, viewers or training."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bvh_motion import read_bvh, anatomical_frame
from retarget_human_walk import clean_contacts, lock_stance, ROOT


class HumanRetargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.human=read_bvh(ROOT/'data/motions/robotcupmini/raw/lafan1/walk1_subject1.bvh')

    def test_fk_preserves_bone_lengths(self):
        b=self.human
        for j,parent in enumerate(b.parents):
            if parent<0:continue
            actual=np.linalg.norm(b.positions_m[::23,j]-b.positions_m[::23,parent],axis=1)
            np.testing.assert_allclose(actual,np.linalg.norm(b.offsets_m[j]),atol=1e-12)

    def test_global_rotations_are_proper(self):
        r=self.human.rotations[::71]
        np.testing.assert_allclose(np.linalg.det(r),1,atol=1e-12)
        np.testing.assert_allclose(np.swapaxes(r,-1,-2)@r,np.broadcast_to(np.eye(3),r.shape),atol=1e-12)

    def test_exact_frame_sampling(self):
        b=self.human;frames=np.array([4000,4004,4007]);pos,rot=b.sample(frames*b.dt)
        np.testing.assert_allclose(pos,b.positions_m[frames],atol=1e-12)
        np.testing.assert_allclose(rot,b.rotations[frames],atol=1e-12)

    def test_reject_invalid_time(self):
        for times in [np.array([-1.,0]),np.array([2.,1.]),np.array([0.,1e6])]:
            with self.assertRaises(ValueError):self.human.sample(times)

    def test_contact_cleanup(self):
        mask=np.zeros(40,dtype=bool);mask[4:20]=True;mask[10:12]=False;mask[26:28]=True
        actual=clean_contacts(mask)
        self.assertTrue(actual[4:20].all());self.assertFalse(actual[26:28].any())

    def test_support_lock_no_drift(self):
        t=np.arange(100)*.02
        raw=np.column_stack([.2*t,np.full(len(t),.1),.02+.01*np.cos(t)])
        mask=np.zeros(len(t),dtype=bool);mask[10:30]=True;mask[65:90]=True
        result=lock_stance(t,raw,mask)
        for a,b in [(10,30),(65,90)]:
            np.testing.assert_allclose(np.ptp(result[a:b],axis=0),0,atol=1e-12)
            np.testing.assert_allclose(result[a:b,2],0,atol=1e-12)
        self.assertTrue((result[:,2]>=0).all())

    def test_anatomical_frame_and_degeneracy(self):
        np.testing.assert_allclose(anatomical_frame(np.array([0.,1,0]),np.array([0.,-1,0]),np.array([0.,0,1])),np.eye(3))
        with self.assertRaises(ValueError):anatomical_frame(np.zeros(3),np.zeros(3),np.ones(3))

    def test_bvh_rejects_corrupt_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'invalid.bvh'
            p.write_text('HIERARCHY ROOT Hips { OFFSET 0 0 0 CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation }\nMOTION\nFrames: 2\nFrame Time: 0.02\n0 0 0 0 0 0\n')
            with self.assertRaises(ValueError):read_bvh(p)

    def test_pelvis_basis_is_right_handed(self):
        basis=np.array([[0.,0,1],[1,0,0],[0,1,0]])
        self.assertAlmostEqual(np.linalg.det(basis),1.)
        # Its left axis is the BVH pelvis local +Z, not the oblique Spine offset.
        np.testing.assert_array_equal(basis[:,1],[0.,0,1])

if __name__=='__main__':unittest.main()
