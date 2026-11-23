import sys,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference_body_kinematics import torso_world_angular_velocity
from physical_reference_model import build_physical_model,mj

class BodyKinematicsTests(unittest.TestCase):
    def test_world_angular_velocity_matches_native_body_measurement(self):
        names=[s+'_'+j+'_joint' for s in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]+['waist_yaw_joint']
        m,ids,adr,dofs,*_=build_physical_model(names)
        d=mj.MjData(m);rng=np.random.default_rng(11)
        qs=[];vs=[];expected=[]
        for i in range(10):
            d.qpos[:]=m.qpos0;d.qpos[:3]=[0,0,.6]
            d.qpos[3:7]=Rotation.random(random_state=rng).as_quat()[[3,0,1,2]]
            d.qpos[adr]=rng.uniform(-.1,.1,len(names));d.qvel[:]=rng.normal(size=m.nv)
            mj.mj_forward(m,d);out=np.zeros(6)
            mj.mj_objectVelocity(m,d,mj.mjtObj.mjOBJ_BODY,m.body('torso_link').id,out,0)
            qs.append(d.qpos.copy());vs.append(d.qvel.copy());expected.append(out[:3].copy())
        got=torso_world_angular_velocity(np.array(qs),np.array(vs),int(dofs[-1]))
        np.testing.assert_allclose(got,expected,atol=1e-12)
        np.testing.assert_allclose(torso_world_angular_velocity(qs[0],vs[0],int(dofs[-1])),expected[0],atol=1e-12)

if __name__=='__main__':unittest.main()
