"""Create a nominal PD-target warm start; explicitly NOT a physical demonstration.

Full-body inverse rigid dynamics with approximate foot point forces. Failed
force allocations are logged, not presented as feasibility proof. The separate
MPC and actual 1 kHz replay are still mandatory. Only NumPy/SciPy/MuJoCo needed.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from typing import Any,cast
import numpy as np
from scipy.optimize import minimize,LinearConstraint
from physical_reference_model import build_physical_model,mj
from play_retarget_motion import load_motion
# SciPy runtime accepts vector bounds; its installed stub types ub too narrowly.
linear_constraint=cast(Any,LinearConstraint)


def generate(motion:Path,output:Path):
    if output.exists():raise FileExistsError('Refusing overwrite')
    if 'lilin' in str(motion.resolve()).lower() or 'lilin' in str(output.resolve()).lower():raise ValueError('Outside project scope')
    a,names,dt=load_motion(motion)
    m,ids,adr,dofs,bids,vertices,anchors,kp,kd,effort=build_physical_model(names)
    n=len(a['time_s']);Q=np.tile(m.qpos0,(n,1));Q[:,:3]=a['root_positions_m'];Q[:,3:7]=a['root_quaternions_xyzw'][:,[3,0,1,2]];Q[:,adr]=a['joint_positions_rad']
    V=np.zeros((n,m.nv))
    for i in range(n):
        lo,hi=max(0,i-1),min(n-1,i+1);mj.mj_differentiatePos(m,V[i],(hi-lo)*dt,Q[lo],Q[hi])
    A=np.gradient(V,dt,axis=0);d=mj.MjData(m);M=np.zeros((m.nv,m.nv));jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);FF=[];failures=[];scale=float(m.body_mass.sum()*9.81)
    for i in range(n):
        d.qpos[:]=Q[i];d.qvel[:]=V[i];mj.mj_forward(m,d);mj.mj_fullM(m,d,M)
        required=M@A[i]+d.qfrc_bias-d.qfrc_passive;columns=[]
        for side in np.flatnonzero(a['contact_candidates'][i]):
            b=bids[side];r=d.xmat[b].reshape(3,3);v=vertices[side];z=v[:,2].min()
            for x in [v[:,0].min()+.005,v[:,0].max()-.005]:
                for y in [v[:,1].min()+.005,v[:,1].max()-.005]:
                    point=d.xpos[b]+r@np.array([x,y,z]);mj.mj_jac(m,d,jp,jr,point,b);columns.append(jp.T.copy())
        T=np.hstack(columns)*scale;nf=T.shape[1];Tj=T[dofs]
        H=Tj.T@np.diag(1/kp**2)@Tj+np.eye(nf)*1.;c=-Tj.T@(required[dofs]/kp**2)
        E=T[:6]/scale;ee=required[:6]/scale
        G=[];upper=[]
        for j in range(nf//3):
            row=np.zeros(nf);row[j*3+2]=-1;G.append(row);upper.append(0.)
            for axis in [0,1]:
                for sign in [-1,1]:
                    row=np.zeros(nf);row[3*j+axis]=sign;row[3*j+2]=-.8/np.sqrt(2);G.append(row);upper.append(0.)
        G=np.asarray(G);f0=np.tile([0,0,1/(nf//3)],nf//3)
        constraints=[linear_constraint(E,ee,ee),linear_constraint(G,-np.inf,np.array(upper)),linear_constraint(Tj/effort[:,None],(required[dofs]-effort)/effort,(required[dofs]+effort)/effort)]
        res=minimize(lambda f:.5*f@H@f+c@f,f0,jac=lambda f:H@f+c,method='SLSQP',constraints=constraints,options={'maxiter':150,'ftol':1e-9})
        valid=bool(res.success and np.max(abs(E@res.x-ee))<1e-5)
        if not valid:
            failures.append(i)
            # A failed force QP is allowed as a NUMERICAL INITIALIZER only.
            # A root-wrench slack is never applied to physical simulation.
            H2=H+1e4*E.T@E;c2=c-1e4*E.T@ee
            res=minimize(lambda f:.5*f@H2@f+c2@f,f0,jac=lambda f:H2@f+c2,method='SLSQP',constraints=constraints[1:2],options={'maxiter':150,'ftol':1e-9})
        FF.append(np.clip(required[dofs]-Tj@res.x,-effort,effort))
    FF=np.asarray(FF);U=Q[:,adr]+(FF+kd*V[:,dofs])/kp
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,controls=U,torque_seed_Nm=FF,reference_sha256=np.array(hashlib.sha256(motion.read_bytes()).hexdigest()),physical_reference=np.array(False),force_qp_failed_frames=np.asarray(failures,dtype=int))
    output.with_suffix('.json').write_text(json.dumps({'status':'NUMERICAL_WARM_START_ONLY_NOT_A_DEMONSTRATION','input':str(motion.resolve()),'force_qp_failed_frames':failures,'note':'No physical rollout or AMP approval implied.'},indent=2))
    print('SEED',output,'approximate-force failures',failures,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();generate(args.motion,args.output)

if __name__=='__main__':main()
