"""Optimize ONLY first 20ms PD target, using exact sole mesh in native physics.
Numerical control warm start, not a certificate. No state edits or root forces.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from mujoco import rollout
from scipy.spatial import ConvexHull
from physical_reference_model import build_physical_model,mj
from play_retarget_motion import load_motion

def exact_z(sensor,verts,anchor):
    qw,qx,qy,qz=np.moveaxis(sensor[...,3:7],-1,0)
    rz=np.stack([2*(qx*qz-qw*qy),2*(qy*qz+qw*qx),1-2*(qx*qx+qy*qy)],axis=-1)
    hull=(verts-anchor)[ConvexHull(verts).vertices]
    return sensor[...,2]+np.einsum('bti,vi->btv',rz,hull).min(-1)
def prepare(args):
    root=Path(__file__).resolve().parents[2]
    for path in (args.motion,args.run,args.warm,args.output):
        if path is None:continue
        if not path.resolve().is_relative_to(root) or 'lilin' in str(path.resolve()).lower():raise ValueError('Outside user project')
    if args.output.exists():raise FileExistsError(args.output)
    a,names,dt=load_motion(args.motion)
    refsha=hashlib.sha256(args.motion.read_bytes()).hexdigest()
    m,ids,adr,dofs,feet,verts,anchors,kp,kd,effort=build_physical_model(names,.006,sensors=True,joint_limit_time_constant=.004)
    m.opt.iterations=60
    if args.run is not None:
        manifest=json.loads((args.run/'run_manifest.json').read_text())
        if manifest['input_reference_sha256']!=refsha:raise ValueError('Reference mismatch')
        with np.load(args.run/'rollout.npz',allow_pickle=False) as z:
            initial=z['states'][0].copy();u=z['controls'].copy()
        origin='Matching reference: previous native rollout initial condition'
    else:
        with np.load(args.warm,allow_pickle=False) as z:
            if str(z['reference_sha256'].item())!=refsha:raise ValueError('Warm reference mismatch')
            u=z['controls'].copy()
        # Generate the new reference initial condition, identically to MPC.
        # Old numerical controls are allowed; no old reference states are copied.
        Q=np.tile(m.qpos0,(len(a['time_s']),1))
        Q[:,:3]=a['root_positions_m']
        Q[:,3:7]=a['root_quaternions_xyzw'][:,[3,0,1,2]]
        Q[:,adr]=a['joint_positions_rad']
        v0=np.zeros(m.nv)
        mj.mj_differentiatePos(m,v0,dt,Q[0],Q[1])
        initial=np.r_[0.,Q[0],v0]
        initial[3]-=.00015
        origin='New reference initial condition, same as optimize_reference_sampling.py'
    if u.ndim!=2 or u.shape[1]!=13 or len(u)<10 or not np.isfinite(u).all():
        raise ValueError('Invalid numerical control warm start')
    pool=rollout.Rollout(nthread=4);workers=[mj.MjData(m) for _ in range(4)]
    rng=np.random.default_rng(20261120);mean=u[0].copy();std=np.full(13,.045);best=np.inf;bestu=mean.copy();bestreport=None
    base=u[:10].copy()
    for it in range(10):
        candidates=rng.normal(mean,std,(256,13));candidates[0]=bestu;candidates=np.clip(candidates,u[0]-.18,u[0]+.18)
        commands=np.broadcast_to(base,(len(candidates),)+base.shape).copy();commands[:,0]=candidates
        states,sensors=pool.rollout(m,workers,initial[None],np.repeat(commands,20,axis=1))
        z=np.stack([exact_z(sensors[:,:,j*13:(j+1)*13],verts[j],anchors[j]) for j in range(2)],axis=-1)
        penetration=np.maximum(-z.min((1,2)),0.)
        q=states[:,:,1:1+m.nq];v=states[:,:,1+m.nq:]
        vel=abs(v[:,:,dofs]).max((1,2));lim=np.maximum(m.jnt_range[ids,0]-q[:,:,adr],0)+np.maximum(q[:,:,adr]-m.jnt_range[ids,1],0)
        cost=1e12*np.maximum(penetration-.0015,0)**2+1e5*np.maximum(vel-4.8,0)**2+1e8*lim.max((1,2))**2+20*np.sum((candidates-u[0])**2,axis=1)+300*np.sum((states[:,-1]-states[0,-1])**2,axis=1)
        elite=np.argsort(cost)[:24];mean=candidates[elite].mean(0);std=np.maximum(.005,candidates[elite].std(0))
        if cost[elite[0]]<best:
            k=elite[0];best=cost[k];bestu=candidates[k].copy();bestreport={'predicted_max_mesh_penetration_m':float(penetration[k]),'predicted_max_joint_speed_rad_s':float(vel[k]),'target_delta_rad':(bestu-u[0]).tolist()}
        print(it,bestreport,flush=True)
    with np.load(args.motion.parent/'optimizer_seed.npz',allow_pickle=False) as z:full=z['controls'].copy()
    full[:len(u)]=u;u=full;u[0]=bestu;args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,controls=u,reference_sha256=np.array(refsha),physical_reference=np.array(False),fixed_prefix_steps=np.array(1))
    (args.output.with_suffix('.json')).write_text(json.dumps({'status':'INITIAL_CONTROL_GUESS_REQUIRES_FULL_REPLAY','previous_run':str(args.run) if args.run else None,'numerical_warm':str(args.warm) if args.warm else None,'initialization_origin':origin,'initial_state_modified':False,'root_force_applied':False,**bestreport},indent=2));pool.close()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['motion','output']:p.add_argument('--'+key,type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--run',type=Path);g.add_argument('--warm',type=Path)
    prepare(p.parse_args())
