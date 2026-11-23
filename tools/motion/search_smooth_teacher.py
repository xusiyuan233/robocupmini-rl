"""Search smoothed PD COMMANDS then re-integrate unmodified free-root physics.

Never smooth/overwrite saved poses. Every candidate starts from the same initial
state, runs native PD at 1kHz, and resets only numerical solver warm starts each
20ms (same as the independent auditor). Output candidates are NOT certificates.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from mujoco import rollout
from physical_reference_model import build_physical_model,oriented_box_min_z,contact_speed_bound,mj
from play_retarget_motion import load_motion,PROJECT_ROOT
from reference_coordination_quality import measure_motion
from reference_contact_quality import interior_mask


def search(run:Path,output:Path):
    for p in (run,output):
        if not p.resolve().is_relative_to(PROJECT_ROOT) or 'lilin' in str(p.resolve()).lower():raise ValueError('Outside project')
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((run/'run_manifest.json').read_text());reference=Path(manifest['input_reference']).resolve()
    if not reference.is_relative_to(PROJECT_ROOT) or 'lilin' in str(reference).lower():raise ValueError('Reference outside project')
    if hashlib.sha256(reference.read_bytes()).hexdigest()!=manifest['input_reference_sha256']:raise ValueError('Reference changed')
    a,names,dt=load_motion(reference)
    with np.load(run/'rollout.npz',allow_pickle=False) as z:initial=z['states'][0].copy();original=z['controls'].copy()
    m,ids,adr,dofs,bids,verts,anchors,*_=build_physical_model(names,manifest['contact_time_constant_s'],sensors=True,joint_limit_time_constant=manifest['joint_limit_time_constant_s'])
    specs=[(0.,'all')]+[(s,g) for s in (.35,.6,1.,1.4,2.,3.) for g in ('all','legs','pitch')]
    controls=[]
    for sigma,group in specs:
        u=original.copy();idx=list(range(13)) if group=='all' else list(range(12)) if group=='legs' else [0,3,4,6,9,10]
        if sigma:u[:,idx]=gaussian_filter1d(u[:,idx],sigma,axis=0,mode='nearest')
        controls.append(u)
    U=np.asarray(controls);B,N,_=U.shape;S=np.broadcast_to(initial,(B,len(initial))).copy();history=[S.copy()];sensors=[];geometry=np.zeros(B);contact_cost=np.zeros(B);invalid=np.zeros(B,dtype=bool)
    sm=interior_mask(a['contact_candidates'],5);sw=interior_mask(~a['contact_candidates'],5)
    workers=[mj.MjData(m) for _ in range(6)]
    print('Testing',B,'command filters,',N*dt,'s, unchanged physics',flush=True)
    with rollout.Rollout(nthread=6) as pool:
        for k in range(N):
            st,se=pool.rollout(m,workers,S,np.repeat(U[:,k:k+1,:],20,axis=1))
            invalid|=~np.isfinite(st).all(axis=(1,2))|np.any(abs(st[:,:,0]-(k*dt+np.arange(1,21)*.001))>1e-8,axis=1)
            S=st[:,-1].copy();history.append(S.copy());sensors.append(se[:,-1].copy())
            for j in range(2):
                sd=se[:,:,j*13:(j+1)*13];minimum=oriented_box_min_z(sd[:,:,:3],sd[:,:,3:7],(verts[j]-anchors[j]).min(0),(verts[j]-anchors[j]).max(0))
                geometry+=np.maximum(-.002-minimum,0).max(1)**2*1e7
                if sw[min(k,len(sw)-1),j]:geometry+=np.maximum(.003-minimum,0).max(1)**2*1e7
                if sm[min(k,len(sm)-1),j]:contact_cost+=np.maximum(contact_speed_bound(sd[:,:,7:10],sd[:,:,10:13])-.025,0).mean(1)**2*1e5
            if k%100==0:print('simulation',round(k*dt,2),flush=True)
    states=np.stack(history,axis=1);sens=np.stack(sensors,axis=1);reports=[]
    output.mkdir(parents=True)
    for c,(sigma,group) in enumerate(specs):
        Q=states[c,:,1:21];V=states[c,:,21:];b={k:v.copy() for k,v in a.items()};initial_feet=a['foot_targets_world_m'][0]
        b.update(time_s=states[c,:,0],root_positions_m=Q[:,:3],root_quaternions_xyzw=Q[:,[4,5,6,3]],joint_positions_rad=Q[:,adr],joint_velocities_rad_s=V[:,dofs],foot_targets_world_m=np.concatenate([initial_feet[None],np.stack([sens[c,:,:3],sens[c,:,13:16]],axis=1)]))
        measured=measure_motion(b);t=measured['torso'];sway=measured['detrended_pelvis_sway_range_m'];width=measured['foot_width_p05_median_p95_m'];lo,hi=m.jnt_range[ids].T
        score=(t['highfreq_rms_deg']/.2)**2+(sway/.055)**2+max(t['acceleration_p95_deg_s2']/400-1,0)**2*10+geometry[c]+contact_cost[c]
        score+=1e7*(np.maximum(.4-Q[:,2],0)**2).sum()+1e6*(np.maximum(lo-Q[:,adr],0)**2+np.maximum(Q[:,adr]-hi,0)**2).sum()
        tilt=np.arccos(np.clip(Rotation.from_quat(Q[:,[4,5,6,3]]).as_matrix()[:,2,2],-1,1))
        score+=1e5*(np.maximum(tilt-np.deg2rad(25),0)**2).sum()
        score+=100*max(t['highfreq_rms_deg']/.2-1,0)**2+100*max(sway/.055-1,0)**2
        score+=1e5*max(.08-(Q[-1,0]-Q[0,0])/(N*dt),0)**2
        score+=1e5*(np.maximum(np.abs(V[:,dofs])-5,0)**2).sum()+1e6*max(.15-width[1],0)**2
        if invalid[c]:score=1e15
        spec=dict(index=c,filter_sigma_frames=sigma,group=group,score=float(score),measured=measured,numerical_invalid=bool(invalid[c]))
        reports.append(spec)
        dest=output/f'case_{c:02d}';dest.mkdir()
        nm={**manifest,'method':'Command filter search; full new native-PD physical integration, NOT filtered states','parameters':dict(sigma_frames=sigma,group=group),
            'optimizer_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'status':'REQUIRES_INDEPENDENT_FULL_AUDIT',
            'previous_control_run':str(run.resolve()),'previous_rollout_sha256':hashlib.sha256((run/'rollout.npz').read_bytes()).hexdigest()}
        (dest/'run_manifest.json').write_text(json.dumps(nm,indent=2));np.savez_compressed(dest/'rollout.npz',states=states[c],controls=U[c])
    reports.sort(key=lambda r:r['score']);(output/'search_report.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports[:4],indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();search(a.run,a.output)
if __name__=='__main__':main()
