"""Uniformly retime a measured source candidate; never retime an accepted physics clip.

Interpolating accepted physical states does NOT create a physical motion. This
utility therefore only emits an explicitly unapproved kinematic initializer.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def retime(source: Path, output: Path, factor: float):
    for p in (source,output):
        if 'lilin' in str(p.resolve()).lower(): raise ValueError('Outside user project')
    if output.exists(): raise FileExistsError(output)
    if not .6 <= factor <= 1.5: raise ValueError('Invalid time adaptation')
    with np.load(source,allow_pickle=False) as ar: a={k:ar[k].copy() for k in ar.files}
    if 'dynamics_certified' in a and bool(a['dynamics_certified']):
        raise ValueError('Use original human kinematic source, not certified physical states')
    old=a['time_s'];dt=float(a['sample_dt_s']);new=np.arange(0,old[-1]*factor+1e-9,dt)
    st=new/factor;n=len(old)
    for key,value in list(a.items()):
        if value.ndim==0 or value.shape[0]!=n: continue
        if key=='root_quaternions_xyzw':
            a[key]=Slerp(old,Rotation.from_quat(value))(st).as_quat()
        elif value.ndim==3 and value.shape[1:]==(3,3):
            a[key]=Slerp(old,Rotation.from_matrix(value))(st).as_matrix()
        elif value.ndim==4 and value.shape[2:]==(3,3):
            a[key]=np.stack([Slerp(old,Rotation.from_matrix(value[:,j]))(st).as_matrix() for j in range(value.shape[1])],axis=1)
        elif value.dtype.kind=='b':
            a[key]=value[np.clip(np.searchsorted(old,st),0,n-1)]
        elif value.dtype.kind in 'fiu':
            flat=value.reshape(n,-1)
            a[key]=np.stack([np.interp(st,old,c) for c in flat.T],axis=1).reshape((len(new),)+value.shape[1:])
    a['time_s']=new
    for key in ('joint_velocities_rad_s','root_linear_velocities_world_m_s','root_angular_velocities_world_rad_s'):
        if key in a:a[key]/=factor
    a.update(valid_for_training=np.array(False),valid_for_style_kinematics=np.array(False),
             candidate_requires_dynamics_qualification=np.array(True),retiming_factor=np.array(factor))
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'candidate_motion.npz',**a)
    report={'status':'UNAPPROVED_KINEMATIC_INITIALIZER','source':str(source.resolve()),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'uniform_time_factor':factor,
            'duration_s':float(new[-1]),'source_timestamps_preserved':True,
            'note':'Not a time-scaled physics certificate; full new physical optimization and replay required.'}
    (output/'generation_report.json').write_text(json.dumps(report,indent=2));print(report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--factor',type=float,required=True)
    a=p.parse_args();retime(a.input,a.output,a.factor)
if __name__=='__main__': main()
