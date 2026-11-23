"""Measured BVH landmarks/contacts for a NEW constrained-IK solve, NOT a demo.

Unlike retarget_human_walk.generate, this does not solve IK and makes no joint
speed/pose feasibility claim. It cannot be selected by verified_style_data.
The initializer's joint zeros are numerical guesses, never training targets.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from retarget_human_walk import source_targets, ROOT, DT
from play_retarget_motion import build_model, joint_mapping, MODEL_PATH
from physical_reference_model import mj


def prepare(source: Path, start: float, end: float, time_scale: float, output: Path):
    source=source.resolve();output=output.resolve()
    for p in (source,output):
        if not p.is_relative_to(ROOT) or 'lilin' in str(p).lower():
            raise ValueError('Outside user project')
    if output.exists(): raise FileExistsError(output)
    if not 0 <= start < end or not .6 <= time_scale <= 2.:
        raise ValueError('Invalid source interval/timing')
    names=json.loads((ROOT/'data/motions/robotcupmini/retarget_config.json').read_text())['joint_names']
    model=build_model(MODEL_PATH);data=mj.MjData(model);ids,adr,_=joint_mapping(model,names)
    args=argparse.Namespace(source=source,start=start,end=end,time_scale=time_scale)
    s=source_targets(args,model,data,ids,adr);n=len(s['times'])
    joints=np.zeros((n,13));joints[:,-1]=s['waist']
    quat=Rotation.from_euler('xyz',s['root_rpy']).as_quat()
    a=dict(time_s=s['times'],sample_dt_s=np.array(DT),joint_names=np.array(names),
           joint_positions_rad=joints,joint_velocities_rad_s=np.zeros_like(joints),
           root_positions_m=s['root_target'],root_quaternions_xyzw=quat,
           root_linear_velocities_world_m_s=np.gradient(s['root_target'],DT,axis=0),
           root_angular_velocities_world_rad_s=np.zeros((n,3)),
           foot_targets_world_m=s['feet'],foot_target_rotations=s['foot_rotations'],
           contact_candidates=s['contacts'],foot_names=np.array(['left','right']),
           source_time_s=s['source_times'],source_file=np.array(str(source.relative_to(ROOT))),
           source_motion_sha256=np.array(hashlib.sha256(source.read_bytes()).hexdigest()),
           human_joint_names=np.array(s['human_names']),human_parents=s['human_parents'],
           human_positions_m=s['human_positions'],source_pelvis_rotations=s['human_pelvis_rotations'],
           periodic_reference=np.array(False),valid_for_training=np.array(False),
           valid_for_style_kinematics=np.array(False),initializer_only=np.array(True))
    if (~s['contacts'].any(1)).any():
        raise ValueError('Source contains unsupported frames; choose measured walking, not inferred flight')
    output.mkdir(parents=True)
    np.savez_compressed(output/'candidate_motion.npz',**a)
    report=dict(status='SOURCE_LANDMARK_INITIALIZER_NOT_IK_OR_PHYSICS',source=str(source),
                source_sha256=str(a['source_motion_sha256']),start_s=start,end_s=end,
                time_scale=time_scale,frames=n,duration_s=float(s['times'][-1]),
                joint_arrays_are_numerical_guesses_not_motion=True,valid_for_training=False)
    (output/'generation_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'data/motions/robotcupmini/raw/lafan1/walk1_subject1.bvh')
    p.add_argument('--start',type=float,required=True);p.add_argument('--end',type=float,required=True)
    p.add_argument('--time-scale',type=float,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.source,a.start,a.end,a.time_scale,a.output)
if __name__=='__main__':main()
