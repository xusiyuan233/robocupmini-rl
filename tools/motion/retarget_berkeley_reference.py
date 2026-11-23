"""Prepare measured Berkeley-policy landmarks for the existing constrained IK.
No copying of Berkeley joint indices/angles into this robot. The thirteenth
(waist) joint has no source counterpart and stays neutral, not a fake human sway.
Writes an UNAPPROVED numerical initializer, never an expert demonstration.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from play_retarget_motion import build_model,MODEL_PATH,joint_mapping
from retarget_human_walk import lock_stance,clean_contacts
ROOT=Path(__file__).resolve().parents[2]

def prepare(args):
    if not (0 <= args.start < args.end and .6 <= args.time_scale <= 2 and .4 <= args.stride_scale <= 1 and .48 <= args.root_height <= .56 and .015 <= args.max_clearance <= .05):
        raise ValueError('Invalid robot-source projection parameters')
    for p in [args.source,args.output]:
        if not p.resolve().is_relative_to(ROOT) or 'lilin' in str(p).lower():raise ValueError('Outside user project')
    if args.output.exists():raise FileExistsError(args.output)
    with np.load(args.source,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    times=np.arange(0,(args.end-args.start)*args.time_scale-1e-8,.02);st=args.start+times/args.time_scale
    if st[0]<a['time_s'][0] or st[-1]>a['time_s'][-1]:raise ValueError('Source interval out of range')
    def interp(v):
        flat=np.asarray(v).reshape(len(a['time_s']),-1)
        out=np.stack([np.interp(st,a['time_s'],col) for col in flat.T],axis=1)
        return out.reshape((len(st),)+v.shape[1:])
    q=interp(a['qpos']);feet=interp(a['foot_positions_m']);root=q[:,:3].copy()
    crot=Slerp(a['time_s'],Rotation.from_quat(a['qpos'][:,[4,5,6,3]]))(st)
    travel=root[-1,:2]-root[0,:2];heading=float(np.arctan2(travel[1],travel[0]));align=Rotation.from_euler('z',-heading)
    origin=np.r_[root[0,:2],0.]
    root=align.apply(root-origin);feet=align.apply((feet-origin).reshape(-1,3)).reshape(-1,2,3);crot=align*crot
    contacts=[];clear=[];fr=[]
    nearest=np.argmin(abs(a['time_s'][:,None]-st[None,:]),axis=0)
    for side in range(2):
        contact=clean_contacts(a['contact_load_N'][nearest,side]>10.)
        contacts.append(contact)
        baseline=float(np.median(feet[contact,side,2]));height=np.maximum(0.,feet[:,side,2]-baseline)
        height[contact]=0.;clear.append(np.minimum(height,args.max_clearance))
        # The source foot-site local axes are not the target link axes. Calibrate
        # planted sole orientation rather than copying an arbitrary site frame.
        raw=Slerp(a['time_s'],Rotation.from_matrix(a['foot_rotations'][:,side]))(st)
        mean=(align*raw[contact]).mean();rot=(align*raw)*mean.inv()
        fr.append(rot.as_matrix())
        feet[:,side]=lock_stance(times,feet[:,side],contact)
    contacts=np.asarray(contacts).T
    if (~contacts.any(1)).any():raise ValueError('Source support was lost after sampling')
    # Fit spatial reach conservatively, then let IK/contact/dynamics do the work.
    root[:,:2]*=args.stride_scale;feet[:,:,:2]*=args.stride_scale
    root[:,2]=args.root_height+.35*(root[:,2]-root[:,2].mean())
    names=json.loads((ROOT/'data/motions/robotcupmini/retarget_config.json').read_text())['joint_names']
    m=build_model(MODEL_PATH);_,adr,_=joint_mapping(m,names)
    joints=np.tile(m.qpos0[adr],(len(times),1));joints[:,3]=.4;joints[:,9]=.4;joints[:,[0,6]]=-.2;joints[:,[4,10]]=-.2
    rp=crot.as_euler('xyz');rp[:,2]-=np.mean(rp[:,2]);rp[:,0]-=np.mean(rp[:,0]);rp[:,1]-=np.mean(rp[:,1])
    quat=Rotation.from_euler('xyz',rp).as_quat()
    out=dict(time_s=times,sample_dt_s=np.array(.02),joint_names=np.array(names),joint_positions_rad=joints,joint_velocities_rad_s=np.zeros_like(joints),root_positions_m=root,root_quaternions_xyzw=quat,root_linear_velocities_world_m_s=np.gradient(root,.02,axis=0),root_angular_velocities_world_rad_s=np.zeros((len(times),3)),foot_targets_world_m=feet,foot_target_rotations=np.stack(fr,axis=1),contact_candidates=contacts,source_sole_clearance_m=np.asarray(clear).T,source_time_s=st,source_file=np.array(str(args.source.relative_to(ROOT))),source_kind=np.array('berkeley_published_policy_rollout'),source_policy_sha256=a['source_policy_sha256'],source_robot_root_positions_m=interp(a['qpos'][:,:3]),source_robot_root_quaternions_xyzw=Slerp(a['time_s'],Rotation.from_quat(a['qpos'][:,[4,5,6,3]]))(st).as_quat(),source_robot_joint_positions_rad=interp(a['qpos'][:,7:]),source_robot_joint_names=a['joint_names'],source_robot_torso_rotations=crot.as_matrix(),source_alignment_yaw_rad=np.array(-heading),periodic_reference=np.array(False),initializer_only=np.array(True),valid_for_training=np.array(False),valid_for_style_kinematics=np.array(False))
    args.output.mkdir(parents=True);np.savez_compressed(args.output/'candidate_motion.npz',**out)
    report={'status':'ROBOT_SOURCE_INITIALIZER_ONLY_NOT_IK_OR_DYNAMICS','source':str(args.source),'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),'source_interval_s':[args.start,args.end],'time_scale':args.time_scale,'stride_scale':args.stride_scale,'root_height_m':args.root_height,'frames':len(times),'duration_s':float(times[-1]),'waist_rule':'No counterpart in source; neutral waist','not_the_original_video_checkpoint':True,'maximum_planned_sole_clearance_m':args.max_clearance,'valid_for_training':False}
    (args.output/'generation_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=float,default=6.);p.add_argument('--end',type=float,default=13.);p.add_argument('--time-scale',type=float,default=1.2);p.add_argument('--stride-scale',type=float,default=.65);p.add_argument('--root-height',type=float,default=.54);p.add_argument('--max-clearance',type=float,default=.04);prepare(p.parse_args())
