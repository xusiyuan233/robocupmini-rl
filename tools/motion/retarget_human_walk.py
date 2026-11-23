"""Source-driven, morphology-aware BVH -> RobotcupMini motion retargeting.

Not a controller or an AMP implementation. No sine-wave gait, COM replacement or
forced periodicity. Stores source timestamps and human landmarks for inspection.
Contact is inferred from BVH geometry (not measured contact forces). Old assets,
URDF, training tasks, policies and neighbouring projects are never modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast
import mujoco as _mj
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from bvh_motion import read_bvh, anatomical_frame
from play_retarget_motion import build_model, joint_mapping, MODEL_PATH
from audit_style_pretraining import mesh_points, runs
from reference_contact_quality import source_swing_clearance

mj = cast(Any, _mj)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT/'data/motions/robotcupmini/retargeted/walk1_candidate_v6_clearance'
DT = .02
SIDES = ('left', 'right')
PELVIS_ROLL_GAIN = .5  # robot ankle roll is only +/-15deg hard limit
REACH_MARGIN_M = .045  # avoids straight-leg singularity during long stance
MIN_KNEE_RAD = .18  # about 10deg; do not snap through a straight knee
SWING_ROLL_CAP_DEG = 8.
SWING_PITCH_CAP_DEG = 20.


def smooth(x, window=7):
    return savgol_filter(x, window, 3, axis=0)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def clean_contacts(mask):
    """Remove <100ms islands and fill <60ms internal holes, retaining clip edges."""
    result = mask.copy()
    for start, stop in runs(~result):
        if start > 0 and stop < len(result) and stop-start <= 2:
            result[start:stop] = True
    for start, stop in runs(result):
        if start > 0 and stop < len(result) and stop-start < 5:
            result[start:stop] = False
    return result


def lock_stance(times, raw, mask):
    """Keep measured swing shape; Hermite correction only removes stance drift.

    Zero endpoint slopes join stance and swing. No synthetic periodic footpath.
    """
    intervals = runs(mask)
    if not intervals:
        raise ValueError('No reliable stance in selected source')
    velocity = np.gradient(raw, times, axis=0)
    result = raw.copy()
    anchors = [np.median(raw[a:b], axis=0) for a,b in intervals]
    for anchor in anchors:
        anchor[2] = 0
    gaps = []
    for i in range(len(intervals)-1):
        a,b = intervals[i][1]-1, intervals[i+1][0]
        gaps.append((a,b,anchors[i]-raw[a],anchors[i+1]-raw[b],-velocity[a],-velocity[b]))
    a = intervals[0][0]
    if a:
        correction = anchors[0]-raw[a]
        gaps.append((0,a,correction,correction,np.zeros(3),-velocity[a]))
    a = intervals[-1][1]-1
    if a < len(times)-1:
        correction = anchors[-1]-raw[a]
        gaps.append((a,len(times)-1,correction,correction,-velocity[a],np.zeros(3)))
    for a,b,da,db,va,vb in gaps:
        correction = CubicHermiteSpline(times[[a,b]],np.stack([da,db]),np.stack([va,vb]))
        result[a:b+1] += correction(times[a:b+1])
    for (a,b), anchor in zip(intervals,anchors):
        result[a:b] = anchor
    # Marker noise near a ground crossing is not copied as a negative clearance.
    result[:,2] = np.maximum(result[:,2], 0)
    return result


def source_targets(args, model, data, ids, addresses) -> dict[str, Any]:
    human = read_bvh(args.source)
    times = np.arange(0, (args.end-args.start)*args.time_scale-1e-8, DT)
    source_times = args.start + times/args.time_scale
    positions, rotations = human.sample(source_times)
    # Y-up BVH -> Z-up; align only the whole clip's travel direction once.
    basis = np.array([[1.,0,0],[0,0,-1],[0,1,0]])
    positions = np.asarray(positions @ basis.T, dtype=float)
    positions[...,:2] -= positions[0,0,:2].copy()
    travel = positions[-1,0,:2]-positions[0,0,:2]
    if np.linalg.norm(travel) < .1:
        raise ValueError('Source is not a forward walking clip')
    heading = float(np.arctan2(travel[1],travel[0]))
    align = Rotation.from_euler('z',-heading).as_matrix()
    positions = positions @ align.T
    rotations = align @ basis @ rotations
    positions = smooth(positions)
    at = lambda name: positions[:,human.names.index(name)]
    # The BVH pelvis local +X is anatomical UP, +Z is LEFT. The Spine offset
    # is oblique and must NOT be mistaken for pelvis vertical (old ~19deg bias).
    pelvis_basis = np.array([[0.,0,1],[1,0,0],[0,1,0]])
    pelvis = rotations[:,0] @ pelvis_basis
    root_rpy = smooth(np.unwrap(Rotation.from_matrix(pelvis).as_euler('xyz'),axis=0))
    human_pelvis_rotations = pelvis.copy()
    root_rpy[:,0] *= PELVIS_ROLL_GAIN
    torso = anatomical_frame(at('LeftShoulder'),at('RightShoulder'),at('Neck')-at('Spine2'))
    relative = np.swapaxes(pelvis,1,2) @ torso
    # Extract the reachable z-twist, not a fictitious multi-axis spine joint.
    twist = np.arctan2(relative[:,1,0]-relative[:,0,1],relative[:,0,0]+relative[:,1,1])
    waist = smooth(np.unwrap(twist))
    if np.any((waist<model.jnt_range[ids[12],0]) | (waist>model.jnt_range[ids[12],1])):
        raise ValueError('Measured waist twist exceeds robot joint limits')
    data.qpos[:] = model.qpos0
    mj.mj_forward(model,data)
    foot_ids = [model.body(s+'_ankle_roll_link').id for s in SIDES]
    ankle_ids = [model.body(s+'_ankle_pitch_link').id for s in SIDES]
    knee_ids = [model.body(s+'_knee_link').id for s in SIDES]
    hip_ids = [model.body(s+'_hip_yaw_link').id for s in SIDES]
    pelvis_hip_ids = [model.body(s+'_hip_pitch_link').id for s in SIDES]
    vertices, anchors = [], []
    for bid in foot_ids:
        v = np.concatenate([mesh_points(model,g) for g in range(model.ngeom)
                            if model.geom_bodyid[g]==bid and model.geom_contype[g]
                            and model.geom_type[g]==mj.mjtGeom.mjGEOM_MESH])
        vertices.append(v)
        anchors.append(np.r_[(v.min(0)[:2]+v.max(0)[:2])/2,v[:,2].min()])
    # Overall skeleton reach sets the travel scale. Each leg segment is
    # separately remapped below using its own actual robot length.
    human_reach = np.mean([np.linalg.norm(at(s+'UpLeg')-at(s+'Leg'),axis=1).mean()
                           + np.linalg.norm(at(s+'Leg')-at(s+'Foot'),axis=1).mean()
                           for s in ['Left','Right']])
    robot_reach = np.mean([np.linalg.norm(data.xpos[k]-data.xpos[h]) +
                           np.linalg.norm(data.xpos[a]-data.xpos[k])
                           for h,k,a in zip(pelvis_hip_ids,knee_ids,ankle_ids)])
    scale = float(robot_reach/human_reach)
    root_target = scale*at('Hips').copy()
    # Align median pelvis-to-ankle height to robot morphology, leaving the
    # human's original vertical oscillation intact. This initial bind offset
    # cancels when ground is fitted from measured support samples below.
    straight_height = np.mean([-(data.xpos[f]+data.xmat[f].reshape(3,3)@a)[2]
                               for f,a in zip(foot_ids,anchors)])
    root_target[:,2] += straight_height-.025-np.median(root_target[:,2])
    contacts = []; foot_targets = []; foot_rotations = []; knee_planes = []; flexions=[]
    calibration=[]
    for j,side in enumerate(['Left','Right']):
        ankle, toe, hip, knee = [at(side+x) for x in ['Foot','Toe','UpLeg','Leg']]
        speed = np.linalg.norm(np.gradient(ankle,times,axis=0)[:,:2],axis=1)
        floor = np.percentile(ankle[:,2],10)
        contact = clean_contacts((speed < .25/args.time_scale) & (ankle[:,2] < floor+.035))
        if contact.sum()<10:
            raise ValueError('Insufficient reliable human stance samples')
        # Calibrate human foot orientation against planted frames. Roll/pitch
        # deviations follow measured rotations; remove ankle->toe bone tilt,
        # which is not the same thing as the shoe sole angle.
        hr = rotations[:,human.names.index(side+'Foot')]
        mid = contact & (speed<.06/args.time_scale)
        ref = Rotation.from_matrix(hr[mid]).mean().as_matrix()
        toe_vec = toe-ankle
        yaw0 = float(np.median(np.arctan2(toe_vec[mid,1],toe_vec[mid,0])))
        fr = hr @ ref.T @ Rotation.from_euler('z',yaw0).as_matrix()
        quat = Rotation.from_matrix(fr).as_quat()
        foot_twist = np.unwrap(2*np.arctan2(quat[:,2],quat[:,3]))
        swing = (Rotation.from_euler('z',-foot_twist)*Rotation.from_matrix(fr)).as_rotvec()
        # Swing/twist decomposition avoids Euler yaw flips when the human foot
        # approaches vertical during toe-off.
        frpy = smooth(np.column_stack([swing[:,:2],foot_twist]))
        # Inferred flat-support segments: keep a fixed sole pose. Over swings,
        # source rotations are smoothly blended in/out (no invented phase wave).
        targets_rpy = frpy.copy()
        weight = np.ones(len(times))
        for a,b in runs(contact):
            targets_rpy[a:b] = [0,0,float(np.median(frpy[a:b,2]))]
            weight[a:b] = 0
        for a,b in runs(~contact):
            u = np.arange(b-a)*DT
            gate = np.ones(b-a)
            if a>0:
                z=np.clip(u/(.14*args.time_scale),0,1);gate*=z**3*(10-15*z+6*z*z)
            if b<len(times):
                z=np.clip((b-a-1-np.arange(b-a))*DT/(.14*args.time_scale),0,1);gate*=z**3*(10-15*z+6*z*z)
            weight[a:b] = gate
        # Robot has a rigid sole and no toe joint. Compress large source toe-off
        # rotation smoothly; this documented morphology projection is not a
        # learned/physically certified contact model.
        targets_rpy[:,0] = np.deg2rad(SWING_ROLL_CAP_DEG)*np.tanh(frpy[:,0]/np.deg2rad(SWING_ROLL_CAP_DEG))*weight
        targets_rpy[:,1] = np.deg2rad(SWING_PITCH_CAP_DEG)*np.tanh(frpy[:,1]/np.deg2rad(SWING_PITCH_CAP_DEG))*weight
        # Interpolate stance yaw corrections across swings, then retain the
        # measured swing yaw only when the foot is clear of the ground.
        targets_rpy[:,2] = np.interp(times,times[contact],targets_rpy[contact,2])*(1-weight)+frpy[:,2]*weight
        foot_rot = Rotation.from_euler('xyz',targets_rpy).as_matrix()
        # Match each segment's measured DIRECTION using the robot's actual
        # segment length. Uniform whole-body scaling is wrong here: this robot
        # has a much shorter thigh relative to its shank than the human does.
        upper_length=np.linalg.norm(data.xpos[knee_ids[j]]-data.xpos[hip_ids[j]])
        lower_length=np.linalg.norm(data.xpos[ankle_ids[j]]-data.xpos[knee_ids[j]])
        hip_offset=data.xpos[hip_ids[j]].copy()
        sole_offset=data.xpos[foot_ids[j]]+data.xmat[foot_ids[j]].reshape(3,3)@anchors[j]-data.xpos[ankle_ids[j]]
        root_rotations=Rotation.from_euler('xyz',root_rpy).as_matrix()
        foot=(root_target + np.einsum('tij,j->ti',root_rotations,hip_offset)
              + upper_length*unit(knee-hip) + lower_length*unit(ankle-knee)
              + np.einsum('tij,j->ti',foot_rot,sole_offset))
        upper = knee-hip; lower=ankle-knee
        plane=unit(np.cross(upper,lower))
        flex=np.arccos(np.clip(np.sum(unit(upper)*unit(lower),axis=1),-1,1))
        contacts.append(contact);foot_targets.append(foot);foot_rotations.append(foot_rot)
        knee_planes.append(plane);flexions.append(flex)
        calibration.append(dict(side=side,source_stance_baseline_m=float(np.median(ankle[contact,2])),
                                stance_frames=int(contact.sum()),human_foot_bind_matrix=ref.tolist(),
                                stance_intervals_frames=[list(map(int,x)) for x in runs(contact)]))
    # Fit a single vertical bind offset from all observed support samples;
    # retain measured root height variation, not a chosen constant crouch.
    contact_array=np.stack(contacts,axis=1)
    foot_array=np.stack(foot_targets,axis=1)
    ground=float(np.median(foot_array[:,:,2][contact_array]))
    root_target[:,2]-=ground+REACH_MARGIN_M  # source oscillation retained
    foot_array[:,:,2]-=ground
    for j in range(2):
        foot_array[:,j]=lock_stance(times,foot_array[:,j],contact_array[:,j])
        rotated=np.einsum('tij,vj->tvi',foot_rotations[j],vertices[j]-anchors[j])
        # A human ankle/toe marker is not this robot's rigid sole. The old
        # max(-lowest, 0) correction left moving swing soles ON the floor.
        # Use measured human lift, then offset the entire transformed collision
        # mesh above it. Support intervals remain exactly grounded.
        ankle_height = positions[:,human.names.index(('Left' if j == 0 else 'Right')+'Foot'),2]
        clearance = source_swing_clearance(times, ankle_height, contact_array[:,j],
                                          scale, blend_s=.12*args.time_scale)
        foot_array[:,j,2] = clearance - rotated[:,:,2].min(axis=1)
    foot_targets=[foot_array[:,0],foot_array[:,1]]
    return dict(times=times,source_times=source_times,human_names=human.names,human_parents=human.parents,
                human_positions=positions,human_rotations=rotations,human_pelvis_rotations=human_pelvis_rotations,root_target=root_target,
                root_rpy=root_rpy,waist=waist,contacts=np.stack(contacts,axis=1),
                feet=np.stack(foot_targets,axis=1),foot_rotations=np.stack(foot_rotations,axis=1),
                knee_planes=np.stack(knee_planes,axis=1),flexions=np.stack(flexions,axis=1),
                foot_ids=foot_ids,knee_ids=knee_ids,hip_ids=hip_ids,vertices=vertices,anchors=anchors,
                scale=scale,heading=heading,calibration=calibration)


def generate(args):
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    cfg=json.loads((ROOT/'data/motions/robotcupmini/retarget_config.json').read_text())
    names=cfg['joint_names'];model=build_model(MODEL_PATH);data=mj.MjData(model)
    ids,addresses,root=joint_mapping(model,names)
    if root != 0:
        raise ValueError('Expected root at qpos[0]')
    s=source_targets(args,model,data,ids,addresses)
    n=len(s['times']);hard=model.jnt_range[ids];mid=hard.mean(1);half=np.diff(hard,axis=1).ravel()/2
    lo=mid-.9*half+.005;hi=mid+.9*half-.005;lo[[3,9]]=np.maximum(lo[[3,9]],MIN_KNEE_RAD)
    initial=np.r_[[-.15,0,0,.3,-.15,0]*2, s['root_target'][0],np.zeros(3)]
    qpos=[];checks=[]
    prev=None;prevprev=None
    for i in range(n):
        target_r=Rotation.from_euler('xyz',s['root_rpy'][i])
        root_target=s['root_target'][i]
        bounds_lo=np.r_[lo[:12],root_target-[.025,.025,.035],[-.055]*3]
        bounds_hi=np.r_[hi[:12],root_target+[.025,.025,.035],[.055]*3]
        initial=np.clip(initial,bounds_lo+1e-8,bounds_hi-1e-8)
        pred=initial[:12].copy() if prev is None else prev if prevprev is None else 2*prev-prevprev
        def set_pose(v):
            data.qpos[:]=model.qpos0
            data.qpos[:3]=v[12:15]
            r=target_r*Rotation.from_rotvec(v[15:18])
            data.qpos[3:7]=r.as_quat()[[3,0,1,2]]
            data.qpos[addresses[:12]]=v[:12]
            data.qpos[addresses[12]]=s['waist'][i]
            mj.mj_forward(model,data)
        def residual(v):
            set_pose(v);error=[]
            for j,bid in enumerate(s['foot_ids']):
                r=data.xmat[bid].reshape(3,3)
                foot=data.xpos[bid]+r@s['anchors'][j]
                error.extend((foot-s['feet'][i,j])/.0007)
                error.extend(Rotation.from_matrix(s['foot_rotations'][i,j].T@r).as_rotvec()/.006)
                # Knee stays in the human's measured bending plane, not merely
                # wherever a foot-only IK solution happens to put it.
                knee=data.xpos[s['knee_ids'][j]];hip=data.xpos[s['hip_ids'][j]]
                error.append(np.dot(knee-hip,s['knee_planes'][i,j])/.035)
                error.append((v[3+6*j]-s['flexions'][i,j])/.8)
            error.extend((v[12:15]-root_target)/.012)
            error.extend(v[15:18]/.025)
            error.extend((v[:12]-pred)/.35)
            return np.asarray(error)
        result=least_squares(residual,initial,bounds=(bounds_lo,bounds_hi),max_nfev=65,
                             ftol=1e-8,xtol=1e-8,gtol=1e-8)
        set_pose(result.x);initial=result.x.copy();qpos.append(data.qpos.copy())
        poserr=[];roterr=[]
        for j,bid in enumerate(s['foot_ids']):
            r=data.xmat[bid].reshape(3,3)
            poserr.append(float(np.linalg.norm(data.xpos[bid]+r@s['anchors'][j]-s['feet'][i,j])))
            roterr.append(float(Rotation.from_matrix(s['foot_rotations'][i,j].T@r).magnitude()))
        checks.append(dict(frame=i,success=bool(result.success),foot_position_error_m=poserr,
                           foot_rotation_error_rad=roterr,root_correction_m=(result.x[12:15]-root_target).tolist()))
        prevprev=prev;prev=result.x[:12].copy()
        if i%50==0:print(f'IK {i}/{n}: foot error {max(poserr)*1000:.3f} mm',flush=True)
        if not result.success or max(poserr)>.004 or max(roterr)>.025:
            raise RuntimeError(f'IK failed quality gate at {i}: {checks[-1]}; joints_deg={np.rad2deg(result.x[:12]).round(2)}; delta_rpy_deg={np.rad2deg(result.x[15:18])}')
    Q=np.asarray(qpos);V=np.zeros((n,model.nv))
    for i in range(n):
        a,b=max(0,i-1),min(n-1,i+1)
        mj.mj_differentiatePos(model,V[i],(b-a)*DT,Q[a],Q[b])
    joint_velocity=V[:,model.jnt_dofadr[ids]]
    peak=float(abs(joint_velocity).max())
    if args.diagnostics is not None:
        np.savez_compressed(args.diagnostics,Q=Q,V=V,feet=s['feet'],contacts=s['contacts'],root_target=s['root_target'],foot_rotations=s['foot_rotations'],source_times=s['source_times'])
    if peak>4.7:
        raise RuntimeError(f'Joint speed {peak:.3f} exceeds 4.7rad/s. Retarget again with --time-scale >= {args.time_scale*peak/4.4:.3f}; do not clip poses.')
    sha=hashlib.sha256(args.source.read_bytes()).hexdigest()
    arrays=dict(time_s=s['times'],sample_dt_s=np.array(DT),joint_names=np.asarray(names),
        joint_positions_rad=Q[:,addresses],joint_velocities_rad_s=joint_velocity,
        root_positions_m=Q[:,:3],root_quaternions_xyzw=Q[:,[4,5,6,3]],
        root_linear_velocities_world_m_s=V[:,:3],
        root_angular_velocities_world_rad_s=Rotation.from_quat(Q[:,[4,5,6,3]]).apply(V[:,3:6]),
        root_rpy_rad=Rotation.from_quat(Q[:,[4,5,6,3]]).as_euler('xyz'),
        foot_targets_world_m=s['feet'],foot_target_rotations=s['foot_rotations'],contact_candidates=s['contacts'],
        foot_names=np.asarray(['left_ankle_roll_link','right_ankle_roll_link']),
        source_time_s=s['source_times'],source_file=np.array(str(args.source.relative_to(ROOT))),
        source_motion_sha256=np.array(sha),human_joint_names=np.asarray(s['human_names']),
        human_parents=s['human_parents'],human_positions_m=s['human_positions'],
        source_pelvis_rotations=s['human_pelvis_rotations'],
        robot_root_target_rotations=Rotation.from_euler('xyz',s['root_rpy']).as_matrix(),
        source_root_target_m=s['root_target'],source_waist_target_rad=s['waist'],
        periodic_reference=np.array(False),valid_for_style_kinematics=np.array(True),
        valid_for_training=np.array(False))
    report=dict(method='measured BVH pelvis/waist, source swing trajectories and timing, constrained robot IK',
        source=str(args.source),source_sha256=sha,source_start_s=args.start,source_end_s=args.end,
        source_time_scale=args.time_scale,spatial_scale=s['scale'],aligned_heading_rad=s['heading'],
        morphology_projection=dict(pelvis_roll_gain=PELVIS_ROLL_GAIN,reach_margin_m=REACH_MARGIN_M,
             min_knee_rad=MIN_KNEE_RAD,swing_roll_cap_deg=SWING_ROLL_CAP_DEG,swing_pitch_cap_deg=SWING_PITCH_CAP_DEG,
             separate_thigh_and_shank_scaling=True,hip_landmark='hip_yaw_link',contact_swing_blend_source_s=.14,
             rigid_sole_clearance='source ankle lift + smooth minimum interior clearance',
             minimum_interior_clearance_m=.015,clearance_blend_source_s=.12),
        frame_count=n,sample_dt_s=DT,duration_s=float(s['times'][-1]),
        mean_forward_speed_m_s=float((Q[-1,0]-Q[0,0])/s['times'][-1]),
        root_lateral_range_m=float(np.ptp(Q[:,1])),root_vertical_range_m=float(np.ptp(Q[:,2])),
        peak_joint_velocity_rad_s=peak,peak_foot_target_error_m=max(max(c['foot_position_error_m']) for c in checks),
        root_max_correction_m=np.max(abs(Q[:,:3]-s['root_target']),axis=0).tolist(),
        contact_calibration=s['calibration'],frame_checks=checks,
        limitations=['Contact inferred from BVH geometry, not force ground truth',
                     'Robot has no articulated arms or spine roll/pitch; only measured reachable waist twist is mapped',
                     'Rigid robot sole lacks human toe joints; extreme swing foot roll/pitch compressed smoothly',
                     'Nonperiodic source: no transitions from last frame back to first may be used for training',
                     'Kinematic candidate; no policy training or dynamic stability certification',
                     'Original source usage/distribution license must be confirmed before redistribution',
                     'Pelvis roll attenuated for ankle limits; 45mm bind/reach margin and small IK root corrections mean this is not an exact human copy'])
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'candidate_motion.npz',**arrays)
    (args.output/'generation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['frame_checks','contact_calibration']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'data/motions/robotcupmini/raw/lafan1/walk1_subject1.bvh')
    p.add_argument('--start',type=float,default=135.5);p.add_argument('--end',type=float,default=141.5)
    p.add_argument('--time-scale',type=float,default=2.0)
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--diagnostics',type=Path,help='Optional scratch arrays, including rejected candidates; never use as training motion')
    a=p.parse_args()
    if not 0<=a.start<a.end or a.time_scale<=0:raise ValueError('Invalid timing')
    generate(a)

if __name__=='__main__':main()
