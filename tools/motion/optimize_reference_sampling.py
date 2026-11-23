"""Sampling-based contact-aware physical teacher; no PPO/AMP training.

Cross-entropy trajectory optimization repairs a source-human, robot-adapted
motion under free-root MuJoCo physics. Outputs are actual physical states, not
IK poses. Independent audit_physical_reference.py must approve the FULL clip
before any style demonstration is exported. The optimizer cannot certify itself.

Warm starts are only numerical guesses. No root forces, pose resets, altered
masses/inertias/effort limits, or other users' projects are used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, cast
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.spatial import ConvexHull
from mujoco import rollout
from physical_reference_model import build_physical_model, check_transition, mj, oriented_box_min_z, contact_speed_bound
from reference_contact_quality import interior_mask
from reference_body_kinematics import torso_world_angular_velocity
from reference_torso_smoothing import torso_rotation_vector
from play_retarget_motion import load_motion, MODEL_PATH

def prediction_frames(start: int, steps: int, substeps: int, frame_count: int):
    """Rollout state[0] is AFTER its first physics step, never at time start."""
    times = start + np.arange(1, steps + 1) / substeps
    idx = np.minimum(times.astype(int), frame_count - 1)
    return idx, np.minimum(idx + 1, frame_count - 1), times % 1


def exact_mesh_min_z(position, quat_wxyz, local_hull):
    """Exact convex-mesh support height; unlike a box, never invents corners."""
    qw,qx,qy,qz=np.moveaxis(quat_wxyz,-1,0)
    rz=np.stack([2*(qx*qz-qw*qy),2*(qy*qz+qw*qx),1-2*(qx*qx+qy*qy)],axis=-1)
    return position[...,2]+np.einsum('bti,vi->btv',rz,local_hull).min(axis=-1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion', type=Path, required=True)
    p.add_argument('--warm', type=Path, required=True)
    p.add_argument('--neutral-waist-target',action='store_true',help='Hold the waist MOTOR TARGET at its source neutral, not its physical state')
    p.add_argument('--fixed-prefix-steps',type=int,default=0,help='Keep independently optimized initial PD targets; no state overwrite')
    p.add_argument('--prefix-warm', type=Path, default=None)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--duration', type=float, default=12.0)
    p.add_argument('--contact', type=float, default=0.006)
    p.add_argument('--joint-stop', type=float, default=0.004)
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--iterations', type=int, default=5)
    p.add_argument('--horizon', type=int, default=50)
    p.add_argument('--replan', type=int, default=4)
    p.add_argument('--sigma', type=float, default=0.035)
    p.add_argument('--joint-limit-weight', type=float, default=3000.,
                   help='Predictive penalty, not a change to physical joint limits')
    p.add_argument('--joint-limit-margin', type=float, default=0.,
                   help='Additional planning margin inside existing soft limits, rad')
    p.add_argument('--contact-slip-weight', type=float, default=20000.)
    p.add_argument('--contact-speed-threshold', type=float, default=.015)
    p.add_argument('--penetration-safety-weight',type=float,default=0.,help='Planning penalty using EXACT sole hull, not the conservative bounding box; independent gate unchanged')
    p.add_argument('--joint-speed-safety-weight',type=float,default=0.)
    p.add_argument('--joint-speed-safety-limit',type=float,default=4.7)
    p.add_argument('--support-slip-safety-weight',type=float,default=0.)
    p.add_argument('--support-slip-safety-limit',type=float,default=.035)
    p.add_argument('--swing-safety-weight', type=float, default=0., help='Priority peak penalty for predicted swing scuffing; no model/physics changes')
    p.add_argument('--swing-safety-clearance', type=float, default=.005)
    p.add_argument('--swing-clearance-weight', type=float, default=100000.)
    p.add_argument('--swing-clearance-target', type=float, default=.008)
    p.add_argument('--root-forward-weight',type=float,default=250.,help='Fore-aft position tracking; unchanged old default')
    p.add_argument('--root-forward-velocity-weight',type=float,default=0.,help='Prevent standing to evade contact costs on a moving source')
    p.add_argument('--foot-forward-weight',type=float,default=1200.,help='Source fore-aft sole tracking; unchanged old default')
    p.add_argument('--root-lateral-weight', type=float, default=500.,
                   help='Track the corrected lateral reference instead of accepting a wider physical sway')
    p.add_argument('--root-lateral-velocity-weight', type=float, default=0., help='Predictive world-Y velocity tracking; no external root force')
    p.add_argument('--lateral-corridor-weight',type=float,default=0.,help='Predictive peak tracking error penalty; does not modify physical state')
    p.add_argument('--lateral-corridor-error',type=float,default=.004)
    p.add_argument('--foot-lateral-safety-weight',type=float,default=0.)
    p.add_argument('--foot-lateral-safety-error',type=float,default=.008)
    p.add_argument('--foot-width-safety-weight',type=float,default=0.)
    p.add_argument('--heading-safety-weight',type=float,default=0.)
    p.add_argument('--heading-safety-deg',type=float,default=3.)
    p.add_argument('--root-angular-smoothness-weight', type=float, default=0.)
    p.add_argument('--torso-highfreq-weight',type=float,default=0.,help='Predictive 50Hz orientation high-frequency cost; actual recorded states stay unfiltered')
    p.add_argument('--torso-rate-weight', type=float, default=0.)
    p.add_argument('--torso-accel-weight', type=float, default=0.)
    p.add_argument('--torso-pose-weight', type=float, default=0.)
    p.add_argument('--waist-pose-weight', type=float, default=0.)
    p.add_argument('--foot-width-weight', type=float, default=0.)
    p.add_argument('--foot-lateral-track-weight', type=float, default=1800.)
    p.add_argument('--swing-lateral-velocity-weight', type=float, default=0.)
    p.add_argument('--minimum-foot-width', type=float, default=.03)
    p.add_argument('--maximum-foot-width', type=float, default=.25)
    p.add_argument('--control-rate-weight', type=float, default=30.)
    p.add_argument('--control-curvature-weight', type=float, default=10.)
    p.add_argument('--cubic-noise', action='store_true')
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    if not 0 <= args.fixed_prefix_steps <= 10:
        p.error('Invalid fixed initial PD target count')
    if not 32 <= args.samples <= 512 or not 1 <= args.threads <= 8 or (not 5 <= args.horizon <= 100) or (not 1 <= args.iterations <= 20) or (not 1 <= args.replan <= args.horizon) or (not 0.001 <= args.sigma <= 0.3) or (not np.isfinite(args.duration)) or (args.duration <= 0):
        p.error('Invalid CPU sampling parameters')
    if (not np.isfinite(args.joint_limit_weight) or not 3000. <= args.joint_limit_weight <= 1000000.
            or not np.isfinite(args.joint_limit_margin) or not 0. <= args.joint_limit_margin <= .04):
        p.error('Invalid predictive joint-limit penalty/margin')
    if (not np.isfinite(args.contact_slip_weight) or not 20000. <= args.contact_slip_weight <= 1000000.
            or not np.isfinite(args.contact_speed_threshold) or not .003 <= args.contact_speed_threshold <= .015):
        p.error('Invalid predictive contact penalty')
    if (not np.isfinite(args.swing_clearance_weight) or not 100000. <= args.swing_clearance_weight <= 2000000.
            or not np.isfinite(args.swing_clearance_target) or not .008 <= args.swing_clearance_target <= .025):
        p.error('Invalid predictive swing clearance penalty')
    if not np.isfinite(args.root_lateral_weight) or not 500. <= args.root_lateral_weight <= 5000000.:
        p.error('Invalid lateral tracking penalty')
    if not np.isfinite(args.penetration_safety_weight) or not 0 <= args.penetration_safety_weight <= 1e14:
        p.error('Invalid planning penetration penalty')
    if not np.isfinite(args.swing_safety_weight) or not 0 <= args.swing_safety_weight <= 1e14 or not .003 <= args.swing_safety_clearance <= .01:
        p.error('Invalid predictive swing safety parameters')
    if (not 0 <= args.joint_speed_safety_weight <= 1e14 or not 0 <= args.support_slip_safety_weight <= 1e14
            or not 3. <= args.joint_speed_safety_limit <= 4.9 or not .015 <= args.support_slip_safety_limit <= .04
            or not all(np.isfinite(x) for x in [args.joint_speed_safety_weight,args.support_slip_safety_weight])):
        p.error('Invalid predictive peak-safety parameters')
    if not np.isfinite(args.root_forward_weight) or not 0 <= args.root_forward_weight <= 50000000.:
        p.error('Invalid forward tracking penalty')
    peak_weights=[args.torso_highfreq_weight,args.lateral_corridor_weight,args.foot_lateral_safety_weight,
                  args.foot_width_safety_weight,args.heading_safety_weight]
    if (not all(np.isfinite(x) and 0 <= x <= 1e14 for x in peak_weights)
            or not .001 <= args.lateral_corridor_error <= .03
            or not .002 <= args.foot_lateral_safety_error <= .03
            or not .5 <= args.heading_safety_deg <= 10.):
        p.error('Invalid predictive coordination corridor')
    if not np.isfinite(args.torso_pose_weight) or not 0 <= args.torso_pose_weight <= 1e8:
        p.error('Invalid predictive torso pose penalty')
    quality_weights=[args.root_forward_velocity_weight,args.foot_forward_weight,args.root_lateral_velocity_weight,args.root_angular_smoothness_weight,args.torso_rate_weight,args.torso_accel_weight,
                     args.waist_pose_weight,args.foot_width_weight,args.foot_lateral_track_weight,
                     args.swing_lateral_velocity_weight,args.control_rate_weight,
                     args.control_curvature_weight]
    if (not all(np.isfinite(x) and 0 <= x <= 1000000 for x in quality_weights)
            or not .03 <= args.minimum_foot_width < args.maximum_foot_width <= .3):
        p.error('Invalid coordination planning weights or footprint bounds')
    for path in [args.motion, args.warm, args.prefix_warm, args.output]:
        if path is not None and 'lilin' in str(path.resolve()).lower():
            p.error('Another user project is outside scope')
    if args.output.exists():
        p.error('Refusing to overwrite output')
    return args

def run(args: argparse.Namespace) -> None:
    source_sha_at_start = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (a, names, dt) = load_motion(args.motion)
    if abs(dt-.02)>1e-10:
        raise ValueError('This teacher implements a 50 Hz control contract')
    (m, ids, adr, dofs, bids, verts, anchors, kp, kd, effort) = build_physical_model(names, args.contact, sensors=True, joint_limit_time_constant=args.joint_stop)
    m.opt.iterations = 60
    H = args.horizon
    B = args.samples
    stride = 20
    nv = m.nv
    nq = m.nq
    pool = cast(Any, rollout.Rollout(nthread=args.threads))
    workers = [mj.MjData(m) for _ in range(args.threads)]
    rd = mj.MjData(m)
    n = len(a['time_s'])
    Q = np.tile(m.qpos0, (n, 1))
    Q[:, :3] = a['root_positions_m']
    Q[:, 3:7] = a['root_quaternions_xyzw'][:, [3, 0, 1, 2]]
    Q[:, adr] = a['joint_positions_rad']
    V = np.zeros((n, nv))
    for i in range(n):
        (l, h) = (max(0, i - 1), min(n - 1, i + 1))
        mj.mj_differentiatePos(m, V[i], (h - l) * dt, Q[l], Q[h])
    waist_index = names.index('waist_yaw_joint')
    waist_adr = int(adr[waist_index]); waist_dof = int(dofs[waist_index])
    rootR = Rotation.from_quat(Q[:, [4,5,6,3]])
    torso_ref_quat = (rootR * Rotation.from_euler('z', Q[:, waist_adr])).as_quat()[:, [3,0,1,2]]
    ref_world_omega = rootR.apply(V[:,3:6]) + rootR.apply(np.tile([0.,0.,1.],(n,1))) * V[:,waist_dof,None]
    ref_world_alpha = np.gradient(ref_world_omega, dt, axis=0)
    ref_root_alpha = np.gradient(rootR.apply(V[:,3:6]), dt, axis=0)
    refp = []
    refquat = []
    for q in Q:
        rd.qpos[:] = q
        mj.mj_forward(m, rd)
        refp.append([rd.xpos[b] + rd.xmat[b].reshape(3, 3) @ an for (b, an) in zip(bids, anchors)])
        refquat.append([Rotation.from_matrix(rd.xmat[b].reshape(3, 3)).as_quat()[[3, 0, 1, 2]] for b in bids])
    refp = np.asarray(refp)
    ref_foot_velocity=np.gradient(refp,dt,axis=0)
    refquat = np.asarray(refquat)
    with np.load(args.warm,allow_pickle=False) as seed:
        if 'reference_sha256' in seed and str(seed['reference_sha256'].item())!=hashlib.sha256(args.motion.read_bytes()).hexdigest():
            raise ValueError('Warm start belongs to another reference')
        warm=seed['controls'].copy()
    Ubase = Q[:, adr].copy()
    Ubase[:len(warm)] = warm
    if args.prefix_warm is not None:
        prefix_manifest=args.prefix_warm.parent/'run_manifest.json'
        if not prefix_manifest.exists() or json.loads(prefix_manifest.read_text())['input_reference_sha256']!=hashlib.sha256(args.motion.read_bytes()).hexdigest():
            raise ValueError('Prefix target joint mapping/reference is not verified')
        prefix = np.load(args.prefix_warm,allow_pickle=False)['controls']
        Ubase[:len(prefix)] = prefix
    fixed_initial=Ubase[:args.fixed_prefix_steps].copy()
    if args.neutral_waist_target:
        Ubase[:,waist_index]=Q[:,waist_adr]
        Ubase[:args.fixed_prefix_steps]=fixed_initial
    Ubase = gaussian_filter1d(Ubase, 0.5, axis=0, mode='nearest')
    Ubase[:args.fixed_prefix_steps]=fixed_initial
    U = Ubase[:H].copy()
    s = np.r_[0, Q[0], V[0]]
    s[3] -= 0.00015
    S = [s.copy()]
    US = []
    LOG = []
    rng = np.random.default_rng(20261119)
    wall = time.time()
    (lo, hi) = m.jnt_range[ids].T
    softlo = (lo + hi) / 2 - (hi - lo) * 0.45 + args.joint_limit_margin
    softhi = (lo + hi) / 2 + (hi - lo) * 0.45 - args.joint_limit_margin

    clearance_mask=interior_mask(~a['contact_candidates'],5)
    stance_mask=interior_mask(a['contact_candidates'],5)
    sole_lower=[(v-an).min(0) for v,an in zip(verts,anchors)]
    sole_upper=[(v-an).max(0) for v,an in zip(verts,anchors)]
    sole_hulls=[(v-an)[ConvexHull(v-an).vertices] for v,an in zip(verts,anchors)]
    for v,an in zip(verts,anchors):
        if np.linalg.norm(v-an,axis=1).max()>.14:
            raise ValueError('Foot mesh exceeds the declared contact speed-bound radius')

    def world_torso_omega(q,v):
        return torso_world_angular_velocity(q,v,waist_dof)

    def cost(states, sensors, C, k):
        steps = states.shape[1]
        idx, idx2, w = prediction_frames(k, steps, stride, n)
        # A control interval's contact label is held over ALL its 1ms substeps.
        # Future pose interpolation and held contact masks have different clocks.
        # In particular, the final substep may not borrow the NEXT interval's
        # relaxed landing mask; the independent audit tests the current one.
        phase_idx = np.minimum((k + np.arange(steps) / stride).astype(int), n - 1)
        rt = Q[idx, :3] * (1 - w[:, None]) + Q[idx2, :3] * w[:, None]
        rq = Q[idx, 3:7] * (1 - w[:, None]) + Q[idx2, 3:7] * w[:, None]
        rq /= np.linalg.norm(rq, axis=1, keepdims=True)
        q = states[:, :, 1:1 + nq]
        v = states[:, :, 1 + nq:]
        root = q[:, :, :3] - rt
        val = np.sum(root ** 2 * np.array([args.root_forward_weight, args.root_lateral_weight, 1500]), axis=-1)
        if args.root_forward_velocity_weight:
            target_vx=V[idx,0]*(1-w)+V[idx2,0]*w
            val+=args.root_forward_velocity_weight*(v[:,:,0]-target_vx)**2
        if args.root_lateral_velocity_weight:
            target_vy = V[idx, 1] * (1 - w) + V[idx2, 1] * w
            val += args.root_lateral_velocity_weight * (v[:, :, 1] - target_vy)**2
        lateral_peak=np.maximum(abs(root[:,:,1])-args.lateral_corridor_error,0.).max(axis=1)
        # Target root yaw, not waist-compensated torso yaw. A turned pelvis can
        # reduce body-frame foot width even when world footprints look parallel.
        heading_peak=np.zeros(len(states))
        if args.heading_safety_weight:
            qw,qx,qy,qz=[q[:,:,3+i] for i in range(4)]
            rw,rx,ry,rz=[rq[:,i] for i in range(4)]
            yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
            target_yaw=np.arctan2(2*(rw*rz+rx*ry),1-2*(ry*ry+rz*rz))
            error=np.arctan2(np.sin(yaw-target_yaw),np.cos(yaw-target_yaw))
            heading_peak=np.maximum(abs(error)-np.deg2rad(args.heading_safety_deg),0.).max(axis=1)
        width_peak=np.zeros(len(states))
        root_dot = np.sum(q[:, :, 3:7] * rq, axis=-1)
        val += 1200.0 * np.maximum(1 - root_dot ** 2, 0)
        if args.torso_rate_weight or args.torso_accel_weight:
            omega = world_torso_omega(q,v)
            target_omega = ref_world_omega[idx]*(1-w[:,None]) + ref_world_omega[idx2]*w[:,None]
            val += args.torso_rate_weight*np.sum((omega-target_omega)**2,axis=-1)
            if args.torso_accel_weight:
                current_omega = world_torso_omega(s[1:1+nq],s[1+nq:])
                before = np.broadcast_to(current_omega,(len(states),1,3))
                alpha=np.diff(np.concatenate([before,omega],axis=1),axis=1)/.001
                val += args.torso_accel_weight*np.sum((alpha-ref_world_alpha[idx])**2,axis=-1)
        if args.root_angular_smoothness_weight:
            root_v=v.copy();root_v[...,waist_dof]=0.
            omega_root=world_torso_omega(q,root_v)
            current_v=s[1+nq:].copy();current_v[waist_dof]=0.
            before=np.broadcast_to(world_torso_omega(s[1:1+nq],current_v),(len(states),1,3))
            alpha_root=np.diff(np.concatenate([before,omega_root],axis=1),axis=1)/.001
            val += args.root_angular_smoothness_weight*np.sum((alpha_root-ref_root_alpha[idx])**2,axis=-1)
        if args.torso_pose_weight:
            half=q[:,:,waist_adr]/2; cz=np.cos(half); sz=np.sin(half)
            qw,qx,qy,qz=[q[:,:,3+i] for i in range(4)]
            torso_quat=np.stack([qw*cz-qz*sz,qx*cz+qy*sz,qy*cz-qx*sz,qz*cz+qw*sz],axis=-1)
            torso_dot=np.sum(torso_quat*torso_ref_quat[idx],axis=-1)
            val += args.torso_pose_weight*np.maximum(1-torso_dot**2,0)
        val += args.waist_pose_weight*(q[:,:,waist_adr]-Q[idx,waist_adr])**2
        if args.foot_width_weight or args.foot_width_safety_weight:
            separation=sensors[:,:,:3]-sensors[:,:,13:16]
            qw,qx,qy,qz=[q[:,:,3+i] for i in range(4)]
            # Root's world Y axis projected against the left-right vector.
            yaxis=np.stack([2*(qx*qy-qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz+qw*qx)],axis=-1)
            width=np.sum(separation*yaxis,axis=-1)
            val += args.foot_width_weight*(np.maximum(args.minimum_foot_width-width,0)**2+
                                         np.maximum(width-args.maximum_foot_width,0)**2)
            width_peak=(np.maximum(args.minimum_foot_width-width,0.)+
                        np.maximum(width-args.maximum_foot_width,0.)).max(axis=1)
        foot_lateral_peak=np.zeros(len(states))
        dq = q[:, :, adr] - Q[idx][:, adr]
        val += 3.0 * np.sum(dq * dq, axis=-1)
        val += 1500.0 * np.sum(np.maximum(abs(v[:, :, dofs]) - 4.0, 0) ** 2, axis=-1)
        val += 1.5 * np.sum((v[:, :, dofs] - V[idx][:, dofs]) ** 2, axis=-1)
        val += args.joint_limit_weight * np.sum((q[:, :, adr] - np.clip(q[:, :, adr], softlo, softhi)) ** 2, axis=-1)
        safety_peak = np.zeros(len(states))
        penetration_peak = np.zeros(len(states))
        slip_peak = np.zeros(len(states))
        speed_peak=np.maximum(abs(v[:,:,dofs]).max(axis=(1,2))-args.joint_speed_safety_limit,0.)
        for side in range(2):
            sd = sensors[:, :, side * 13:(side + 1) * 13]
            rp = refp[idx, side] * (1 - w[:, None]) + refp[idx2, side] * w[:, None]
            delta = sd[:, :, :3] - rp
            foot_lateral_peak=np.maximum(foot_lateral_peak,
                np.maximum(abs(delta[:,:,1])-args.foot_lateral_safety_error,0.).max(axis=1))
            val += np.sum(delta ** 2 * np.array([args.foot_forward_weight, args.foot_lateral_track_weight, 5000]), axis=-1)
            if args.swing_lateral_velocity_weight:
                target_v=ref_foot_velocity[idx,side,1]*(1-w)+ref_foot_velocity[idx2,side,1]*w
                val += args.swing_lateral_velocity_weight*(sd[:,:,8]-target_v)**2*(~a['contact_candidates'][phase_idx,side])
            minimum=oriented_box_min_z(sd[:,:,:3],sd[:,:,3:7],sole_lower[side],sole_upper[side])
            if args.penetration_safety_weight:
                exact_minimum=exact_mesh_min_z(sd[:,:,:3],sd[:,:,3:7],sole_hulls[side])
                penetration_peak=np.maximum(penetration_peak,np.maximum(-.0015-exact_minimum,0).max(axis=1))
            safety_peak = np.maximum(safety_peak, (np.maximum(args.swing_safety_clearance-minimum,0)*clearance_mask[phase_idx,side]).max(axis=1))
            val+=args.swing_clearance_weight*np.maximum(args.swing_clearance_target-minimum,0)**2*clearance_mask[phase_idx,side]
            upper_slip=contact_speed_bound(sd[:,:,7:10],sd[:,:,10:13])
            val+=args.contact_slip_weight*np.maximum(upper_slip-args.contact_speed_threshold,0)**2*stance_mask[phase_idx,side]
            slip_peak=np.maximum(slip_peak,(np.maximum(upper_slip-args.support_slip_safety_limit,0)*stance_mask[phase_idx,side]).max(axis=1))
            sdq = sd[:, :, 3:7]
            dot = np.sum(sdq * refquat[idx, side], axis=-1)
            val += 50.0 * np.maximum(1 - dot ** 2, 0)
            mask = a['contact_candidates'][phase_idx, side]
            val += 2000 * np.sum(sd[:, :, 7:10] ** 2, axis=-1) * mask
            val += 60.0 * np.sum(sd[:, :, 10:13] ** 2, axis=-1) * mask
        val = val.sum(1) / stride
        val += args.swing_safety_weight * safety_peak**2
        val += args.penetration_safety_weight * penetration_peak**2
        val += args.joint_speed_safety_weight * speed_peak**2
        val += args.support_slip_safety_weight * slip_peak**2
        val += args.lateral_corridor_weight*lateral_peak**2
        val += args.foot_lateral_safety_weight*foot_lateral_peak**2
        val += args.foot_width_safety_weight*width_peak**2
        val += args.heading_safety_weight*heading_peak**2
        if args.torso_highfreq_weight:
            # The planning feature is smoothed ONLY to measure predicted jitter.
            # Both applied controls and exported 1kHz states remain native/unfiltered.
            predicted=torso_rotation_vector(q[:,stride-1::stride],waist_adr)
            history=torso_rotation_vector(np.asarray(S[-26:])[:,1:1+nq],waist_adr)
            history=np.broadcast_to(history,(len(predicted),)+history.shape)
            combined=np.concatenate([history,predicted],axis=1)
            low=gaussian_filter1d(combined,4.5,axis=1,mode='nearest')
            residual=combined[:,history.shape[1]:]-low[:,history.shape[1]:]
            val+=args.torso_highfreq_weight*np.sum(residual**2,axis=(1,2))
        val += 0.4 * np.sum((C - Ubase[k:k + C.shape[1]]) ** 2, axis=(1, 2))
        previous = US[-1] if US else Ubase[0]
        delta = np.diff(np.concatenate([np.broadcast_to(previous, (len(C), 1, 13)), C], axis=1), axis=1)
        val += args.control_rate_weight * np.sum(delta ** 2, axis=(1, 2))
        val += args.control_curvature_weight * np.sum(np.diff(delta, axis=1) ** 2, axis=(1, 2))
        val += 1000.0 * np.sum(root[:, -1, :2] ** 2, axis=-1) + 3000 * (1 - root_dot[:, -1] ** 2)
        expected_time = (k * dt + np.arange(1, steps + 1) * 0.001)[None, :]
        invalid = np.any(abs(states[:, :, 0] - expected_time) > 1e-08, axis=1) | ~np.isfinite(states).all(axis=(1, 2))
        val[invalid] = 1000000000000000.0
        return val
    manifest = {'input_reference': str(args.motion.resolve()), 'input_reference_sha256': hashlib.sha256(args.motion.read_bytes()).hexdigest(), 'mjcf_sha256': hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(), 'contact_time_constant_s': args.contact, 'joint_limit_time_constant_s': args.joint_stop, 'method': 'Sampling MPC, free root, native PD, no training or forces', 'parameters': vars(args), 'status': 'IN_PROGRESS_NOT_APPROVED', 'optimizer_source_sha256': source_sha_at_start, 'torso_smoothing_helper_sha256': hashlib.sha256(Path(__file__).with_name('reference_torso_smoothing.py').read_bytes()).hexdigest(), 'body_kinematics_helper_sha256': hashlib.sha256(Path(__file__).with_name('reference_body_kinematics.py').read_bytes()).hexdigest(), 'warm_start_sha256': hashlib.sha256(args.warm.read_bytes()).hexdigest(), 'prefix_warm_sha256': hashlib.sha256(args.prefix_warm.read_bytes()).hexdigest() if args.prefix_warm else None}
    (args.output / 'run_manifest.json').write_text(json.dumps(manifest, default=str, indent=2))
    N = min(int(round(args.duration / dt)), n - 1)
    for k in range(N):
        if k % args.replan == 0:
            h = min(H, n - 1 - k)
            if k:
                U = np.vstack([U[args.replan:], Ubase[k + len(U) - args.replan:k + len(U)]])[:h]
            nc = max(3, int(np.ceil(h / 8)) + 1)
            xx = np.linspace(0, h - 1, nc)
            yy = np.arange(h)
            mean = np.zeros((nc, 13))
            std = np.full((nc, 13), args.sigma)
            best = np.inf
            bestU = U.copy()
            for it in range(args.iterations):
                z = rng.normal(size=(B, nc, 13)) * std + mean
                z[0] = 0
                z[1] = mean
                noise = np.empty((B, h, 13))
                if args.cubic_noise and h == 1:
                    noise[:] = z[:,:1,:]
                elif args.cubic_noise:
                    noise[:] = CubicSpline(xx,z,axis=1,bc_type='natural')(yy)
                else:
                    for j in range(13):
                        for sample in range(B):
                            noise[sample, :, j] = np.interp(yy, xx, z[sample, :, j])
                C = np.clip(U[None] + noise, softlo - 1.5, softhi + 1.5)
                if args.neutral_waist_target:
                    C[:,:,waist_index]=Q[k:k+h,waist_adr]
                count=min(h,max(0,args.fixed_prefix_steps-k))
                if count:C[:,:count]=Ubase[k:k+count]
                (st, se) = pool.rollout(m, workers, s[None], np.repeat(C, stride, axis=1))
                values = cost(st, se, C, k)
                inds = np.argsort(values)[:max(8, B // 12)]
                elite = z[inds]
                newmean = elite.mean(0)
                newstd = elite.std(0)
                mean = 0.3 * mean + 0.7 * newmean
                std = np.maximum(0.3 * std + 0.7 * newstd, 0.008)
                if values[inds[0]] < best:
                    best = values[inds[0]]
                    bestU = C[inds[0]].copy()
            U = bestU
            q = s[1:1 + nq]
            tilt = np.rad2deg(np.arccos(np.clip(Rotation.from_quat(q[[4, 5, 6, 3]]).as_matrix()[2, 2], -1, 1)))
            print('CEM', k, k * dt, 'cost', round(best, 2), 'height', round(q[2], 3), 'tilt', round(tilt, 2), 'v', round(max(abs(s[1 + nq + dofs])), 2), 'wall', round(time.time() - wall, 1), flush=True)
            LOG.append([k, k * dt, best, 0, q[2], tilt, max(abs(s[1 + nq + dofs]))])
        u = U[k % args.replan]
        (st, _) = pool.rollout(m, workers, s[None], np.repeat(u[None, None], stride, axis=1))
        new = st[0, -1]
        check_transition(s, new, dt)
        s = new
        S.append(s.copy())
        US.append(u.copy())
        if s[3] < 0.35:
            print('FALL', k, flush=True)
            break
        if (k + 1) % args.replan == 0:
            np.savez_compressed(args.output / 'progress.npz', states=S, controls=US, log=LOG)
    np.savez_compressed(args.output / 'rollout.npz', states=S, controls=US, log=LOG)
    manifest.update(method='Sampling MPC, actual physical states, no learning or forces',
                    status='OPTIMIZATION_FINISHED_REQUIRES_INDEPENDENT_AUDIT')
    (args.output / 'run_manifest.json').write_text(json.dumps(manifest, default=str, indent=2))
    print('DONE', len(US) * dt, flush=True)
    pool.close()

def main():
    run(parse_args())
if __name__ == '__main__':
    main()
