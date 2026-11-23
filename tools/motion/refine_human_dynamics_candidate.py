"""Experimental source-driven dynamics refinement; NEVER certifies its own output.

No RL training, root forces, model changes, policy export or neighbouring repo
access. Always run audit_style_pretraining.py on the generated candidate. A
feasible contact-force screen is necessary, but NOT a successful physics rollout.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast
import mujoco as _mj
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import solve_banded
from reference_sway_quality import walking_centerline
from bvh_motion import anatomical_frame
from play_retarget_motion import load_motion, build_model, joint_mapping, MODEL_PATH
from audit_style_pretraining import mesh_points, runs

mujoco = cast(Any, _mj)
ROOT = Path(__file__).resolve().parents[2]


def refine(args):
    """Write an explicitly unapproved candidate to a NEW directory only."""
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    (a, names, dt) = load_motion(args.input)
    m = build_model(MODEL_PATH)
    d = mujoco.MjData(m)
    (ids, adr, root) = joint_mapping(m, names)
    n = len(a['time_s'])
    Q = np.tile(m.qpos0, (n, 1))
    Q[:, :3] = a['root_positions_m']
    Q[:, 3:7] = a['root_quaternions_xyzw'][:, [3, 0, 1, 2]]
    Q[:, adr] = a['joint_positions_rad']
    bids = [m.body(s + '_ankle_roll_link').id for s in ['left', 'right']]
    verts = []
    anchors = []
    rot = []
    feet = []
    com = []
    for b in bids:
        v = np.concatenate([mesh_points(m, g) for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]])
        verts.append(v)
        anchors.append(np.r_[(v.min(0)[:2] + v.max(0)[:2]) / 2, v[:, 2].min()])
    for q in Q:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        rot.append([d.xmat[b].reshape(3, 3).copy() for b in bids])
        feet.append([d.xpos[b] + d.xmat[b].reshape(3, 3) @ an for (b, an) in zip(bids, anchors)])
        com.append(d.subtree_com[1].copy())
    rot = np.array(rot)
    feet = np.array(feet)
    com = np.array(com)
    if args.use_source_foot_targets:
        # Only an initializer: measured, morphology-projected source targets are
        # rebuilt by constrained IK below. Never call these actual physical feet.
        feet = np.asarray(a['foot_targets_world_m'], dtype=float).copy()
        rot = np.asarray(a['foot_target_rotations'], dtype=float).copy()
        if feet.shape != (n, 2, 3) or rot.shape != (n, 2, 3, 3):
            raise ValueError('Invalid source foot targets')
    sourcefeet = feet.copy()
    feet[:, :, 0] *= args.stride_scale
    Q[:, 0] *= args.stride_scale
    com[:, 0] *= args.stride_scale
    Q[:, 2] = args.root_height + args.root_heave_gain * gaussian_filter1d(Q[:, 2] - Q[:, 2].mean(), 5)
    contact = a['contact_candidates']
    t = a['time_s']
    # Independent path centreline prevents source step sway feeding back into
    # alternating footprints. Keep the old mode available for reproducibility.
    lateral_path = (np.full(n, float(np.median(com[:, 1]))) if args.straight_path else
                    com[:, 1] if args.footprint_centerline_s is None else
                    walking_centerline(com[:, 1], dt, args.footprint_centerline_s))
    for j in range(2):
        target = []
        tt = []
        for (lo, hi) in runs(contact[:, j]):
            desired = float(np.median(lateral_path[lo:hi])) + args.width / 2 * (1 if j == 0 else -1)
            tt.extend([t[lo], t[hi - 1]])
            target.extend([desired] * 2)
        feet[:, j, 1] = np.interp(t, tt, target)
        rpy = Rotation.from_matrix(rot[:, j]).as_euler('xyz')
        rpy[:, 0] = 0
        rpy[:, 1] *= 0.4
        rpy[:, 2] *= 0.2
        rot[:, j] = Rotation.from_euler('xyz', rpy).as_matrix()
        rv = np.einsum('tij,vj->tvi', rot[:, j], verts[j] - anchors[j])
        if 'source_sole_clearance_m' in a:
            # Robot-policy sources already carry calibrated sole clearance in
            # TARGET metres. Do not invent human landmarks or scale twice.
            measured = np.asarray(a['source_sole_clearance_m'], dtype=float)
            if measured.shape != (n, 2) or not np.isfinite(measured).all() or (measured < 0).any():
                raise ValueError('Invalid robot-source sole clearance')
            clear = measured[:, j].copy()
        else:
            humanfoot = a['human_positions_m'][:, a['human_joint_names'].tolist().index(('Left' if j == 0 else 'Right') + 'Foot'), 2]
            baseline = np.median(humanfoot[contact[:, j]])
            clear = np.maximum(0, (humanfoot - baseline) * 0.608)
        for (start, stop) in runs(~contact[:, j]):
            u = np.arange(stop - start) * dt
            gate = np.ones(stop - start)
            if start > 0:
                v = np.clip(u / 0.14, 0, 1)
                gate *= v * v * (3 - 2 * v)
            if stop < n:
                v = np.clip((stop - start - 1 - np.arange(stop - start)) * dt / 0.14, 0, 1)
                gate *= v * v * (3 - 2 * v)
            clear[start:stop] = np.maximum(clear[start:stop], 0.015 * gate)
            if start > 0:
                clear[start:stop] *= np.minimum(1, u / 0.08) ** 2
            if stop < n:
                clear[start:stop] *= np.minimum(1, (stop - start - 1 - np.arange(stop - start)) * dt / 0.08) ** 2
        if args.max_swing_clearance is not None:
            # Smooth morphology adaptation, not a new oscillator/contact schedule.
            clear = args.max_swing_clearance * np.tanh(clear / args.max_swing_clearance)
        clear[contact[:, j]] = 0
        feet[:, j, 2] = clear - rv[:, :, 2].min(1)
    load = contact.astype(float)
    for (lo, hi) in runs(contact.all(1)):
        if lo == 0:
            load[lo:hi] = [0.5, 0.5]
            continue
        before = contact[lo - 1, 0]
        u = np.linspace(0, 1, hi - lo)
        w = u ** 3 * (10 - 15 * u + 6 * u * u)
        load[lo:hi, 0] = 1 - w if before else w
        load[lo:hi, 1] = 1 - load[lo:hi, 0]
    cop = np.zeros((n, 2))
    for j in range(2):
        local = np.einsum('tji,tj->ti', rot[:, j], com - feet[:, j])
        local[:, 0] = np.clip(local[:, 0], -0.08, 0.04)
        local[:, 1] = np.clip(local[:, 1], -0.04 + args.margin, 0.04 - args.margin)
        if args.medial_cop:
            # Plan weight transfer at the medial part of the unchanged rigid
            # sole. This is a target, not an applied wrench or contact override.
            local[:, 1] = (-1 if j == 0 else 1) * (0.04 - args.margin)
        local[:, 2] = 0
        cop += load[:, j, None] * (feet[:, j] + np.einsum('tij,tj->ti', rot[:, j], local))[:, :2]
    pad = 150
    speed = (cop[-1] - cop[0]) / ((n - 1) * dt)
    trend = cop[0] + t[:, None] * speed
    padded = np.pad(cop - trend, ((pad, pad), (0, 0)), mode='edge')
    lam = float(com[:, 2].mean()) / 9.81 / dt ** 2
    ab = np.zeros((3, len(padded)))
    ab[0, 1:] = -lam
    ab[1] = 1 + 2 * lam
    ab[2, :-1] = -lam
    ab[1, [0, -1]] = 1 + lam
    COM = solve_banded((1, 1), ab, padded)[pad:-pad] + trend
    rpy = Rotation.from_quat(Q[:, [4, 5, 6, 3]]).as_euler('xyz')
    if args.root_roll_source == 'human_torso':
        # This robot's upper body has NO spine roll joint. Copying pelvis roll
        # onto it reverses the measured human counter-lean. Map the reachable
        # torso roll onto the root instead, and solve leg IK beneath that root.
        hn=a['human_joint_names'].tolist(); hp=a['human_positions_m']
        at=lambda name:hp[:,hn.index(name)]
        human_torso=anatomical_frame(at('LeftShoulder'),at('RightShoulder'),at('Neck')-at('Spine2'))
        measured_roll=np.unwrap(Rotation.from_matrix(human_torso).as_euler('xyz')[:,0])
        rpy[:,0]=gaussian_filter1d(measured_roll-measured_roll.mean(),2.)*args.source_roll_gain
    else:
        rpy[:, 0] *= args.source_roll_gain
    if args.center_source_orientation:
        # Remove source bind/path offsets for a straight walking task, but keep
        # the measured fluctuations. Waist yaw must use the SAME yaw gain.
        rpy[:, 1] = .5 * (rpy[:, 1] - rpy[:, 1].mean())
        rpy[:, 2] -= rpy[:, 2].mean()
        Q[:, adr[-1]] = args.source_yaw_gain * (Q[:, adr[-1]] - Q[:, adr[-1]].mean())
    rpy[:, 2] *= args.source_yaw_gain
    if args.orientation_smoothing_sigma > 0:
        # Smooth source orientation BEFORE solving legs/COM, not physical states.
        # The measured step/foot schedule is unchanged and must be re-audited.
        rpy = gaussian_filter1d(rpy, args.orientation_smoothing_sigma, axis=0, mode='nearest')
        Q[:, adr[-1]] = gaussian_filter1d(Q[:, adr[-1]], args.orientation_smoothing_sigma, mode='nearest')
    rootrot = Rotation.from_euler('xyz', rpy)
    lo = m.jnt_range[ids].mean(1) - 0.9 * np.diff(m.jnt_range[ids], axis=1).ravel() / 2 + 0.008
    hi = m.jnt_range[ids].mean(1) + 0.9 * np.diff(m.jnt_range[ids], axis=1).ravel() / 2 - 0.008
    lo[[3, 9]] = np.maximum(lo[[3, 9]], 0.35)
    for iteration in range(args.iterations):
        res = []
        checks = []
        initial = np.r_[Q[0, adr[:12]], Q[0, :3], np.zeros(3)]
        for i in range(n):

            def calc(x):
                d.qpos[:] = Q[i]
                d.qpos[adr[:12]] = x[:12]
                d.qpos[:3] = x[12:15]
                d.qpos[3:7] = (rootrot[i] * Rotation.from_rotvec(x[15:18])).as_quat()[[3, 0, 1, 2]]
                mujoco.mj_forward(m, d)

            def fun(x):
                calc(x)
                e = []
                for (j, b) in enumerate(bids):
                    r = d.xmat[b].reshape(3, 3)
                    e.extend((d.xpos[b] + r @ anchors[j] - feet[i, j]) / 0.001)
                    e.extend(Rotation.from_matrix(rot[i, j].T @ r).as_rotvec() / 0.01)
                e.extend((d.subtree_com[1, :2] - COM[i]) / 0.002)
                e.extend(x[15:18] / args.root_orientation_sigma)
                e.append((x[14] - Q[i, 2]) / 0.03)
                return np.array(e)
            blo = np.r_[lo[:12], Q[i, :3] - [0.15, 0.1, 0.04], [-0.15] * 3]
            bhi = np.r_[hi[:12], Q[i, :3] + [0.15, 0.1, 0.04], [0.15] * 3]
            fit = least_squares(fun, np.clip(initial, blo + 1e-08, bhi - 1e-08), bounds=(blo, bhi), max_nfev=70, ftol=1e-09, xtol=1e-09, gtol=1e-09)
            initial = fit.x
            calc(initial)
            res.append(d.qpos.copy())
            checks.append(max(abs(fun(initial)[:14])))
            if i % 100 == 0:
                print(i, 'error', checks[-1], flush=True)
        Q = np.array(res)
        V = np.zeros((n, m.nv))
        for i in range(n):
            (l, h) = (max(0, i - 1), min(n - 1, i + 1))
            mujoco.mj_differentiatePos(m, V[i], (h - l) * dt, Q[l], Q[h])
        M = np.zeros((m.nv, m.nv))
        Z = []
        AA = np.gradient(V, dt, axis=0)
        for k in range(n):
            d.qpos[:] = Q[k]
            d.qvel[:] = V[k]
            mujoco.mj_forward(m, d)
            mujoco.mj_fullM(m, d, M)
            w = M @ AA[k] + d.qfrc_bias - d.qfrc_passive
            moment = Rotation.from_quat(Q[k, [4, 5, 6, 3]]).apply(w[3:6]) + np.cross(Q[k, :3], w[:3])
            Z.append(np.array([-moment[1], moment[0]]) / max(w[2], 100.0))
        correction = cop - np.array(Z)
        correction = np.clip(correction, -0.02, 0.02)
        correction = gaussian_filter1d(correction, 3, axis=0)
        padded = np.pad(correction, ((pad, pad), (0, 0)), mode='edge')
        COM += 0.7 * solve_banded((1, 1), ab, padded)[pad:-pad]
        print('ITER', iteration, 'coperr', np.max(abs(correction)), 'speed', abs(V[:, 6:]).max(), flush=True)
    pre_filter_Q = Q.copy()
    sigma = args.smoothing_sigma
    Q[:, :3] = gaussian_filter1d(Q[:, :3], sigma, axis=0, mode='nearest')
    Q[:, adr] = gaussian_filter1d(Q[:, adr], sigma, axis=0, mode='nearest')
    rotations = Rotation.from_quat(Q[:, [4, 5, 6, 3]])
    relative = rotations[0].inv() * rotations
    Q[:, 3:7] = (rotations[0] * Rotation.from_rotvec(gaussian_filter1d(relative.as_rotvec(), sigma, axis=0, mode='nearest'))).as_quat()[:, [3, 0, 1, 2]]
    for i in range(n):
        (low, high) = (max(0, i - 1), min(n - 1, i + 1))
        mujoco.mj_differentiatePos(m, V[i], (high - low) * dt, Q[low], Q[high])
    (actual_feet, actual_com, actual_foot_rotations) = ([], [], [])
    for pose in Q:
        d.qpos[:] = pose
        mujoco.mj_forward(m, d)
        actual_feet.append([d.xpos[b] + d.xmat[b].reshape(3, 3) @ an for (b, an) in zip(bids, anchors)])
        actual_com.append(d.subtree_com[1].copy())
        actual_foot_rotations.append([d.xmat[b].reshape(3, 3).copy() for b in bids])
    a.pop('initializer_only', None)
    a.update(joint_positions_rad=Q[:, adr], joint_velocities_rad_s=V[:, m.jnt_dofadr[ids]], root_positions_m=Q[:, :3], root_quaternions_xyzw=Q[:, [4, 5, 6, 3]], root_rpy_rad=Rotation.from_quat(Q[:, [4, 5, 6, 3]]).as_euler('xyz'), root_linear_velocities_world_m_s=V[:, :3], root_angular_velocities_world_rad_s=Rotation.from_quat(Q[:, [4, 5, 6, 3]]).apply(V[:, 3:6]), foot_targets_world_m=np.asarray(actual_feet), foot_target_rotations=np.asarray(actual_foot_rotations), planned_foot_targets_world_m=feet, reference_com_world_m=np.asarray(actual_com), planned_cop_world_m=cop, valid_for_training=np.array(False), valid_for_style_kinematics=np.array(False), candidate_requires_dynamics_qualification=np.array(True), input_reference_sha256=np.array(hashlib.sha256(args.input.read_bytes()).hexdigest()))
    report = {'status': 'EXPERIMENTAL_NOT_APPROVED_FOR_TRAINING', 'method': 'Source motion -> robot footprint/clearance -> nonperiodic COM preview -> constrained IK -> multibody centroidal correction -> temporal pose smoothing', 'input': str(args.input.resolve()), 'input_sha256': str(a['input_reference_sha256']), 'mjcf_sha256': hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(), 'sample_dt_s': dt, 'frame_count': n, 'parameters': {'orientation_smoothing_sigma_frames': args.orientation_smoothing_sigma, 'root_orientation_sigma_rad': args.root_orientation_sigma, 'medial_cop': args.medial_cop, 'center_source_orientation': args.center_source_orientation, 'straight_path': args.straight_path, 'use_source_foot_targets': args.use_source_foot_targets, 'footprint_centerline_s': args.footprint_centerline_s, 'root_roll_source': args.root_roll_source, 'source_roll_gain': args.source_roll_gain, 'source_yaw_gain': args.source_yaw_gain, 'root_height_m': args.root_height, 'root_heave_gain': args.root_heave_gain, 'max_swing_clearance_m': args.max_swing_clearance, 'foot_center_width_m': args.width, 'forward_translation_scale': args.stride_scale, 'support_lateral_margin_m': args.margin, 'support_cop_x_bounds_relative_to_sole_center_m': [-0.08, 0.04], 'centroidal_iterations': args.iterations, 'smoothing_sigma_frames': sigma, 'minimum_knee_angle_rad': 0.35, 'foot_yaw_gain': 0.2, 'swing_pitch_gain': 0.4}, 'peak_joint_speed_rad_s': float(abs(V[:, m.jnt_dofadr[ids]]).max()), 'root_lateral_range_m': float(np.ptp(Q[:, 1])), 'mean_forward_speed_m_s': float((Q[-1, 0] - Q[0, 0]) / (t[-1] - t[0])), 'max_foot_plan_to_post_filter_fk_error_m': float(abs(np.asarray(actual_feet) - feet).max()), 'maximum_pose_smoothing_joint_change_rad': float(abs(Q[:, adr] - pre_filter_Q[:, adr]).max()), 'valid_for_training': False, 'limitations': ['This file is a candidate generator, not an AMP learner or a deployed controller.', 'The COM preview and centroidal corrections are approximate; the audit must independently test every output.', 'Temporal smoothing can introduce stance drift/mesh penetration; never skip re-auditing.', 'No successful physical teacher rollout or IsaacLab tracking certificate is produced by this generator.', 'Source forward stride, footprint and foot orientation are deliberately adapted; not an exact human copy.', 'The robot has no controllable arms/spine pitch/roll, so human upper-body motion cannot be recreated.', 'Source license/re-distribution permission is not certified here.', 'Nonperiodic clip: never train a transition from the last frame back to the first.']}
    if str(a.get('source_kind', '')) == 'berkeley_published_policy_rollout':
        report['source_kind'] = 'berkeley_published_policy_rollout'
        report['source_policy_sha256'] = str(a['source_policy_sha256'])
        report['limitations'] = [
            'Unapproved robot-source IK candidate; independent physics audit required.',
            'Source is a published Berkeley policy, NOT the checkpoint behind the real-robot video.',
            'Stride, foot width, clearance and timing are adapted to target morphology.',
            'No human skeleton or synthetic human waist/arm motion was introduced.',
            'No AMP learner or IsaacLab tracking is implemented by this generator.',
            'Nonperiodic clip: never train a last-to-first transition.',
        ]
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'candidate_motion.npz', **a)
    (args.output / 'generation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/motions/robotcupmini/retargeted/walk1_candidate_v5_human/candidate_motion.npz')
    parser.add_argument('--output', type=Path, default=ROOT/'data/motions/robotcupmini/retargeted/walk1_candidate_v6_dynamics_draft')
    parser.add_argument('--orientation-smoothing-sigma', type=float, default=0., help='Pre-IK source orientation smoothing, frames; requires fresh physics validation')
    parser.add_argument('--root-orientation-sigma', type=float, default=.1, help='IK attitude correction scale, not a physical lock or extra force')
    parser.add_argument('--medial-cop', action='store_true', help='Plan medial pressure-centre targets within the unchanged sole')
    parser.add_argument('--center-source-orientation', action='store_true', help='Remove constant source orientation offsets and scale waist/root yaw consistently')
    parser.add_argument('--straight-path', action='store_true', help='Plan constant-Y footprints, not the source lateral path; requires fresh physical validation')
    parser.add_argument('--use-source-foot-targets', action='store_true', help='Use unapproved measured source targets as IK goals, never as recorded physics')
    parser.add_argument('--root-height', type=float, default=.507)
    parser.add_argument('--root-heave-gain', type=float, default=1.)
    parser.add_argument('--max-swing-clearance', type=float, default=None)
    parser.add_argument('--footprint-centerline-s', type=float, default=None,
                        help='Slow path filter for stance footprints; excludes within-step COM sway')
    parser.add_argument('--root-roll-source', choices=['pelvis','human_torso'], default='pelvis')
    parser.add_argument('--source-yaw-gain', type=float, default=1.)
    parser.add_argument('--source-roll-gain', type=float, default=.25)
    parser.add_argument('--width', type=float, default=.10)
    parser.add_argument('--margin', type=float, default=.008)
    parser.add_argument('--stride-scale', type=float, default=.65)
    parser.add_argument('--iterations', type=int, default=4)
    parser.add_argument('--smoothing-sigma', type=float, default=1.2)
    args = parser.parse_args()
    for path in [args.input, args.output]:
        if 'lilin' in str(path.resolve()).lower():
            parser.error('Outside user project scope')
    if ((args.footprint_centerline_s is not None and not .5 <= args.footprint_centerline_s <= 5.)
            or not np.isfinite(args.orientation_smoothing_sigma) or not 0. <= args.orientation_smoothing_sigma <= 6.
            or not np.isfinite(args.root_orientation_sigma) or not .01 <= args.root_orientation_sigma <= .1
            or not 0. <= args.source_roll_gain <= 1.
            or not 0. <= args.source_yaw_gain <= 1.
            or not .48 <= args.root_height <= .56 or not .1 <= args.root_heave_gain <= 1.2
            or (args.max_swing_clearance is not None and not .02 <= args.max_swing_clearance <= .1)
            or not all(np.isfinite(x) for x in [args.width,args.margin,args.stride_scale,args.smoothing_sigma])
            or not .08 <= args.width <= .18 or not 0 < args.margin < .03
            or not .4 <= args.stride_scale <= 1.0 or not 1 <= args.iterations <= 10
            or not .1 <= args.smoothing_sigma <= 4.):
        parser.error('Invalid experimental projection parameters')
    refine(args)


if __name__ == '__main__':
    main()
