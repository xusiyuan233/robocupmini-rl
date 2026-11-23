"""重建 RobotcupMini 周期参考：支撑交接、COM 转移与全身 IK。

这是机器人约束下的步态重建，不是人体动作的逐帧忠实复制，不训练策略。
不覆盖已有参考；仅输出新版本。须另做接触力筛查和物理跟踪验证。
运行环境：env_robocup_sim2sim（numpy/scipy/mujoco）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import mujoco as _mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from audit_style_pretraining import mesh_points
from play_retarget_motion import MODEL_PATH, build_model, joint_mapping, load_motion

mujoco = cast(Any, _mujoco)
ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'data/motions/robotcupmini/retargeted/walk1_candidate_v3_coordination/candidate_motion.npz'
OUTPUT = ROOT / 'data/motions/robotcupmini/retargeted/walk1_candidate_v4_balanced'
DT = 0.02


def minimum_jerk(u):
    """0→1 的五次曲线；端点速度、加速度均为零。"""
    return u**3 * (10.0 - 15.0 * u + 6.0 * u**2)


def make_feet_and_cop(count, period, stride, width, lift, duty, cop_inset):
    times = np.arange(count) * DT
    feet = np.zeros((count, 2, 3))
    contacts = np.zeros((count, 2), dtype=bool)
    cop = np.zeros((count, 2))
    for i, time in enumerate(times):
        for side in range(2):
            unwrapped = time / period - side * 0.5
            cycle = np.floor(unwrapped)
            phase = unwrapped - cycle
            landing_x = (cycle + side * 0.5) * stride + duty * stride / 2
            y = (1 if side == 0 else -1) * width / 2
            contacts[i, side] = phase < duty - 1e-9
            if phase < duty:
                feet[i, side] = [landing_x, y, 0.0]
            else:
                u = (phase - duty) / (1 - duty)
                # 接触衔接不再使用会产生加速度跳变的分段线性目标。
                feet[i, side] = [landing_x + stride * minimum_jerk(u), y,
                                 lift * 64 * u**3 * (1-u)**3]
        phase = time / period
        overlap = duty - 0.5
        if phase < overlap:
            left_load = minimum_jerk(phase / overlap)
        elif phase < 0.5:
            left_load = 1.0
        elif phase < duty:
            left_load = 1 - minimum_jerk((phase - 0.5) / overlap)
        else:
            left_load = 0.0
        # 规划压力中心略靠足底内侧，避免踝侧摆贴近软限位。
        left_cop = feet[i, 0, :2] - [0, cop_inset]
        right_cop = feet[i, 1, :2] + [0, cop_inset]
        cop[i] = left_load * left_cop + (1-left_load) * right_cop
    return times, feet, contacts, cop


def plan_com(cop, times, speed, height):
    """周期 LIPM 滤波 c-(h/g)c''=p；只用作轨迹初步规划，非完整动力学证明。"""
    detrended = cop.copy()
    detrended[:, 0] -= speed * times
    omega = 2 * np.pi * np.fft.fftfreq(len(times), DT)
    result = np.fft.ifft(np.fft.fft(detrended, axis=0) /
                         (1 + height / 9.81 * omega[:, None]**2), axis=0).real
    result[:, 0] += speed * times
    return result


def periodic_velocities(model, qpos, translation):
    """跨周期接缝的四元数感知速度；根位置延拓时加减整段前进距离。"""
    before = qpos[-1].copy()
    before[:3] -= translation
    after = qpos[0].copy()
    after[:3] += translation
    extended = np.vstack([before, qpos, after])
    velocity = np.zeros((len(qpos), model.nv))
    for i in range(len(qpos)):
        mujoco.mj_differentiatePos(model, velocity[i], 2*DT, extended[i], extended[i+2])
    return velocity


def generate(args):
    if args.output.exists():
        raise FileExistsError(f'拒绝覆盖已有输出：{args.output}')
    source, names, _ = load_motion(args.input)
    model = build_model(MODEL_PATH)
    data = mujoco.MjData(model)
    ids, addresses, root_address = joint_mapping(model, names)
    if root_address != 0 or len(names) != 13 or names[-1] != 'waist_yaw_joint':
        raise ValueError('本工具只适用于已核对的 RobotcupMini 13 关节模型')
    if not (0.5 < args.duty < 0.75 and 0 < args.cop_inset < 0.03 and
            args.width > 2*args.cop_inset and args.cycles >= 2 and args.period > 0):
        raise ValueError('支撑占空比、周期、步宽或压力中心参数无效')
    count = round(args.period / DT)
    period = count * DT
    if not np.isclose(period, args.period):
        raise ValueError('周期必须为 0.02 秒的整数倍')
    times, feet, contact, cop = make_feet_and_cop(
        count, period, args.stride, args.width, args.lift, args.duty, args.cop_inset)
    phase = times / period
    # 腿步进、骨盆转动、腰部反向转动共用相位，不再各自任意加大幅度。
    root_rpy = np.column_stack([
        -np.deg2rad(0.8) * np.sin(2*np.pi*phase),
        np.deg2rad(1.7) + np.deg2rad(0.25) * np.cos(4*np.pi*phase),
        -np.deg2rad(2.5) * np.cos(2*np.pi*phase),
    ])
    root_rotation = Rotation.from_euler('xyz', root_rpy)
    waist = np.deg2rad(4.0) * np.cos(2*np.pi*phase)
    foot_ids = [model.body(side+'_ankle_roll_link').id for side in ['left', 'right']]
    anchors = []
    for body in foot_ids:
        vertices = np.concatenate([mesh_points(model, geom) for geom in range(model.ngeom)
            if model.geom_bodyid[geom] == body and
            model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_contype[geom]])
        low, high = vertices.min(0), vertices.max(0)
        anchors.append(np.r_[(low[:2]+high[:2])/2, low[2]])
    hard = model.jnt_range[ids]
    midpoint = hard.mean(1)
    half_range = np.diff(hard, axis=1).ravel()/2
    soft_lower = midpoint - 0.9*half_range
    soft_upper = midpoint + 0.9*half_range
    lower = soft_lower[:12] + 0.005
    upper = soft_upper[:12] - 0.005
    lower[[3, 9]] = np.maximum(lower[[3, 9]], 0.10)
    initial = np.r_[np.clip(source['joint_positions_rad'][0, :12], lower, upper), 0.0, 0.0]
    total_mass = model.body_mass.sum()
    com_height = 0.52
    final = {}
    for iteration in range(3):
        target_com = plan_com(cop, times, args.stride/period, com_height)
        positions, roots, coms, diagnostics = [], [], [], []
        for frame in range(count):
            quat = root_rotation[frame].as_quat()[[3, 0, 1, 2]]

            def residual(vector):
                data.qpos[:] = model.qpos0
                data.qpos[:3] = [vector[12], vector[13], args.height]
                data.qpos[3:7] = quat
                data.qpos[addresses[:12]] = vector[:12]
                data.qpos[addresses[12]] = waist[frame]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                error = []
                for side, body in enumerate(foot_ids):
                    rotation = data.xmat[body].reshape(3, 3)
                    position = rotation @ anchors[side] + data.xpos[body]
                    error.extend((position - feet[frame, side]) / 0.005)
                    error.extend(Rotation.from_matrix(rotation).as_rotvec() / 0.04)
                com = np.sum(model.body_mass[:, None] * data.xipos, axis=0) / total_mass
                error.extend((com[:2]-target_com[frame]) / 0.005)
                return np.asarray(error)

            solved = least_squares(residual, initial,
                bounds=(np.r_[lower, -5.0, -1.0], np.r_[upper, 5.0, 1.0]),
                max_nfev=100, ftol=1e-11, xtol=1e-11, gtol=1e-11)
            initial = solved.x
            error = residual(initial)
            diagnostics.append({'frame':frame, 'success':bool(solved.success),
                                'max_scaled_residual':float(abs(error).max())})
            if not solved.success or abs(error).max() > 1e-4:
                raise RuntimeError(f'第 {frame} 帧 IK 不满足几何/COM 精度，不输出伪合格动作：{diagnostics[-1]}')
            positions.append(np.r_[initial[:12], waist[frame]])
            roots.append(data.qpos[:3].copy())
            coms.append(np.sum(model.body_mass[:, None]*data.xipos, axis=0)/total_mass)
        coms = np.asarray(coms)
        new_height = float(coms[:, 2].mean())
        final = dict(positions=np.asarray(positions), roots=np.asarray(roots), coms=coms,
                     diagnostics=diagnostics, target_com=target_com)
        print(f'COM/IK pass {iteration+1}: mean COM height={new_height:.6f} m', flush=True)
        initial = np.r_[positions[0][:12], roots[0][:2]]
        if abs(new_height-com_height) < 1e-5:
            com_height = new_height
            break
        com_height = new_height
    positions = final['positions']
    cycles = args.cycles
    root_all = np.concatenate([final['roots'] + [cycle*args.stride, 0, 0] for cycle in range(cycles)])
    feet_all = np.concatenate([feet + [cycle*args.stride, 0, 0] for cycle in range(cycles)])
    joint_all = np.tile(positions, (cycles, 1))
    quaternions = np.tile(root_rotation.as_quat(), (cycles, 1))
    qpos = np.tile(model.qpos0, (cycles*count, 1))
    qpos[:, :3] = root_all
    qpos[:, 3:7] = quaternions[:, [3, 0, 1, 2]]
    qpos[:, addresses] = joint_all
    velocity = periodic_velocities(model, qpos, np.array([cycles*args.stride, 0, 0]))
    joint_velocity = velocity[:, model.jnt_dofadr[ids]]
    if abs(joint_velocity).max() > 4.5:
        raise RuntimeError('关节速度超过本参考 4.5 rad/s 验收上限，拒绝输出')
    if np.any(joint_all < soft_lower+0.005-1e-7) or np.any(joint_all > soft_upper-0.005+1e-7):
        raise RuntimeError('未保留软限位余量，拒绝输出')
    world_omega = Rotation.from_quat(quaternions).apply(velocity[:, 3:6])
    source_sha = hashlib.sha256(args.input.read_bytes()).hexdigest()
    motion = dict(
        time_s=np.arange(cycles*count)*DT, sample_dt_s=np.array(DT), joint_names=np.array(names),
        joint_positions_rad=joint_all, joint_velocities_rad_s=joint_velocity,
        root_positions_m=root_all, root_quaternions_xyzw=quaternions,
        root_linear_velocities_world_m_s=velocity[:, :3], root_angular_velocities_world_rad_s=world_omega,
        foot_targets_world_m=feet_all, contact_candidates=np.tile(contact, (cycles, 1)),
        foot_names=np.array(['left','right']), root_rpy_rad=np.tile(root_rpy, (cycles, 1)),
        desired_torso_quaternions_xyzw=np.tile((root_rotation*Rotation.from_euler('z', waist)).as_quat(), (cycles, 1)),
        reference_com_world_m=np.concatenate([final['coms']+[k*args.stride,0,0] for k in range(cycles)]),
        planned_cop_world_m=np.concatenate([cop+[k*args.stride,0] for k in range(cycles)]),
        cycle_duration_s=np.array(period), cycle_translation_m=np.array([args.stride,0,0]),
        periodic_reference=np.array(True), source_motion_sha256=np.array(source_sha),
        valid_for_style_kinematics=np.array(True), valid_for_training=np.array(False))
    report = dict(status='robot_specific_periodic_reference_geometry_validated_not_policy_or_dynamics_certified',
        source_motion=str(args.input), source_sha256=source_sha,
        reconstruction='Robot-constrained periodic reconstruction, not frame-faithful human retargeting',
        parameters=dict(period_s=period, stride_m=args.stride, speed_m_s=args.stride/period,
                        foot_center_width_m=args.width, swing_clearance_m=args.lift, stance_fraction=args.duty,
                        double_support_fraction=2*args.duty-1, root_height_m=args.height,
                        cop_inset_m=args.cop_inset, cycles=cycles, sample_dt_s=DT),
        joint_order=names, frame_count=cycles*count, max_joint_speed_rad_s=float(abs(joint_velocity).max()),
        min_soft_limit_margin_rad=float(np.minimum(joint_all-soft_lower,soft_upper-joint_all).min()),
        max_scaled_ik_residual=max(d['max_scaled_residual'] for d in final['diagnostics']),
        average_com_height_m=com_height, root_lateral_range_m=[float(root_all[:,1].min()),float(root_all[:,1].max())],
        valid_for_style_kinematics=True, valid_for_training=False,
        limitations=['No policy training or true-robot validation','LIPM is approximate; full-model contact/PD checks remain separate',
                     'No arms or torso roll/pitch DOFs added; only existing 13 joints used',
                     'A cyclic root translation is required for spatial looping; never connect raw world positions across reset'])
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'candidate_motion.npz', **motion)
    (args.output/'generation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=INPUT)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--height',type=float,default=0.53)
    parser.add_argument('--stride',type=float,default=0.32)
    parser.add_argument('--period',type=float,default=1.8)
    parser.add_argument('--width',type=float,default=0.22)
    parser.add_argument('--lift',type=float,default=0.035)
    parser.add_argument('--duty',type=float,default=0.60)
    parser.add_argument('--cop-inset',type=float,default=0.015)
    parser.add_argument('--cycles',type=int,default=4)
    generate(parser.parse_args())


if __name__ == '__main__':
    main()
