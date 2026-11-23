"""v3：按躯干世界姿态统一分配骨盆和腰部旋转，修正 v2 侧倾映射。"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

import refine_walk_upper_body as upper
from robot_leg_ik import RobotLegIK
from check_robot_fk import forward_kinematics, read_urdf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = (PROJECT_ROOT / "data/motions/robotcupmini/retargeted"
              / "walk1_candidate_v1_contact/candidate_motion.npz")
OUTPUT_DIR = (PROJECT_ROOT / "data/motions/robotcupmini/retargeted"
              / "walk1_candidate_v3_coordination")
SIDES = ("left", "right")
TORSO_GAINS_RPY = np.array([0.45, 0.45, 0.55])
PELVIS_YAW_GAIN = 0.55
LEAN_GAIN = 0.45
EXTRA_ROOT_LOWERING_M = 0.004  # 为新姿态保留几何伸展余量，不改变地面或脚目标。
ROOT_LIMIT_DEG = np.array([6.0, 6.0, 7.0])
WAIST_LIMIT_DEG = 8.0
MIN_KNEE_RAD = 0.05
SMOOTH_WINDOW = 11


def centered(values):
    return values - np.median(values, axis=0)


def angle_correlation(left, right):
    if np.std(left) < 1e-10 or np.std(right) < 1e-10:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{OUTPUT_DIR}")
    with np.load(INPUT_FILE, allow_pickle=False) as archive:
        motion = {name: archive[name].copy() for name in archive.files}
    ik = RobotLegIK(PROJECT_ROOT)
    times = motion["time_s"]
    dt = float(motion["sample_dt_s"].item())
    source_times = motion["source_time_s"]
    roots = motion["root_positions_m"].copy()
    feet = motion["foot_targets_world_m"]
    contacts = motion["contact_candidates"]
    count = len(times)
    if (count < 11 or not np.isfinite(dt) or dt <= 0
            or source_times.shape != (count,)
            or roots.shape != (count, 3) or feet.shape != (count, 2, 3)
            or contacts.shape != (count, 2) or contacts.dtype != np.dtype(bool)):
        raise ValueError("输入数组形状、接触类型或采样间隔无效")
    if not all(np.isfinite(array).all() for array in (times, source_times, roots, feet)):
        raise ValueError("输入含 NaN 或 Inf")
    if not np.allclose(np.diff(times), dt, atol=1e-8, rtol=1e-6):
        raise ValueError("时间轴与采样间隔不匹配")
    if not np.allclose(source_times, times + ik.config["source"]["start_time_s"]):
        raise ValueError("人体源时间与当前配置不一致")
    if motion["joint_names"].tolist() != ik.names:
        raise ValueError("候选关节顺序与项目配置不一致")
    if not np.allclose(motion["root_quaternions_xyzw"], [0, 0, 0, 1]):
        raise ValueError("此脚本应从未旋转骨盆的 v1 重新生成，不能叠加到 v2")

    # 此分配只适用于本机器人：腰关节直接连接 pelvis 与 torso_link，轴为 Z。
    waist_spec = next(joint for joint in ik.joints if joint.name == "waist_yaw_joint")
    if (waist_spec.parent != ik.root or waist_spec.child != "torso_link"
            or not np.allclose(waist_spec.axis, [0, 0, 1], atol=1e-8)
            or not np.allclose(waist_spec.origin[:3, :3], np.eye(3), atol=1e-8)):
        raise ValueError("腰部结构与当前旋转分配模型不兼容")
    for side in SIDES:
        column = ik.leg_names.index(f"{side}_knee_joint")
        ik.lower[column] = max(ik.lower[column], MIN_KNEE_RAD)

    # 复用已存在的读取与解剖坐标工具；不是复用 v2 的错误姿态映射。
    upper.PROJECT_ROOT = PROJECT_ROOT
    human_names, xyz = upper.sample_human(ik.config, source_times)

    def point(name):
        return xyz[:, human_names.index(name)]

    human_pelvis = upper.anatomical_frame(
        point("LeftUpLeg"), point("RightUpLeg"), point("Spine") - point("Hips"))
    human_torso = upper.anatomical_frame(
        point("LeftShoulder"), point("RightShoulder"), point("Neck") - point("Hips"))
    pelvis_rpy = np.unwrap(Rotation.from_matrix(human_pelvis).as_euler("xyz"), axis=0)
    torso_rpy = np.unwrap(Rotation.from_matrix(human_torso).as_euler("xyz"), axis=0)
    window = min(SMOOTH_WINDOW, count if count % 2 else count - 1)
    pelvis_smooth = savgol_filter(pelvis_rpy, window, 3, axis=0)
    torso_smooth = savgol_filter(torso_rpy, window, 3, axis=0)

    # 与 v2 的主要区别：侧倾/前倾从躯干提取，不把骨盆侧倾当躯干侧倾。
    desired_torso_rpy = centered(torso_smooth) * TORSO_GAINS_RPY
    desired_torso_rpy[:, 1] += LEAN_GAIN * np.median(torso_smooth[:, 1])
    desired_torso_rotation = Rotation.from_euler("xyz", desired_torso_rpy)
    pelvis_yaw_hint = PELVIS_YAW_GAIN * centered(pelvis_smooth[:, 2])
    waist_yaw = desired_torso_rpy[:, 2] - pelvis_yaw_hint

    # 精确满足 R_torso = R_root @ R_waist。
    # 根姿态不再与腰部各自独立缩放后随意相加。
    root_rotation = desired_torso_rotation * Rotation.from_euler("z", -waist_yaw)
    root_matrices = root_rotation.as_matrix()
    root_rpy = root_rotation.as_euler("xyz")
    if np.any(np.abs(np.degrees(root_rpy)) > ROOT_LIMIT_DEG):
        raise ValueError("骨盆姿态超过候选幅度限制，不自动截断")
    if np.max(np.abs(np.degrees(waist_yaw))) > WAIST_LIMIT_DEG:
        raise ValueError("腰部姿态超过候选幅度限制")
    limits = read_urdf(PROJECT_ROOT / ik.config["target"]["urdf_file"])[3]
    lower, upper_limit = limits["waist_yaw_joint"]
    if np.any(waist_yaw < lower) or np.any(waist_yaw > upper_limit):
        raise ValueError("腰部角度超过 URDF 限位")
    roots[:, 2] -= EXTRA_ROOT_LOWERING_M

    neutral_feet = ik.foot_poses(ik.default)
    angles_sequence = []
    checks = []
    torso_error_deg = []
    seed = ik.default.copy()
    for frame in range(count):
        rotation = root_matrices[frame]
        goals = {side: (
            rotation.T @ (feet[frame, index] - roots[frame]),
            rotation.T @ neutral_feet[side][1],
        ) for index, side in enumerate(SIDES)}
        angles, check = ik.solve(goals, initial=seed, base_height=0.0)
        seed = angles.copy()
        angles["waist_yaw_joint"] = float(waist_yaw[frame])
        transforms = forward_kinematics(ik.root, ik.joints, angles, base_height=0.0)
        actual_torso = rotation @ transforms["torso_link"][:3, :3]
        target_torso = desired_torso_rotation[frame].as_matrix()
        error = float(np.degrees(Rotation.from_matrix(target_torso.T @ actual_torso).magnitude()))
        torso_error_deg.append(error)
        angles_sequence.append([angles[name] for name in ik.names])
        checks.append(check)
        if frame % 50 == 0:
            print(f"frame={frame}/{count} accepted={check['accepted']}")
    positions = np.asarray(angles_sequence)
    velocities = np.gradient(positions, dt, axis=0)
    actual_torso_rpy = (root_rotation * Rotation.from_euler("z", waist_yaw)).as_euler("xyz")
    correlations = {label: angle_correlation(torso_smooth[:, column], actual_torso_rpy[:, column])
                    for column, label in enumerate(("roll", "pitch", "yaw"))}
    failed = [frame for frame, check in enumerate(checks)
              if not check["accepted"] or torso_error_deg[frame] > 0.01]
    step_rates = Rotation.from_matrix(root_matrices[1:] @ np.swapaxes(
        root_matrices[:-1], -1, -2)).as_rotvec() / dt
    angular_velocity = np.vstack([step_rates[0],
                                  0.5 * (step_rates[:-1] + step_rates[1:]), step_rates[-1]])
    report = {
        "status": "torso_world_pose_candidate_not_dynamics_validated",
        "valid_for_training": False,
        "input_motion": str(INPUT_FILE),
        "source": ik.config["source"],
        "frame_count": count,
        "joint_names": ik.names,
        "parameters": {"torso_gains_rpy": TORSO_GAINS_RPY.tolist(),
                       "pelvis_yaw_gain": PELVIS_YAW_GAIN,
                       "lean_gain": LEAN_GAIN,
                       "extra_root_lowering_m": EXTRA_ROOT_LOWERING_M,
                       "smooth_window_frames": window},
        "root_rpy_min_deg": np.degrees(root_rpy).min(axis=0).tolist(),
        "root_rpy_max_deg": np.degrees(root_rpy).max(axis=0).tolist(),
        "torso_world_rpy_min_deg": np.degrees(actual_torso_rpy).min(axis=0).tolist(),
        "torso_world_rpy_max_deg": np.degrees(actual_torso_rpy).max(axis=0).tolist(),
        "waist_yaw_range_deg": [float(np.degrees(waist_yaw).min()), float(np.degrees(waist_yaw).max())],
        "source_torso_angle_correlations": correlations,
        "max_torso_orientation_error_deg": max(torso_error_deg),
        "max_foot_position_error_m": max(check["max_position_error_m"] for check in checks),
        "max_foot_orientation_error_deg": max(check["max_orientation_error_deg"] for check in checks),
        "max_joint_speed_rad_s": float(np.abs(velocities).max()),
        "all_frames_ik_accepted": not failed,
        "failed_frames": failed,
        "limitations": ["Correlation checks mapping, not physical support or naturalness",
                        "Landmark orientation proxies, not exact human joint rotations",
                        "Foot targets/contact candidates unchanged from v1; no loop blending",
                        "No lateral weight-shift optimization, collision or dynamics validation"],
        "frame_diagnostics": checks,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    if failed:
        raise RuntimeError(f"{len(failed)} 帧未通过，只保存报告，不输出动作")
    motion.update({
        "joint_positions_rad": positions,
        "joint_velocities_rad_s": velocities,
        "root_positions_m": roots,
        "root_quaternions_xyzw": root_rotation.as_quat(),
        "root_rpy_rad": root_rpy,
        "desired_torso_quaternions_xyzw": desired_torso_rotation.as_quat(),
        "root_linear_velocities_world_m_s": np.gradient(roots, dt, axis=0),
        "root_angular_velocities_world_rad_s": angular_velocity,
        "source_torso_proxy_rpy_rad": torso_smooth,
        "valid_for_training": np.array(False),
    })
    np.savez_compressed(OUTPUT_DIR / "candidate_motion.npz", **motion)
    figure, axes = plt.subplots(2, 2, figsize=(12, 7))
    for column, label in enumerate(("roll", "pitch", "yaw")):
        axis = axes.flat[column]
        axis.plot(times, np.degrees(centered(torso_smooth[:, column])) * TORSO_GAINS_RPY[column],
                  "--", label="human torso (centered + scaled)")
        axis.plot(times, np.degrees(centered(actual_torso_rpy[:, column])), label="robot world torso")
        axis.set_title(f"World torso {label} (deg): mapping check")
    axes[1, 1].plot(times, np.degrees(root_rpy[:, 2]), label="pelvis world yaw")
    axes[1, 1].plot(times, np.degrees(waist_yaw), label="waist local yaw")
    axes[1, 1].plot(times, np.degrees(actual_torso_rpy[:, 2]), label="torso world yaw")
    axes[1, 1].set_title("Yaw allocation (deg)")
    for axis in axes.flat:
        axis.grid(True)
        axis.legend(fontsize=8)
        axis.set_xlabel("Time (s)")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "diagnostics.png", dpi=150)
    plt.close(figure)
    print(f"IK 通过：{count}/{count}")
    print("躯干 roll/pitch/yaw 与源信号相关系数：", correlations)
    print("躯干世界角度最小值(deg)：", np.round(report["torso_world_rpy_min_deg"], 2))
    print("躯干世界角度最大值(deg)：", np.round(report["torso_world_rpy_max_deg"], 2))
    print("腰部角度范围(deg)：", np.round(report["waist_yaw_range_deg"], 2))
    print(f"最大关节速度：{report['max_joint_speed_rad_s']:.3f} rad/s")
    print(f"输出：{OUTPUT_DIR}")
    print("仅完成几何姿态映射修正，未验证动态平衡或力矩，不能用于训练。")


if __name__ == "__main__":
    main()
