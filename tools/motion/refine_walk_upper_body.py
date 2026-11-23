"""从人体解剖标志提取小幅骨盆/腰部运动，保持 v1 双脚世界目标。"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from preview_lafan1 import load_bvh
from robot_leg_ik import RobotLegIK
from check_robot_fk import read_urdf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = (PROJECT_ROOT / "data/motions/robotcupmini/retargeted"
              / "walk1_candidate_v1_contact/candidate_motion.npz")
OUTPUT_DIR = (PROJECT_ROOT / "data/motions/robotcupmini/retargeted"
              / "walk1_candidate_v2_upper")
SIDES = ("left", "right")
PELVIS_GAINS = np.array([0.25, 0.30, 0.25])  # roll、pitch、yaw
TORSO_LEAN_GAIN = 0.30
WAIST_YAW_GAIN = 0.50
ROOT_ANGLE_LIMIT_DEG = np.array([4.0, 4.0, 4.0])
WAIST_LIMIT_DEG = 6.0
MIN_KNEE_RAD = 0.05
SMOOTH_WINDOW = 11


def normalize(vectors):
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(lengths).all() or np.any(lengths < 1e-6):
        raise ValueError("人体标志向量无效，不能构造姿态坐标系")
    return vectors / lengths


def anatomical_frame(left, right, upward):
    """列向量依次为前、左、上；由骨架位置构造正交右手坐标系。"""
    y_axis = normalize(left - right)
    z_axis = normalize(upward - y_axis * np.sum(y_axis * upward, axis=-1, keepdims=True))
    x_axis = normalize(np.cross(y_axis, z_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=-1)


def sample_human(config, absolute_times):
    """以候选动作的原始时间戳采样，沿用 v0 的坐标与前进方向约定。"""
    source = config["source"]
    if source["position_unit"] != "cm" or source["up_axis"] != "Y":
        raise ValueError("当前读取器只支持厘米、Y 向上的 LAFAN1 输入")
    names, _, positions, source_dt = load_bvh(PROJECT_ROOT / source["file"])
    source_times = np.arange(len(positions)) * source_dt
    if absolute_times[0] < 0 or absolute_times[-1] > source_times[-1]:
        raise ValueError("候选时间超出原始 BVH 范围")
    flat = positions.reshape(len(positions), -1)
    sampled = np.column_stack([
        np.interp(absolute_times, source_times, flat[:, column])
        for column in range(flat.shape[1])
    ]).reshape(len(absolute_times), len(names), 3)
    xyz = sampled[..., [0, 2, 1]].copy()
    xyz[..., 1] *= -1
    hips = names.index("Hips")
    xyz[..., :2] -= xyz[0, hips, :2].copy()
    displacement = xyz[-1, hips, :2]
    if np.linalg.norm(displacement) < 0.1:
        raise ValueError("片段净位移过小")
    yaw = np.arctan2(displacement[1], displacement[0])
    c, s = np.cos(yaw), np.sin(yaw)
    xyz[..., :2] = xyz[..., :2] @ np.array([[c, -s], [s, c]])
    return names, xyz


def extract_upper_pose(names, xyz):
    """提取姿态代理，去除解剖坐标静态偏置，再小幅映射到机器人。"""
    def point(name):
        return xyz[:, names.index(name)]

    pelvis = anatomical_frame(point("LeftUpLeg"), point("RightUpLeg"),
                              point("Spine") - point("Hips"))
    torso = anatomical_frame(point("LeftShoulder"), point("RightShoulder"),
                             point("Neck") - point("Hips"))
    relative = np.swapaxes(pelvis, -1, -2) @ torso
    pelvis_angles = np.unwrap(Rotation.from_matrix(pelvis).as_euler("xyz"), axis=0)
    torso_angles = np.unwrap(Rotation.from_matrix(torso).as_euler("xyz"), axis=0)
    relative_angles = np.unwrap(Rotation.from_matrix(relative).as_euler("xyz"), axis=0)
    window = min(SMOOTH_WINDOW, len(xyz) if len(xyz) % 2 else len(xyz) - 1)
    if window < 5:
        raise ValueError("动作太短，不能进行当前平滑处理")
    pelvis_smooth = savgol_filter(pelvis_angles, window, 3, axis=0)
    relative_smooth = savgol_filter(relative_angles, window, 3, axis=0)
    robot_angles = (pelvis_smooth - np.median(pelvis_smooth, axis=0)) * PELVIS_GAINS
    # 不复制人体骨盆标志坐标系的静态俯仰；采用小幅躯干平均前倾。
    lean_bias = float(TORSO_LEAN_GAIN * np.median(torso_angles[:, 1]))
    robot_angles[:, 1] += lean_bias
    waist = WAIST_YAW_GAIN * (relative_smooth[:, 2] - np.median(relative_smooth[:, 2]))
    if np.any(np.abs(np.degrees(robot_angles)) > ROOT_ANGLE_LIMIT_DEG):
        raise ValueError("骨盆角度超过当前候选幅度上限；不自动截断或导出")
    if np.max(np.abs(np.degrees(waist))) > WAIST_LIMIT_DEG:
        raise ValueError("腰部角度超过当前候选幅度上限")
    return robot_angles, waist, lean_bias, pelvis_angles, relative_angles


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{OUTPUT_DIR}")
    with np.load(INPUT_FILE, allow_pickle=False) as archive:
        motion = {name: archive[name].copy() for name in archive.files}
    ik = RobotLegIK(PROJECT_ROOT)
    times = motion["time_s"]
    dt = float(motion["sample_dt_s"].item())
    absolute_times = motion["source_time_s"]
    roots = motion["root_positions_m"]
    foot_targets = motion["foot_targets_world_m"]
    contacts = motion["contact_candidates"]
    if motion["joint_names"].tolist() != ik.names:
        raise ValueError("关节顺序与项目配置不一致")
    if (len(times) < 5 or not np.isfinite(dt) or dt <= 0
            or absolute_times.shape != times.shape
            or roots.shape != (len(times), 3)
            or foot_targets.shape != (len(times), 2, 3)
            or contacts.shape != (len(times), 2)):
        raise ValueError("输入形状或采样间隔不正确")
    if not all(np.isfinite(array).all() for array in (times, absolute_times, roots, foot_targets)):
        raise ValueError("输入含 NaN 或 Inf")
    if not np.allclose(np.diff(times), dt, atol=1e-8, rtol=1e-6):
        raise ValueError("时间轴与采样间隔不匹配")
    if not np.allclose(absolute_times, times + ik.config["source"]["start_time_s"]):
        raise ValueError("源时间戳与当前配置不匹配")
    if not np.allclose(motion["root_quaternions_xyzw"], [0, 0, 0, 1]):
        raise ValueError("当前脚本预期输入为竖直骨盆的 v1 数据")

    human_names, human_xyz = sample_human(ik.config, absolute_times)
    rpy, waist, lean_bias, raw_pelvis, raw_relative = extract_upper_pose(human_names, human_xyz)
    root_rotations = Rotation.from_euler("xyz", rpy).as_matrix()
    _, _, _, limits = read_urdf(PROJECT_ROOT / ik.config["target"]["urdf_file"])
    waist_lower, waist_upper = limits["waist_yaw_joint"]
    if np.any(waist < waist_lower) or np.any(waist > waist_upper):
        raise ValueError("腰部角度超过 URDF 限位")
    for side in SIDES:
        index = ik.leg_names.index(f"{side}_knee_joint")
        ik.lower[index] = max(ik.lower[index], MIN_KNEE_RAD)
    neutral_feet = ik.foot_poses(ik.default)

    positions = []
    diagnostics = []
    world_errors = []
    seed = ik.default.copy()
    for frame in range(len(times)):
        rotation = root_rotations[frame]
        # 把不变的世界脚目标变换到骨盆坐标系，复用旧 IK，无需改旧文件。
        goals = {side: (
            rotation.T @ (foot_targets[frame, index] - roots[frame]),
            rotation.T @ neutral_feet[side][1],
        ) for index, side in enumerate(SIDES)}
        angles, check = ik.solve(goals, initial=seed, base_height=0.0)
        seed = angles.copy()
        # 腰关节不在腿部运动链上；只更新输出，不改变下一帧 IK 的腰部种子。
        angles["waist_yaw_joint"] = float(waist[frame])
        poses = ik.foot_poses(angles, base_height=0.0)
        error = max(float(np.linalg.norm(rotation @ poses[side][0] + roots[frame]
                                         - foot_targets[frame, index]))
                    for index, side in enumerate(SIDES))
        positions.append([angles[name] for name in ik.names])
        diagnostics.append(check)
        world_errors.append(error)
        if frame % 50 == 0:
            print(f"frame={frame}/{len(times)} accepted={check['accepted']}")

    positions = np.asarray(positions)
    velocities = np.gradient(positions, dt, axis=0)
    failed = [frame for frame, item in enumerate(diagnostics)
              if not item["accepted"] or world_errors[frame] > 0.001]
    # 相邻旋转的相对旋转向量用于估计世界坐标角速度，不把欧拉角导数冒充角速度。
    step_rates = Rotation.from_matrix(root_rotations[1:] @ np.swapaxes(
        root_rotations[:-1], -1, -2)).as_rotvec() / dt
    angular_velocity = np.vstack([step_rates[0],
                                  0.5 * (step_rates[:-1] + step_rates[1:]),
                                  step_rates[-1]])
    report = {
        "status": "upper_pose_candidate_not_dynamics_validated",
        "valid_for_training": False,
        "parent_motion_file": str(INPUT_FILE),
        "source": ik.config["source"],
        "frame_count": len(times),
        "joint_names": ik.names,
        "parameters": {"pelvis_gains_rpy": PELVIS_GAINS.tolist(),
                       "torso_lean_gain": TORSO_LEAN_GAIN,
                       "waist_yaw_gain": WAIST_YAW_GAIN,
                       "root_angle_limit_deg": ROOT_ANGLE_LIMIT_DEG.tolist(),
                       "waist_limit_deg": WAIST_LIMIT_DEG,
                       "smooth_window_frames": SMOOTH_WINDOW},
        "root_rpy_min_deg": np.degrees(rpy).min(axis=0).tolist(),
        "root_rpy_max_deg": np.degrees(rpy).max(axis=0).tolist(),
        "lean_bias_deg": float(np.degrees(lean_bias)),
        "waist_yaw_range_deg": [float(np.degrees(waist).min()), float(np.degrees(waist).max())],
        "max_joint_speed_rad_s": float(np.abs(velocities).max()),
        "max_world_foot_position_error_m": max(world_errors),
        "max_foot_orientation_error_deg": max(item["max_orientation_error_deg"] for item in diagnostics),
        "all_frames_ik_accepted": not failed,
        "failed_frames": failed,
        "limitations": ["Anatomical landmark pose proxies, not full human pose reproduction",
                        "Heuristic stance labels and unchanged flat foot targets from v1",
                        "No collision, torque, support-force or dynamic tracking validation",
                        "No loop blending; robot has only one waist yaw joint"],
        "frame_diagnostics": diagnostics,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    if failed:
        raise RuntimeError(f"{len(failed)} 帧不合格，只保存报告，不输出动作")
    motion.update({
        "joint_positions_rad": positions,
        "joint_velocities_rad_s": velocities,
        "root_quaternions_xyzw": Rotation.from_matrix(root_rotations).as_quat(),
        "root_linear_velocities_world_m_s": np.gradient(roots, dt, axis=0),
        "root_angular_velocities_world_rad_s": angular_velocity,
        "root_rpy_rad": rpy,
        "source_pelvis_proxy_rpy_rad": raw_pelvis,
        "source_torso_relative_proxy_rpy_rad": raw_relative,
        "valid_for_training": np.array(False),
    })
    np.savez_compressed(OUTPUT_DIR / "candidate_motion.npz", **motion)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for index, label in enumerate(("roll", "pitch", "yaw")):
        axes[0, 0].plot(times, np.degrees(rpy[:, index]), label=label)
    axes[0, 0].set_title("Robot pelvis angles (deg)")
    axes[0, 1].plot(times, np.degrees(waist), label="waist yaw")
    axes[0, 1].set_title("Robot waist yaw (deg)")
    for index, side in enumerate(SIDES):
        axes[1, 0].plot(times, foot_targets[:, index, 2], label=side)
    axes[1, 0].set_title("Unchanged foot target heights (m)")
    axes[1, 1].plot(times, np.max(np.abs(velocities), axis=1), label="max |dq|")
    axes[1, 1].set_title("Max joint speed (rad/s): numerical derivative")
    for axis in axes.flat:
        axis.grid(True)
        axis.legend()
        axis.set_xlabel("Time (s)")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "diagnostics.png", dpi=150)
    plt.close(figure)
    print(f"IK 通过：{len(times)}/{len(times)}")
    print("骨盆 RPY 最小角度(deg)：", np.round(report["root_rpy_min_deg"], 2))
    print("骨盆 RPY 最大角度(deg)：", np.round(report["root_rpy_max_deg"], 2))
    print("腰部角度范围(deg)：", np.round(report["waist_yaw_range_deg"], 2))
    print(f"最大关节速度：{report['max_joint_speed_rad_s']:.3f} rad/s")
    print(f"输出：{OUTPUT_DIR}")
    print("仅完成几何姿态候选；没有验证动态平衡或力矩，不能用于训练。")


if __name__ == "__main__":
    main()
