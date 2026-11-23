"""生成几何重定向候选；不启动训练，也不认定为合格示范数据。"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from preview_lafan1 import load_bvh
from robot_leg_ik import RobotLegIK

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (PROJECT_ROOT / "data/motions/robotcupmini/retargeted"
              / "walk1_candidate_v0")

# 第一版几何适配参数，不是训练参数，也不是自动标定结果。
HORIZONTAL_SCALE = 0.55
LATERAL_SCALE = 0.30
LIFT_SCALE = 0.40
BASE_HEIGHT_M = 0.535
GROUND_CLEARANCE_M = 0.005
MIN_KNEE_RAD = 0.05


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{OUTPUT_DIR}")

    ik = RobotLegIK(PROJECT_ROOT)
    config = ik.config
    source = config["source"]
    if (source["up_axis"], config["target"]["up_axis"]) != ("Y", "Z"):
        raise ValueError("当前脚本只支持源 Y 向上、目标 Z 向上")
    if source["position_unit"] != "cm":
        raise ValueError("当前 BVH 读取器按厘米输入")

    names, _, positions, source_dt = load_bvh(
        PROJECT_ROOT / source["file"])
    start, end = source["start_time_s"], source["end_time_s"]
    dt = float(config["target"]["sample_dt_s"])
    if not (0 <= start < end <= (len(positions) - 1) * source_dt):
        raise ValueError("源片段时间范围无效")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("输出采样间隔无效")

    # 在原始数据覆盖范围内插值；不把 30 Hz 的帧直接当成 50 Hz。
    times = np.arange(0.0, end - start, dt)
    if len(times) < 3:
        raise ValueError("重采样后至少需要 3 帧")
    absolute_times = start + times
    source_times = np.arange(len(positions)) * source_dt
    flat = positions.reshape(len(positions), -1)
    sampled = np.column_stack([
        np.interp(absolute_times, source_times, flat[:, column])
        for column in range(flat.shape[1])
    ]).reshape(len(times), len(names), 3)

    # (X, Y, Z) -> (X, -Z, Y)，然后对齐片段整体前进方向。
    xyz = sampled[..., [0, 2, 1]].copy()
    xyz[..., 1] *= -1
    pelvis_index = names.index(config["landmarks"]["pelvis"]["human_joint"])
    xyz[..., :2] -= xyz[0, pelvis_index, :2].copy()
    displacement = xyz[-1, pelvis_index, :2]
    if np.linalg.norm(displacement) < 0.1:
        raise ValueError("净位移过小，当前工具只处理前进片段")
    yaw = np.arctan2(displacement[1], displacement[0])
    c, s = np.cos(yaw), np.sin(yaw)
    xyz[..., :2] = xyz[..., :2] @ np.array([[c, -s], [s, c]])

    pelvis = xyz[:, pelvis_index]
    sides = ("left", "right")
    ankle_indices = [names.index(
        config["landmarks"][f"{side}_ankle"]["human_joint"])
        for side in sides]
    ankles = xyz[:, ankle_indices]
    relative = ankles - pelvis[:, None, :]
    neutral_feet = ik.foot_poses(ik.default)

    # 不修改 URDF；仅收紧本次重定向的求解边界，避免膝盖反向弯曲。
    for side in sides:
        index = ik.leg_names.index(f"{side}_knee_joint")
        ik.lower[index] = max(ik.lower[index], MIN_KNEE_RAD)
        if not ik.lower[index] <= ik.default[f"{side}_knee_joint"] <= ik.upper[index]:
            raise ValueError("默认膝角与重定向求解边界不兼容")

    root_positions = np.zeros((len(times), 3))
    root_positions[:, :2] = HORIZONTAL_SCALE * pelvis[:, :2]
    root_positions[:, 2] = BASE_HEIGHT_M + HORIZONTAL_SCALE * (
        pelvis[:, 2] - np.median(pelvis[:, 2]))

    # 这是踝部高度的几何代理，不是真实脚底标定或接触检测。
    ankle_baseline = float(ankles[..., 2].min())
    x_center = float(relative[..., 0].mean())
    targets = np.zeros((len(times), 2, 3))
    for index, side in enumerate(sides):
        neutral_position, _ = neutral_feet[side]
        targets[:, index, 0] = neutral_position[0] + HORIZONTAL_SCALE * (
            relative[:, index, 0] - x_center)
        targets[:, index, 1] = neutral_position[1] + LATERAL_SCALE * (
            relative[:, index, 1] - np.median(relative[:, index, 1]))
        targets[:, index, 2] = GROUND_CLEARANCE_M + LIFT_SCALE * (
            ankles[:, index, 2] - ankle_baseline)

    angles_sequence = []
    diagnostics = []
    seed = ik.default.copy()
    for frame in range(len(times)):
        # 第一版保持双脚朝向默认站姿，不复制人体脚部旋转。
        goals = {side: (targets[frame, index], neutral_feet[side][1])
                 for index, side in enumerate(sides)}
        angles, check = ik.solve(
            goals, initial=seed, base_height=root_positions[frame, 2])
        angles_sequence.append([angles[name] for name in ik.names])
        diagnostics.append(check)
        seed = angles
        if frame % 50 == 0:
            print(f"frame={frame}/{len(times)} accepted={check['accepted']}")

    joint_positions = np.asarray(angles_sequence)
    joint_velocities = np.gradient(joint_positions, dt, axis=0)
    accepted = np.array([item["accepted"] for item in diagnostics], dtype=bool)
    position_errors = np.array([item["max_position_error_m"] for item in diagnostics])
    orientation_errors = np.array([item["max_orientation_error_deg"] for item in diagnostics])
    failed_frames = np.flatnonzero(~accepted).tolist()
    all_passed = bool(accepted.all())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "geometric_candidate_only_not_validated",
        "valid_for_training": False,
        "source": source,
        "sample_dt_s": dt,
        "frame_count": len(times),
        "last_sample_time_s": float(times[-1]),
        "joint_names": ik.names,
        "parameters": {
            "horizontal_scale": HORIZONTAL_SCALE,
            "lateral_scale": LATERAL_SCALE,
            "lift_scale": LIFT_SCALE,
            "base_height_m": BASE_HEIGHT_M,
            "ground_clearance_m": GROUND_CLEARANCE_M,
            "min_knee_rad": MIN_KNEE_RAD,
        },
        "all_frames_ik_accepted": all_passed,
        "failed_frames": failed_frames,
        "max_position_error_m": float(position_errors.max()),
        "max_orientation_error_deg": float(orientation_errors.max()),
        "max_joint_speed_rad_s": float(np.abs(joint_velocities).max()),
        "assumptions": [
            "Upright pelvis; fixed waist; default foot orientations",
            "Ankle-height proxy, not calibrated sole/contact measurements",
            "No collision, contact, torque or dynamics validation",
            "No loop blending or time stretching",
        ],
        "frame_diagnostics": diagnostics,
    }
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 失败时仍保留诊断，但不输出容易被误用的关节动作文件。
    if not all_passed:
        raise RuntimeError(f"{len(failed_frames)} 帧 IK 未通过；只保存报告：{OUTPUT_DIR}")

    foot_targets_world = targets.copy()
    foot_targets_world[..., :2] += root_positions[:, None, :2]
    np.savez_compressed(
        OUTPUT_DIR / "candidate_motion.npz",
        time_s=times,
        source_time_s=absolute_times,
        sample_dt_s=np.array(dt),
        joint_names=np.asarray(ik.names),
        joint_positions_rad=joint_positions,
        joint_velocities_rad_s=joint_velocities,
        root_positions_m=root_positions,
        root_quaternions_xyzw=np.tile([0., 0., 0., 1.], (len(times), 1)),
        foot_targets_world_m=foot_targets_world,
        valid_for_training=np.array(False),
    )

    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for index, side in enumerate(sides):
        axes[0, 0].plot(times, targets[:, index, 2], label=side)
        knee = ik.names.index(f"{side}_knee_joint")
        axes[0, 1].plot(times, np.degrees(joint_positions[:, knee]), label=side)
        axes[1, 0].plot(times, np.linalg.norm(
            np.gradient(foot_targets_world[:, index], dt, axis=0), axis=1), label=side)
    axes[0, 0].set_title("Foot target height (m): geometric proxy")
    axes[0, 1].set_title("Knee angle (deg)")
    axes[1, 0].set_title("Foot target world speed (m/s): no contact labels")
    axes[1, 1].plot(times, np.max(np.abs(joint_velocities), axis=1), label="max |dq|")
    axes[1, 1].set_title("Max joint speed (rad/s): numerical derivative")
    for axis in axes.flat:
        axis.set_xlabel("Time (s)")
        axis.grid(True)
        axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "diagnostics.png", dpi=150)
    plt.close(figure)
    print(f"IK 通过：{int(accepted.sum())}/{len(times)}")
    print(f"最大关节速度：{report['max_joint_speed_rad_s']:.3f} rad/s")
    print(f"输出：{OUTPUT_DIR}")
    print("仅为几何候选，不能接入训练；下一步检查机器人动作和接触约束。")


if __name__ == "__main__":
    main()
