"""从 v0 生成支撑贴地候选；接触由几何推断，不是力传感器真值。"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicHermiteSpline

from robot_leg_ik import RobotLegIK

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = (PROJECT_ROOT / "data/motions/robotcupmini/retargeted"
              / "walk1_candidate_v0/candidate_motion.npz")
OUTPUT_DIR = (PROJECT_ROOT / "data/motions/robotcupmini/retargeted"
              / "walk1_candidate_v1_contact")
SIDES = ("left", "right")
CONTACT_HEIGHT_M = 0.022
CONTACT_SPEED_M_S = 0.12
MIN_STANCE_TIME_S = 0.10
EXTRA_CROUCH_M = 0.010
MIN_KNEE_RAD = 0.05


def true_runs(mask: np.ndarray):
    """返回连续 True 段：[开始帧，结束帧)，不包含结束帧。"""
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def adjust_one_foot(times, raw, velocity, mask):
    """支撑段固定位置；摆动段保留水平轨迹并平滑衔接高度。"""
    intervals = true_runs(mask)
    if not intervals:
        raise ValueError("某只脚没有可靠的支撑候选段")
    result = raw.copy()
    anchors = []
    for start, stop in intervals:
        anchor = np.median(raw[start:stop], axis=0)
        anchor[2] = 0.0
        anchors.append(anchor)
        result[start:stop] = anchor

    # 每段摆动的区间及其两端的修正量、修正量导数。
    gaps = []
    for index in range(len(intervals) - 1):
        left = intervals[index][1] - 1
        right = intervals[index + 1][0]
        gaps.append((left, right, True, True,
                     anchors[index] - raw[left], anchors[index + 1] - raw[right],
                     -velocity[left], -velocity[right]))
    first = intervals[0][0]
    if first > 0:
        delta = anchors[0] - raw[first]
        gaps.append((0, first, False, True,
                     delta, delta, np.zeros(3), -velocity[first]))
    last = intervals[-1][1] - 1
    if last < len(times) - 1:
        delta = anchors[-1] - raw[last]
        gaps.append((last, len(times) - 1, True, False,
                     delta, delta, -velocity[last], np.zeros(3)))

    baseline = float(np.median(raw[mask, 2]))
    for left, right, start_contact, end_contact, d0, d1, v0, v1 in gaps:
        indices = np.arange(left, right + 1)
        correction = CubicHermiteSpline(
            times[[left, right]], np.stack([d0, d1]), np.stack([v0, v1]))
        result[indices] = raw[indices] + correction(times[indices])

        # 不直接把原始踝高度当脚底高度。
        # 使用源轨迹的抬脚峰值及峰值时刻，构造无负高度的摆动轮廓。
        peak = left + int(np.argmax(raw[left:right + 1, 2]))
        knots = sorted(set([left, peak, right]))
        heights = np.maximum(raw[knots, 2] - baseline, 0.0)
        if start_contact:
            heights[0] = 0.0
        if end_contact:
            heights[-1] = 0.0
        lift_curve = CubicHermiteSpline(times[knots], heights, np.zeros(len(knots)))
        result[indices, 2] = lift_curve(times[indices])

    # 保证支撑段与前面设置的锚点完全一致。
    for (start, stop), anchor in zip(intervals, anchors):
        result[start:stop] = anchor
    if not np.isfinite(result).all() or result[:, 2].min() < -1e-9:
        raise ValueError("修正后的脚部轨迹存在无效值或穿地目标")
    result[:, 2] = np.maximum(result[:, 2], 0.0)
    return result, intervals


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{OUTPUT_DIR}")
    with np.load(INPUT_FILE, allow_pickle=False) as archive:
        motion = {name: archive[name].copy() for name in archive.files}
    times = motion["time_s"]
    dt = float(motion["sample_dt_s"].item())
    foot = motion["foot_targets_world_m"]
    roots = motion["root_positions_m"].copy()
    names = motion["joint_names"].tolist()
    if (len(times) < 3 or dt <= 0 or not np.isfinite(dt)
            or foot.shape != (len(times), 2, 3)
            or roots.shape != (len(times), 3)):
        raise ValueError("输入形状、帧数或采样间隔无效")
    if not all(np.isfinite(array).all() for array in (times, foot, roots)):
        raise ValueError("输入存在 NaN 或 Inf")
    if not np.allclose(np.diff(times), dt, atol=1e-8, rtol=1e-6):
        raise ValueError("时间轴不匹配")
    if not np.allclose(motion["root_quaternions_xyzw"], [0, 0, 0, 1]):
        raise ValueError("当前版本只支持 v0 的竖直骨盆姿态")
    ik = RobotLegIK(PROJECT_ROOT)
    if names != ik.names:
        raise ValueError("关节顺序与项目配置不一致")
    if not np.allclose(motion["joint_positions_rad"][:, names.index("waist_yaw_joint")],
                       ik.default["waist_yaw_joint"]):
        raise ValueError("当前版本只支持固定腰部")

    velocity = np.gradient(foot, dt, axis=0)
    horizontal_speed = np.linalg.norm(velocity[..., :2], axis=2)
    contact = ((foot[..., 2] < CONTACT_HEIGHT_M)
               & (horizontal_speed < CONTACT_SPEED_M_S))
    min_frames = max(2, int(np.ceil(MIN_STANCE_TIME_S / dt)))
    for side in range(2):
        for start, stop in true_runs(contact[:, side]):
            if stop - start < min_frames:
                contact[start:stop, side] = False
    if not contact.any(axis=1).all():
        raise ValueError("出现双脚均无支撑候选的帧，请先检查诊断，不自动补造接触")

    corrected = foot.copy()
    stance_intervals = {}
    for index, side in enumerate(SIDES):
        corrected[:, index], intervals = adjust_one_foot(
            times, foot[:, index], velocity[:, index], contact[:, index])
        stance_intervals[side] = [[int(a), int(b)] for a, b in intervals]

    # 贴地会改变腿的伸展要求，适当降低骨盆后重新求解，不能只改显示地面。
    ground_shift = float(np.median(foot[..., 2][contact]))
    total_shift = ground_shift + EXTRA_CROUCH_M
    roots[:, 2] -= total_shift
    for side in SIDES:
        index = ik.leg_names.index(f"{side}_knee_joint")
        ik.lower[index] = max(ik.lower[index], MIN_KNEE_RAD)
    neutral_feet = ik.foot_poses(ik.default)
    angles_sequence = []
    diagnostics = []
    seed = ik.default.copy()
    for frame in range(len(times)):
        xy_shift = np.r_[roots[frame, :2], 0.0]
        goals = {side: (corrected[frame, index] - xy_shift, neutral_feet[side][1])
                 for index, side in enumerate(SIDES)}
        angles, check = ik.solve(goals, initial=seed, base_height=roots[frame, 2])
        angles_sequence.append([angles[name] for name in names])
        diagnostics.append(check)
        seed = angles
        if frame % 50 == 0:
            print(f"frame={frame}/{len(times)} accepted={check['accepted']}")

    positions = np.asarray(angles_sequence)
    velocities = np.gradient(positions, dt, axis=0)
    failed = [index for index, item in enumerate(diagnostics) if not item["accepted"]]
    report = {
        "status": "contact_adjusted_candidate_not_dynamics_validated",
        "valid_for_training": False,
        "parent_motion_file": str(INPUT_FILE),
        "joint_names": names,
        "frame_count": len(times),
        "parameters": {"contact_height_m": CONTACT_HEIGHT_M,
                       "contact_speed_m_s": CONTACT_SPEED_M_S,
                       "min_stance_time_s": MIN_STANCE_TIME_S,
                       "extra_crouch_m": EXTRA_CROUCH_M,
                       "min_knee_rad": MIN_KNEE_RAD},
        "root_lowering_m": total_shift,
        "stance_intervals_exclusive_end": stance_intervals,
        "contact_candidate_frame_counts": contact.sum(axis=0).tolist(),
        "frames_without_support_candidate": int((~contact.any(axis=1)).sum()),
        "max_contact_target_height_m": float(np.abs(corrected[..., 2][contact]).max()),
        "max_position_error_m": max(item["max_position_error_m"] for item in diagnostics),
        "max_orientation_error_deg": max(item["max_orientation_error_deg"] for item in diagnostics),
        "max_joint_speed_rad_s": float(np.abs(velocities).max()),
        "all_frames_ik_accepted": not failed,
        "failed_frames": failed,
        "limitations": ["Heuristic contact candidates, not measured forces",
                        "Upright pelvis; fixed waist; default foot orientations",
                        "No collision, torque or dynamic tracking validation",
                        "No loop blending; source foot markers not fully calibrated"],
        "frame_diagnostics": diagnostics,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if failed:
        raise RuntimeError(f"{len(failed)} 帧 IK 不合格，只保存报告，不输出动作")
    motion.update({
        "joint_positions_rad": positions,
        "joint_velocities_rad_s": velocities,
        "root_positions_m": roots,
        "foot_targets_world_m": corrected,
        "contact_candidates": contact,
        "foot_names": np.asarray(SIDES),
        "valid_for_training": np.array(False),
    })
    np.savez_compressed(OUTPUT_DIR / "candidate_motion.npz", **motion)

    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for index, side in enumerate(SIDES):
        axes[0, 0].plot(times, foot[:, index, 2], "--", alpha=0.5, label=f"{side} v0")
        axes[0, 0].plot(times, corrected[:, index, 2], label=f"{side} v1")
        axes[0, 1].step(times, contact[:, index].astype(int) + 1.2 * index,
                        where="post", label=side)
    axes[0, 0].set_title("Foot target height (m)")
    axes[0, 1].set_title("Inferred stance candidates (offset for display)")
    axes[1, 0].plot(times, motion["root_positions_m"][:, 2] + total_shift, label="v0")
    axes[1, 0].plot(times, roots[:, 2], label="v1")
    axes[1, 0].set_title("Pelvis height (m)")
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
    print(f"支撑候选帧数 left/right：{contact.sum(axis=0).tolist()}")
    print(f"骨盆整体下移：{total_shift * 1000:.2f} mm")
    print(f"最大关节速度：{report['max_joint_speed_rad_s']:.3f} rad/s")
    print(f"输出：{OUTPUT_DIR}")
    print("仅完成接触候选的几何修正；尚未验证物理支撑，不能用于训练。")


if __name__ == "__main__":
    main()
