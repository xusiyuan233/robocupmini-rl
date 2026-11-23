"""读取经过验证的示范，并构建风格学习特征。

本文件只检查和转换数据：
- 不启动 IsaacLab；
- 不训练策略或判别器；
- 不修改原始示范文件。
"""

from pathlib import Path

import numpy as np

from tools.motion.verified_style_data import load_verified_motion


ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    ROOT
    / "data/motions/robotcupmini/style_reference_manifest.json"
)

# 专家示范和机器人在线特征必须共同遵守这个顺序。
JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
)

FEATURE_DIM = 42


def world_to_body(
    vectors: np.ndarray,
    quaternions_xyzw: np.ndarray,
) -> np.ndarray:
    """将世界坐标系中的向量旋转到机器人根部坐标系。

    注意：这里读取的示范四元数顺序是 xyzw。
    """
    xyz = quaternions_xyzw[..., :3]
    w = quaternions_xyzw[..., 3:4]

    cross = np.cross(xyz, vectors)

    # 单位四元数的逆旋转，不包含平移。
    return (
        vectors
        - 2.0 * w * cross
        + 2.0 * np.cross(xyz, cross)
    )


def load_expert_features(
    manifest_path: Path = MANIFEST_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """返回单帧特征和相邻帧特征对。

    单帧特征：[帧数, 42]
    相邻帧特征对：[帧数 - 1, 84]
    """
    # 校验示范、物理报告、风格报告以及关节顺序。
    verified = load_verified_motion(
        manifest_path,
        expected_joint_names=list(JOINT_NAMES),
    )

    arrays = verified.arrays

    root_positions = arrays["root_positions_m"]
    root_quaternions = arrays["root_quaternions_xyzw"]
    num_frames = len(arrays["time_s"])

    # 使用绝对关节角度，单位 rad。
    joint_positions = arrays["joint_positions_rad"]

    # 关节速度，单位 rad/s。
    joint_velocities = arrays["joint_velocities_rad_s"]

    # 根部线速度转换到根部坐标系，单位 m/s。
    linear_velocity_body = world_to_body(
        arrays["root_linear_velocities_world_m_s"],
        root_quaternions,
    )

    # 根部角速度转换到根部坐标系，单位 rad/s。
    angular_velocity_body = world_to_body(
        arrays["root_angular_velocities_world_rad_s"],
        root_quaternions,
    )

    # 单位重力方向，用于描述机器人倾斜情况。
    gravity_world = np.broadcast_to(
        np.array([0.0, 0.0, -1.0]),
        (num_frames, 3),
    )

    gravity_body = world_to_body(
        gravity_world,
        root_quaternions,
    )

    # 脚底参考点相对于根部的位置，再转换到根部坐标系。
    # 顺序：左脚 xyz，然后右脚 xyz。
    feet_relative_world = (
        arrays["foot_targets_world_m"]
        - root_positions[:, None, :]
    )

    feet_relative_body = world_to_body(
        feet_relative_world,
        root_quaternions[:, None, :],
    ).reshape(num_frames, 6)

    # 当前示范为地面 z=0 的平地，因此根部 z 即相对地面高度。
    root_height = root_positions[:, 2:3]

    features = np.concatenate(
        [
            joint_positions,          # 13维
            joint_velocities,         # 13维
            linear_velocity_body,     # 3维
            angular_velocity_body,    # 3维
            gravity_body,             # 3维
            feet_relative_body,       # 6维
            root_height,              # 1维
        ],
        axis=-1,
    ).astype(np.float32)

    if features.shape != (num_frames, FEATURE_DIM):
        raise ValueError(
            f"风格特征维度错误：{features.shape}"
        )

    if not np.isfinite(features).all():
        raise ValueError("风格特征中存在 NaN 或 Inf")

    # 每个样本包含连续两帧，表示“从什么状态运动到什么状态”。
    # 不连接最后一帧和第一帧：这段示范没有经过循环接缝验证。
    transitions = np.concatenate(
        [features[:-1], features[1:]],
        axis=-1,
    )

    return features, transitions


def main() -> None:
    features, transitions = load_expert_features()

    print("专家单帧特征：", features.shape)
    print("专家相邻帧特征对：", transitions.shape)
    print("数据全部有限：", bool(np.isfinite(transitions).all()))
    print("最后一帧与第一帧连接：否")
    print("仅完成数据检查，未启动训练。")


if __name__ == "__main__":
    main()