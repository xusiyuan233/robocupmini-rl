"""从 IsaacLab 机器人读取42维风格特征。

需要在 AppLauncher 启动之后导入本模块。
"""

from typing import cast

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from tools.amp.motion_loader import JOINT_NAMES


class IsaacAmpFeatures:
    """与专家数据保持一致的在线风格特征提取器。"""

    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env
        self.robot = cast(Articulation, env.scene["robot"])

        # 明确检查根部，不能误用上半身作为骨盆。
        if self.robot.body_names[0] != "pelvis":
            raise ValueError(
                "当前机器人根部不是预期的 pelvis，请检查模型。"
            )

        # 按名字映射，不依赖 IsaacLab 内部的默认关节排列。
        self.joint_ids = [
            self.robot.joint_names.index(name)
            for name in JOINT_NAMES
        ]

        self.foot_ids = [
            self.robot.body_names.index("left_ankle_roll_link"),
            self.robot.body_names.index("right_ankle_roll_link"),
        ]

        # 与当前 MuJoCo 示范使用的脚底参考点一致。
        # 坐标位于各自 ankle_roll_link 的局部坐标系中，单位 m。
        # 顺序：左脚、右脚。
        self.sole_offsets = torch.tensor(
            [
                [0.0499999222, -0.0001674525, -0.0299998040],
                [0.0400001212, -0.0001524757, -0.0299998038],
            ],
            dtype=torch.float32,
            device=env.device,
        )

    @torch.no_grad()
    def get(self) -> torch.Tensor:
        """返回形状为 [环境数量, 42] 的特征。"""
        data = self.robot.data
        num_envs = self.env.num_envs

        # 明确使用根连杆原点，不混用根连杆质心。
        root_pos = data.root_link_pos_w
        root_quat = data.root_link_quat_w

        joint_pos = data.joint_pos[:, self.joint_ids]
        joint_vel = data.joint_vel[:, self.joint_ids]

        # 根连杆原点的速度，表达在根部坐标系中。
        linear_velocity_body = data.root_link_lin_vel_b
        angular_velocity_body = data.root_link_ang_vel_b

        gravity_world = root_pos.new_tensor(
            [0.0, 0.0, -1.0]
        ).expand(num_envs, 3)

        gravity_body = quat_apply_inverse(
            root_quat,
            gravity_world,
        )

        # 脚部连杆原点不是脚底参考点，需要加上局部偏移。
        foot_pos = data.body_link_pos_w[:, self.foot_ids]
        foot_quat = data.body_link_quat_w[:, self.foot_ids]

        offsets = self.sole_offsets.unsqueeze(0).expand(
            num_envs, -1, -1
        )

        sole_positions_world = foot_pos + quat_apply(
            foot_quat.reshape(-1, 4),
            offsets.reshape(-1, 3),
        ).reshape(num_envs, 2, 3)

        root_quat_for_feet = root_quat.unsqueeze(1).expand(
            -1, 2, -1
        )

        feet_relative_body = quat_apply_inverse(
            root_quat_for_feet.reshape(-1, 4),
            (sole_positions_world - root_pos.unsqueeze(1)).reshape(-1, 3),
        ).reshape(num_envs, 6)

        # 当前使用平地，减去各并行环境原点的高度。
        root_height = (
            root_pos[:, 2:3]
            - self.env.scene.env_origins[:, 2:3]
        )

        return torch.cat(
            [
                joint_pos,                 # 13
                joint_vel,                 # 13
                linear_velocity_body,      # 3
                angular_velocity_body,     # 3
                gravity_body,              # 3
                feet_relative_body,        # 6
                root_height,               # 1
            ],
            dim=-1,
        )