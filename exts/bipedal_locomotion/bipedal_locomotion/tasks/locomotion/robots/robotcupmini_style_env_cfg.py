"""RobotcupMini 风格学习准备环境。

当前阶段：
1. 继承已有行走环境；
2. 单独注册任务，避免影响原来的速度任务；
3. 固定直线前进指令，便于后续对照；
4. 尚未加入 AMP 判别器和风格奖励。
"""

from isaaclab.utils import configclass

from .robotcupmini_velocity_env_cfg import RobotcupMiniVelocityEnvCfg


@configclass
class RobotcupMiniStylePrepEnvCfg(RobotcupMiniVelocityEnvCfg):
    """风格学习准备配置，目前仍使用原来的任务奖励。"""

    def __post_init__(self):
        # 先创建现有行走环境的机器人、观测、动作和奖励。
        super().__post_init__()

        velocity = self.commands.base_velocity

        # 关闭自动航向修正。
        # 否则即使设置 ang_vel_z=0，航向控制器仍可能生成转向指令。
        velocity.heading_command = False
        velocity.rel_heading_envs = 0.0

        # 准备阶段只测试行走，不混入随机站立。
        velocity.rel_standing_envs = 0.0

        # 固定直线前进指令，方便比较不同版本。
        velocity.ranges.lin_vel_x = (0.13, 0.13)
        velocity.ranges.lin_vel_y = (0.0, 0.0)
        velocity.ranges.ang_vel_z = (0.0, 0.0)

        velocity.debug_vis = False
        # 风格训练专用覆盖，不改变原来的速度任务。
        spacing = self.rewards.feet_lateral_spacing

        if spacing is None:
            raise ValueError("风格训练需要先创建步宽奖励")

        # 与当前示范的步宽对齐。
        # 保留过窄、交叉和过宽惩罚，不直接关闭该项。
        spacing.params.update(
            {
                "min_width": 0.12,
                "max_width": 0.20,
                "target_width": 0.15,
            }
        )

        height_reward = self.rewards.base_height_exp

        if height_reward is None:
            raise ValueError("当前环境没有根部高度奖励")

        # 与当前示范的平均根部高度对齐。
        height_reward.params["target_height"] = 0.53

        # 暂不修改：
        # - 机器人模型与 PD 参数；
        # - 观测维度与关节顺序；
        # - 动作缩放；
        # - 原有奖励权重。
        #
        # 等参考动作验证完成，再逐项确定 AMP 所需的改动。


@configclass
class RobotcupMiniStylePrepEnvCfg_PLAY(RobotcupMiniStylePrepEnvCfg):
    """风格学习准备环境的回放配置。"""

    def __post_init__(self):
        super().__post_init__()

        # 回放只显示一个机器人。
        self.scene.num_envs = 1

        # 初始关节不再加入位置扰动，便于对照观察。
        self.events.reset_robot_joints.params["position_range"] = (
            0.0,
            0.0,
        )