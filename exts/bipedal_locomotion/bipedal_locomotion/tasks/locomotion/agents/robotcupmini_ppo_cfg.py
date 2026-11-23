from isaaclab.utils import configclass

from .limx_rsl_rl_ppo_cfg import SF_TRON2AFlatPPORunnerCfg


@configclass
class RobotcupMiniStandPPORunnerCfg(SF_TRON2AFlatPPORunnerCfg):
    # 训练结果单独保存，避免与官方 SF 实验混在一起
    experiment_name = "robotcupmini_stand"

    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50

    def __post_init__(self):
        # 初期使用较小的探索噪声和学习率
        self.policy.init_noise_std = 0.3
        self.algorithm.learning_rate = 3.0e-4

        # 保持与继承的 SF 算法路径一致
        self.empirical_normalization = False
        self.policy.actor_obs_normalization = False
        self.policy.critic_obs_normalization = False

@configclass
class RobotcupMiniVelocityPPORunnerCfg(RobotcupMiniStandPPORunnerCfg):
    experiment_name = "robotcupmini_velocity"

    max_iterations = 2000
    save_interval = 50

    def __post_init__(self):
        super().__post_init__()

        # 已有行走策略的低学习率微调。
        self.algorithm.learning_rate = 1.0e-5
        self.algorithm.schedule = "fixed"

@configclass
class RobotcupMiniStylePrepPPORunnerCfg(
    RobotcupMiniVelocityPPORunnerCfg
):
    """风格学习准备阶段的配置。

    当前仍使用原来的 PPO，尚未接入 AMP。
    独立设置日志名称，避免与旧实验混淆。
    """

    experiment_name = "robotcupmini_style_prep"

    save_interval = 500