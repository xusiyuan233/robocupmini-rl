import math
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.robotcupmini_cfg import (
    JOINT_NAMES,
    ROBOCUPMINI_CFG,
)

from .limx_solefoot_tron2a_env_cfg import SF_TRON2A_BlindFlatEnvCfg


FOOT_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]


@configclass
class RobotcupMiniStandEnvCfg(SF_TRON2A_BlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 一、场景与机器人
        self.scene.num_envs = 256
        self.scene.robot = ROBOCUPMINI_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.viewer.origin_type = "env"

        # 二、动作：按明确的关节顺序输出位置目标
        self.actions.joint_pos.joint_names = JOINT_NAMES
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = 0.15
        self.actions.joint_pos.use_default_offset = True

        # 三、策略、历史和 Critic 使用相同的关节顺序
        observation_groups = (
            self.observations.policy,
            self.observations.obsHistory,
            self.observations.critic,
        )

        for group in observation_groups:
            group.joint_pos.params["asset_cfg"] = SceneEntityCfg(
                "robot",
                joint_names=JOINT_NAMES,
                preserve_order=True,
            )
            group.joint_vel.params["asset_cfg"] = SceneEntityCfg(
                "robot",
                joint_names=JOINT_NAMES,
                preserve_order=True,
            )

            # 站立阶段暂时不使用步态观测
            group.gait_phase = None
            group.gait_command = None

        # 四、Critic 中的脚部观测改为自己的脚部连杆
        self.observations.critic.feet_lin_vel.params["asset_cfg"] = (
            SceneEntityCfg(
                "robot",
                body_names=FOOT_NAMES,
                preserve_order=True,
            )
        )

        self.observations.critic.feet_contact_force.params["sensor_cfg"] = (
            SceneEntityCfg(
                "contact_forces",
                body_names=FOOT_NAMES,
                preserve_order=True,
            )
        )

        # 五、站立命令：所有环境的目标速度均为零
        self.commands.gait_command = None

        velocity_command = self.commands.base_velocity
        velocity_command.heading_command = False
        velocity_command.ranges.heading = None
        velocity_command.rel_standing_envs = 1.0
        velocity_command.debug_vis = False

        velocity_command.ranges.lin_vel_x = (0.0, 0.0)
        velocity_command.ranges.lin_vel_y = (0.0, 0.0)
        velocity_command.ranges.ang_vel_z = (0.0, 0.0)

        # 初期关闭观测噪声，先训练基础站立
        self.observations.policy.enable_corruption = False
        self.observations.obsHistory.enable_corruption = False
        # 六、初期关闭动力学随机化
        self.events.add_base_mass = None
        self.events.add_link_mass = None
        self.events.randomize_rigid_body_mass_inertia = None
        self.events.robot_physics_material = None
        self.events.randomize_actuator_gains = None
        self.events.robot_center_of_mass = None

        # 关闭原 TRON2 步态辅助初始化
        self.events.prepare_quantity_for_tron1_piper = None

        # 从配置中的默认根姿态开始，初始速度为零
        self.events.reset_robot_base.params["pose_range"] = {}
        self.events.reset_robot_base.params["velocity_range"] = {}

        # 默认关节角附近增加小幅扰动，单位 rad
        self.events.reset_robot_joints.params["position_range"] = (
            -0.015,
            0.015,
        )
        self.events.reset_robot_joints.params["velocity_range"] = (
            0.0,
            0.0,
        )
        # 七、关闭当前站立阶段不使用的奖励
        self.rewards.gait_reward = None
        self.rewards.feet_distance = None
        self.rewards.feet_air_time = None

        self.rewards.knee_joint_orientation = None
        self.rewards.foot_joint_orientation = None
        self.rewards.orientation_exp_knee = None

        self.rewards.dof_power_l1 = None
        self.rewards.foot_contact_force = None
        self.rewards.applied_torque_limits = None
        # 八、站立奖励参数
        self.rewards.base_height_exp.params["target_height"] = 0.566
        self.rewards.base_height_exp.params["std"] = 0.05
        self.rewards.base_height_exp.weight = 2.0

        self.rewards.flat_orientation_l2.weight = -2.0

        # 躯干加速度惩罚改为自己的 pelvis
        self.rewards.body_lin_acc.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            body_names="pelvis",
        )

        # 关节偏离默认姿态的权重改为自己的 13 个关节
        self.rewards.joint_deviation_l1.params["deviation_weight"] = {
            name: 0.02 for name in JOINT_NAMES
        }

        # 除左右脚以外的连杆发生接触时给予惩罚
        self.rewards.undesired_contacts.params["sensor_cfg"] = (
            SceneEntityCfg(
                "contact_forces",
                body_names=(
                    "(?!left_ankle_roll_link$|right_ankle_roll_link$).*"
                ),
            )
        )

        # 足部滑动奖励使用自己的脚部连杆
        for reward_name in ("feet_slide", "feet_ang_slide_penalty"):
            reward_term = getattr(self.rewards, reward_name)

            reward_term.params["sensor_cfg"] = SceneEntityCfg(
                "contact_forces",
                body_names=FOOT_NAMES,
                preserve_order=True,
            )
            reward_term.params["asset_cfg"] = SceneEntityCfg(
                "robot",
                body_names=FOOT_NAMES,
                preserve_order=True,
            )
        # 九、失败终止条件
        self.terminations.base_contact.params["sensor_cfg"] = (
            SceneEntityCfg(
                "contact_forces",
                body_names=["pelvis", "torso_link"],
            )
        )
        self.terminations.bad_orientation.params["limit_angle"] = (
            math.radians(60.0)
        )

        # 十、初期地面参数
        self.scene.terrain.physics_material.static_friction = 0.8
        self.scene.terrain.physics_material.dynamic_friction = 0.8
        self.scene.terrain.physics_material.restitution = 0.0

        self.sim.physics_material = (
            self.scene.terrain.physics_material
        )
@configclass
class RobotcupMiniStandEnvCfg_PLAY(RobotcupMiniStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1

        # 回放从固定的默认关节姿态开始
        self.events.reset_robot_joints.params["position_range"] = (
            0.0,
            0.0,
        )