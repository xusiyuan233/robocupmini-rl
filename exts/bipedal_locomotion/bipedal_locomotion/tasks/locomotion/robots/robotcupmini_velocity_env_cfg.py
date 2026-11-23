# 【奖励修改导航】本文件主要决定速度任务的奖励开关、权重和函数参数。
# 只想调强弱：搜索“[调权重]”，修改 .weight = ... 或 RewTerm 中的 weight=...。
# 想改目标/容差/时序：搜索“[调参数]”，修改 params 里的相应值。
# 想改计算公式：再去 mdp/robotcupmini_rewards.py；不要把计算文件当作权重表。
# 不用另外寻找“初始化后”的独立代码块：下面 RewTerm(...) 中的 weight 本身就是可修改处。
# 同一项可能被重复赋值或被模式分支替换；以当前执行路径最后的设置为准。
# 本次仅添加导航注释并纠正一处角度注释，未改变任何数值、公式或启用状态。
import os
from ..mdp.robotcupmini_rewards import (
    track_forward_velocity_linear,
    feet_phase_height_error,
    feet_lateral_spacing_error,
    feet_forward_heading_error,
    feet_mid_swing_contact_error,
    FeetSwingContinuityError,
    ankle_roll_target_limit_error,
    feet_contact_yaw_motion,
)
from copy import deepcopy
from .limx_solefoot_tron2a_env_cfg import (SF_TRON2A_BlindFlatEnvCfg,)
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.rewards import joint_vel_l2, action_l2
from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.cfg.SF_TRON2A.limx_base_env_cfg import (
    RewardsCfg as BaseRewardsCfg,
)
from .robotcupmini_env_cfg import (
    FOOT_NAMES,
    RobotcupMiniStandEnvCfg,
)
@configclass
class RobotcupMiniVelocityRewardsCfg(BaseRewardsCfg):
    """Declare custom rewards so editors can resolve their names."""

    # [字段声明] 这里只声明自定义奖励名，None 不是权重；真正赋值在 __post_init__ 里。
    track_forward_velocity: RewTerm | None = None
    feet_phase_height: RewTerm | None = None
    waist_velocity_l2: RewTerm | None = None
    feet_lateral_spacing: RewTerm | None = None
    feet_forward_heading: RewTerm | None = None
    feet_mid_swing_contact: RewTerm | None = None
    feet_swing_continuity: RewTerm | None = None
    action_magnitude_l2: RewTerm | None = None
    ankle_roll_target_limits: RewTerm | None = None

@configclass
class RobotcupMiniVelocityEnvCfg(RobotcupMiniStandEnvCfg):
    rewards: RobotcupMiniVelocityRewardsCfg = RobotcupMiniVelocityRewardsCfg()

    def __post_init__(self):
        # [继承入口] 先执行站立父类的设置，本方法后面的配置可以覆盖父类值。
        super().__post_init__()
        # 获取官方环境的步态配置
        reference = SF_TRON2A_BlindFlatEnvCfg()

        # 恢复官方步态命令
        self.commands.gait_command = deepcopy(
            reference.commands.gait_command
        )

        gait = self.commands.gait_command
        gait.debug_vis = False

        # 先使用固定步态参数，减少训练中的变化
        gait.ranges.frequencies = (1.0, 1.0)
        gait.ranges.offsets = (0.5, 0.5)
        gait.ranges.durations = (0.5, 0.5)
        gait.ranges.swing_height = (0.03, 0.03)

        # 恢复策略、历史和Critic中的步态观测
        for group_name in ("policy", "obsHistory", "critic"):
            target_group = getattr(
                self.observations, group_name
            )
            reference_group = getattr(
                reference.observations, group_name
            )

            target_group.gait_phase = deepcopy(
                reference_group.gait_phase
            )
            target_group.gait_command = deepcopy(
                reference_group.gait_command
            )

        # 恢复官方步态奖励
        self.rewards.gait_reward = deepcopy(
            reference.rewards.gait_reward
        )

        gait_reward = self.rewards.gait_reward
        # [调权重] 官方步态奖励总权重；gait_reward 是 self.rewards.gait_reward 的局部别名。
        gait_reward.weight = 1.0

        # 将官方脚部名称替换为自己的脚部连杆
        gait_reward.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=FOOT_NAMES,
            preserve_order=True,
        )
        gait_reward.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            body_names=FOOT_NAMES,
            preserve_order=True,
        )

        # 这一轮先学习按相位卸载摆动脚、保持支撑脚稳定
        # 暂不使用参考足部高度和竖直速度轨迹
        gait_reward.params["use_reference_motion"] = False
        gait_reward.params["tracking_contacts_shaped_height"] = 0.0

        # 一、速度命令
        velocity = self.commands.base_velocity

        # 保持世界坐标系下的初始前进方向。
        velocity.heading_command = True
        velocity.rel_heading_envs = 1.0
        velocity.heading_control_stiffness = 1.0
        velocity.ranges.heading = (0.0, 0.0)

        # 每次重新采样命令时，约 的环境执行零速度站立
        velocity.rel_standing_envs = 0.0

        # 每条命令保持 5～8 秒，再重新采样
        velocity.resampling_time_range = (5.0, 8.0)

        # 第一阶段：低速前进
        velocity.ranges.lin_vel_x = (0.3, 0.5)
        velocity.ranges.lin_vel_y = (0.0, 0.0)
        velocity.ranges.ang_vel_z = (-0.4, 0.4)

        # 训练时先关闭命令箭头
        velocity.debug_vis = False

        # 二、速度跟踪奖励
        # [调权重] 横向速度跟踪权重。
        self.rewards.track_lin_vel_y_exp.weight = 2.0
        # [调参数] 横向速度跟踪函数的 std 尺度，不是整项权重。
        self.rewards.track_lin_vel_y_exp.params["std"] = 0.15

        # [调权重] 转向角速度跟踪权重。
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        # [调参数] 转向速度跟踪函数的 std 尺度。
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.3
        
        # 降低高度奖励在总奖励中的占比
        # [调权重] 身体高度奖励权重；这里会覆盖站立父类中的值。
        self.rewards.base_height_exp.weight = 0.6
        # 三、减轻保持默认关节姿态的约束，允许迈步
        # [调权重] 偏离默认关节姿态的总权重。
        self.rewards.joint_deviation_l1.weight = -0.3

        # 四、恢复官方的单脚支撑辅助奖励
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time_positive_biped,
            # [调权重] 单脚支撑/腾空辅助项权重。 不必新增一段 .weight 赋值，直接改下面这个数。
            weight=1.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=FOOT_NAMES,
                    preserve_order=True,
                ),
            },
        )
        # 五、修正摆动脚接触力奖励的有效范围
        self.rewards.gait_reward.params["gait_force_sigma"] = 1000.0

        # [调参数] 官方步态奖励内部的接触力平滑尺度，具体公式在通用奖励实现中。
        # 保留支撑脚稳定奖励
        # 暂时继续关闭官方参考高度与竖直速度轨迹
        self.rewards.gait_reward.params["use_reference_motion"] = False
        self.rewards.gait_reward.params[
            "tracking_contacts_shaped_height"
        ] = 0.0

        # 六、替换原来的指数前进奖励
        self.rewards.track_lin_vel_x_exp = None

        self.rewards.track_forward_velocity = RewTerm(
            func=track_forward_velocity_linear,
            # [调权重] 前进速度跟踪权重。 不必新增一段 .weight 赋值，直接改下面这个数。
            weight=4.0,
            params={
                "command_name": "base_velocity",
                # [调参数] 前进速度误差的归一化尺度 m/s，不是目标速度。
                "velocity_scale": 0.4,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # 七、增加左右脚分别跟踪相位的高度误差惩罚
        self.rewards.feet_phase_height = RewTerm(
            func=feet_phase_height_error,
            # [调权重] 摆动足底近似高度误差权重。 不必新增一段 .weight 赋值，直接改下面这个数。
            weight=-3.0,
            params={
                "command_name": "gait_command",
                "velocity_command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=FOOT_NAMES,
                    preserve_order=True,
                ),
                # [调参数] 足底相对脚连杆原点的几何偏移 m；它不是希望抬脚的高度，不应随意当训练目标调整。
                "rest_height": 0.03,
                # [调参数] 高度误差函数的目标抬脚峰值 m；它与上方命令里的 swing_height 是不同配置入口。
                "lift_height": 0.03,
                # [调参数] 高度误差归一化尺度 m；同一偏差下减小尺度会放大平方误差。
                "height_scale": 0.03,
            },
        )
        # 抑制腰部快速扭动，允许腰部继续参与平衡
        self.rewards.waist_velocity_l2 = RewTerm(
            func=joint_vel_l2,
            # [调权重] 腰部转动速度惩罚权重。 不必新增一段 .weight 赋值，直接改下面这个数。
            weight=-0.05,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["waist_yaw_joint"],
                ),
            },
        )
        # 减少接触地面时的足部滑动
        # [调权重] 触地脚参考点滑动惩罚权重。
        self.rewards.feet_slide.weight = -0.5
        # 约束左右脚的横向间距，减少交叉落脚。
        self.rewards.feet_lateral_spacing = RewTerm(
            func=feet_lateral_spacing_error,
            # [调权重] 步宽与交叉落脚误差权重。 不必新增一段 .weight 赋值，直接改下面这个数。
            weight=-2.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=FOOT_NAMES,
                    preserve_order=True,
                ),
                # [调参数] 允许的最小左右脚连杆间距 m。
                "min_width": 0.16,
                # [调参数] 允许的最大左右脚连杆间距 m。
                "max_width": 0.28,
                # [调参数] 额外引导的目标左右脚间距 m。
                "target_width": 0.22,
                # [调参数] 步宽误差归一化尺度 m，不是权重。
                "error_scale": 0.10,
            },
        )
        # 加强身体姿态与摇摆约束。
        # [调权重] 身体水平姿态惩罚权重。 [后续会覆盖] 同一属性在本方法后面再次赋值，只改此处不会改变最后值。
        self.rewards.flat_orientation_l2.weight = -6.0
        # [调权重] 身体 roll/pitch 角速度惩罚权重。 [后续会覆盖] 同一属性在本方法后面再次赋值，只改此处不会改变最后值。
        self.rewards.ang_vel_xy_l2.weight = -0.6
        # 减少外八、内八，引导脚尖朝向骨盆前方。
        self.rewards.feet_forward_heading = RewTerm(
            func=feet_forward_heading_error,
            # [调权重] 脚尖相对骨盆朝向偏角惩罚权重。 不必新增一段 .weight 赋值，直接改下面这个数。
            weight=-0.8,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=FOOT_NAMES,
                    preserve_order=True,
                ),
                # [调参数] 允许的脚尖偏角 rad；0.087266 约为 5°，不是 10°。
                "tolerance": 0.087266,   # 约 5°（原注释 10° 有误，数值未改）
                # [调参数] 偏角误差归一化尺度 rad，当前约为 20°。
                "angle_scale": 0.349066, # 20°# 从 v11/model_13000.pt 续训：加强完整摆动的约束。
        # 正向鼓励：摆动相位中，脚部接触力接近零。
            },
        )
        # 左右脚分别在摆动中段卸载、离地。
        self.rewards.feet_mid_swing_contact = RewTerm(
            func=feet_mid_swing_contact_error,
            # [调权重] 摆动中段触地及支撑中段力不足的总惩罚权重。 不必新增一段 .weight 赋值，直接改下面这个数。
            weight=-2.0,
            params={
                "command_name": "gait_command",
                "velocity_command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=FOOT_NAMES,
                    preserve_order=True,
                ),
                # [调参数] 触地力阈值 N；按对应函数分别可能比较合力或竖直力，不是惩罚权重。
                "force_threshold": 5.0,
                # [调参数] 接触力误差归一化尺度 N。
                "force_scale": 50.0,
            },
        )

        # 惩罚一次摆动中提前触地、再次起脚。
        self.rewards.feet_swing_continuity = RewTerm(
            func=FeetSwingContinuityError,
            # [调权重] 摆动中提前触地、重复起脚的总惩罚权重。 不必新增一段 .weight 赋值，直接改下面这个数。
            weight=-4.0,
            params={
                "command_name": "gait_command",
                "velocity_command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=FOOT_NAMES,
                    preserve_order=True,
                ),
                # [调参数] 确认接触使用的合力阈值 N。
                "contact_on": 15.0,
                # [调参数] 确认离地使用的合力阈值 N。
                "contact_off": 5.0,
                # [调参数] 接触/离地连续确认时间 s。
                "debounce_s": 0.04,
                # [调参数] 摆动核心检查窗口的起点，按摆动阶段自身进度归一化。
                "core_start": 0.1,
                # [调参数] 摆动核心检查窗口的终点。
                "core_end": 0.9,
                # [调参数] 重复起脚事件项的时间尺度；不是整项奖励 weight。
                "repeat_event_cost_s": 0.10,
            },
        )
        # 正向鼓励：摆动相位中，脚部接触力接近零。
        # [调参数] 官方步态奖励内部“摆动脚卸载”子项系数，不是 gait_reward 的总权重。
        self.rewards.gait_reward.params[
            "tracking_contacts_shaped_force"
        ] = 1.5

        # 保留当前迈步能力，减少机身与腰部过大运动。
        # [调权重] 身体水平姿态惩罚权重。 [最终赋值处] 本方法里该属性最后一次赋值；前面重复设置不会决定最终值。
        self.rewards.flat_orientation_l2.weight = -6.0
        # [调权重] 身体 roll/pitch 角速度惩罚权重。 [最终赋值处] 本方法里该属性最后一次赋值；前面重复设置不会决定最终值。
        self.rewards.ang_vel_xy_l2.weight = -0.6


        # 单独加强腰部偏离默认角度的惩罚。
        # [调参数] 腰部关节在姿态偏离奖励内部的单关节系数；仍会乘该项的总 weight。
        self.rewards.joint_deviation_l1.params["deviation_weight"][
            "waist_yaw_joint"
        ] = 0.40

        # 温和约束策略输出的幅度。
        self.rewards.action_magnitude_l2 = RewTerm(
            func=action_l2,
            # [调权重] 策略动作幅度惩罚权重；不是动作缩放 Action Scale。 不必新增一段 .weight 赋值，直接改下面这个数。
            weight=-0.0005,
        )
        # 温和约束踝侧摆目标越界，不改变动作接口。
        self.rewards.ankle_roll_target_limits = RewTerm(
            func=ankle_roll_target_limit_error,
            # [调权重] 踝侧摆目标越界惩罚权重；注意后面 control/stance_yaw 模式会关闭此项。 不必新增一段 .weight 赋值，直接改下面这个数。
            weight=-0.02,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_ankle_roll_joint",
                        "right_ankle_roll_joint",
                    ],
                    preserve_order=True,
                ),
            },
        )
        # 默认 off：不改变原有配置。
        # control：对照组；stance_yaw：支撑脚旋转实验组。
        # [模式覆盖] 默认 off；此处决定下面哪些覆盖语句会执行。
        # 若 control/stance_yaw，把前面已创建的踝目标惩罚改为 None，前面的 weight 因此不再参与计算。
        # 若 stance_yaw，还会替换原 feet_ang_slide_penalty；其他模式使用继承的旧实现。
        # 需要改旧旋转惩罚时，建议在本方法对应模式路径中显式覆盖，避免直接改通用基类影响其他机器人。
        trial = os.environ.get("ROBOCUP_GAIT_TRIAL", "off")

        if trial not in ("off", "control", "stance_yaw"):
            raise ValueError(
                f"未知的 ROBOCUP_GAIT_TRIAL: {trial}"
            )

        if trial in ("control", "stance_yaw"):
            # 两组共同关闭该项，避免同时改变多个优化目标。
            # [开关] 这里不是调大小，而是彻底关闭该项；关闭后改前面的 weight 不会生效。
            self.rewards.ankle_roll_target_limits = None

        if trial == "stance_yaw":
            # 替换原来的旋转惩罚，不重复叠加。
            self.rewards.feet_ang_slide_penalty = RewTerm(
                func=feet_contact_yaw_motion,
                # [调权重] 仅 stance_yaw 模式使用的新支撑脚旋转惩罚权重。 不必新增一段 .weight 赋值，直接改下面这个数。
                weight=-0.02,
                params={
                    "sensor_cfg": SceneEntityCfg(
                        "contact_forces",
                        body_names=FOOT_NAMES,
                        preserve_order=True,
                    ),
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=FOOT_NAMES,
                        preserve_order=True,
                    ),
                    # [调参数] 触地力阈值 N；按对应函数分别可能比较合力或竖直力，不是惩罚权重。
                    "force_threshold": 15.0,
                },
            )

@configclass
# [回放提醒] 此类先调用训练配置，再覆盖回放条件；本段没有单独重设奖励权重。
# 修改奖励只改变后续学习/评分，不会让已经保存的 Policy 在回放时自动学会新动作。
class RobotcupMiniVelocityEnvCfg_PLAY(RobotcupMiniVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1

        self.events.reset_robot_joints.params["position_range"] = (
            0.0,
            0.0,
        )

        # 回放固定执行前进指令，避免随机命令影响观察
        velocity = self.commands.base_velocity
        velocity.rel_standing_envs = 0.0

        velocity.ranges.lin_vel_x = (0.4,0.4)
        velocity.ranges.lin_vel_y = (0.0, 0.0)
        velocity.ranges.ang_vel_z = (-0.4, 0.4)
    
        velocity.debug_vis = True
