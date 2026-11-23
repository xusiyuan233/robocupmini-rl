import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from bipedal_locomotion.actuators import DelayedImplicitActuatorCfg


current_dir = os.path.dirname(os.path.abspath(__file__))

usd_path = os.path.normpath(
    os.path.join(
        current_dir,
        "../usd/ROBOCUPMINI/usd/robotcupmini.usd",
    )
)

# 后面的动作和关节观测配置统一使用这个顺序。
JOINT_NAMES = [
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
]

ROBOCUPMINI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
        activate_contact_sensors=True,
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        # 当前默认姿态的初始根高度，后面核对足底接地情况。
        pos=(0.0, 0.0, 0.571),
        joint_pos={
            "left_hip_pitch_joint": -0.075,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.15,
            "left_ankle_pitch_joint": -0.075,
            "left_ankle_roll_joint": 0.0,

            "right_hip_pitch_joint": -0.075,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.15,
            "right_ankle_pitch_joint": -0.075,
            "right_ankle_roll_joint": 0.0,

            "waist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),

    soft_joint_pos_limit_factor=0.9,

    actuators={
        "hips_and_knees": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_.*_joint",
                ".*_knee_joint",
            ],
            stiffness=60.0, #kp
            damping=3.0, #kd
            effort_limit_sim=60.0,
            velocity_limit_sim=5.0,
            min_delay=0,
            max_delay=0,
        ),

        "ankles": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_.*_joint",
            ],
            stiffness=30.0,
            damping=1.5,
            effort_limit_sim=36.0,
            velocity_limit_sim=5.0,
            min_delay=0,
            max_delay=0,
        ),

        "waist": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
            ],
            stiffness=40.0,
            damping=2.0,
            effort_limit_sim=60.0,
            velocity_limit_sim=5.0,
            min_delay=0,
            max_delay=0,
        ),
    },
)
