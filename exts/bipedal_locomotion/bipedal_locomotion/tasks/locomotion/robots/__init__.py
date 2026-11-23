import gymnasium as gym

from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg import (
    SF_TRON2AFlatPPORunnerCfg, WF_TRON2AFlatPPORunnerCfg,
)

from . import limx_solefoot_tron2a_env_cfg, limx_wheelfoot_tron2a_env_cfg

##
# Create PPO runners for RSL-RL
##

limx_sf_tron2a_blind_flat_runner_cfg = SF_TRON2AFlatPPORunnerCfg()

limx_wf_tron2a_blind_flat_runner_cfg = WF_TRON2AFlatPPORunnerCfg()


##
# Register Gym environments
##


######################################
# SF_TRON2A Blind Flat Environment
######################################
gym.register(
    id="Isaac-Limx-SF-TRON2A-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-TRON2A-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_blind_flat_runner_cfg,
    },
)


######################################
# WF_TRON2A Blind Flat Environment
######################################
gym.register(
    id="Isaac-Limx-WF-TRON2A-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_tron2a_env_cfg.WF_TRON2A_BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_tron2a_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-TRON2A-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_tron2a_env_cfg.WF_TRON2A_BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_tron2a_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-DASF-TRON2A-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.robots."
            "limx_dasf_tron2a_env_cfg:DASF_TRON2A_BlindFlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.agents."
            "dasf_rsl_rl_ppo_cfg:DASF_TRON2AFlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Limx-DASF-TRON2A-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.robots."
            "limx_dasf_tron2a_env_cfg:DASF_TRON2A_BlindFlatEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.agents."
            "dasf_rsl_rl_ppo_cfg:DASF_TRON2AFlatPPORunnerCfg"
        ),
    },
)
# Robotcupmini standing tasks
gym.register(
    id="Robocup-Stand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.robots."
            "robotcupmini_env_cfg:RobotcupMiniStandEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.agents."
            "robotcupmini_ppo_cfg:RobotcupMiniStandPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Robocup-Stand-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.robots."
            "robotcupmini_env_cfg:RobotcupMiniStandEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.agents."
            "robotcupmini_ppo_cfg:RobotcupMiniStandPPORunnerCfg"
        ),
    },
)
gym.register(
    id="Robocup-Vel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.robots."
            "robotcupmini_velocity_env_cfg:RobotcupMiniVelocityEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.agents."
            "robotcupmini_ppo_cfg:RobotcupMiniVelocityPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Robocup-Vel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.robots."
            "robotcupmini_velocity_env_cfg:RobotcupMiniVelocityEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.agents."
            "robotcupmini_ppo_cfg:RobotcupMiniVelocityPPORunnerCfg"
        ),
    },
)



# RobotcupMini 风格学习准备任务。
# 当前尚未加入 AMP 判别器和风格奖励。
gym.register(
    id="Robocup-Style-Prep-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.robots."
            "robotcupmini_style_env_cfg:RobotcupMiniStylePrepEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.agents."
            "robotcupmini_ppo_cfg:RobotcupMiniStylePrepPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Robocup-Style-Prep-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.robots."
            "robotcupmini_style_env_cfg:RobotcupMiniStylePrepEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.agents."
            "robotcupmini_ppo_cfg:RobotcupMiniStylePrepPPORunnerCfg"
        ),
    },
)