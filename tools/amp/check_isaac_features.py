"""检查专家特征与 IsaacLab 特征是否一致。

只设置少量参考状态并读取数据：
不运行策略，不进行训练，不证明动力学稳定性。
"""

import argparse
import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rsl_rl"))

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(
        description="检查 AMP 专家与在线特征的一致性"
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    launcher = AppLauncher(args)
    app = launcher.app
    env = None

    try:
        import gymnasium as gym
        import numpy as np
        import torch

        # 导入后注册项目中的任务。
        import bipedal_locomotion  # noqa: F401

        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_tasks.utils import parse_env_cfg

        from tools.amp.isaac_features import IsaacAmpFeatures
        from tools.amp.motion_loader import (
            JOINT_NAMES,
            MANIFEST_PATH,
            load_expert_features,
        )
        from tools.motion.verified_style_data import load_verified_motion

        cfg = parse_env_cfg(
            "Robocup-Style-Prep-v0",
            device=args.device,
            num_envs=1,
        )
        cfg.seed = 42

        env = gym.make("Robocup-Style-Prep-v0", cfg=cfg)
        env.reset()

        base = cast(ManagerBasedRLEnv, env.unwrapped)
        extractor = IsaacAmpFeatures(base)
        robot = extractor.robot

        verified = load_verified_motion(
            MANIFEST_PATH,
            expected_joint_names=list(JOINT_NAMES),
        )
        arrays = verified.arrays
        expert_features, _ = load_expert_features()

        def tensor(value):
            return torch.as_tensor(
                value,
                dtype=torch.float32,
                device=base.device,
            )

        max_error = np.zeros(42)

        # 覆盖示范开头、中间和结尾，共检查7帧。
        frames = np.linspace(
            0,
            len(expert_features) - 1,
            7,
            dtype=int,
        )

        for frame in frames:
            state = robot.data.default_root_state.clone()

            state[0, :3] = (
                tensor(arrays["root_positions_m"][frame])
                + base.scene.env_origins[0]
            )

            # 示范是 xyzw，IsaacLab 接口需要 wxyz。
            state[0, 3:7] = tensor(
                arrays["root_quaternions_xyzw"][
                    frame, [3, 0, 1, 2]
                ]
            )

            state[0, 7:10] = tensor(
                arrays["root_linear_velocities_world_m_s"][frame]
            )
            state[0, 10:13] = tensor(
                arrays["root_angular_velocities_world_rad_s"][frame]
            )

            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = torch.zeros_like(joint_pos)

            joint_pos[:, extractor.joint_ids] = tensor(
                arrays["joint_positions_rad"][frame]
            )
            joint_vel[:, extractor.joint_ids] = tensor(
                arrays["joint_velocities_rad_s"][frame]
            )

            robot.write_root_link_state_to_sim(state)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # 更新运动学和数据缓存，不推进物理时间。
            base.sim.forward()
            base.scene.update(base.physics_dt)

            actual = extractor.get()[0].detach().cpu().numpy()

            if actual.shape != (42,) or not np.isfinite(actual).all():
                raise RuntimeError("在线特征维度错误或存在 NaN/Inf")

            error = np.abs(actual - expert_features[frame])
            max_error = np.maximum(max_error, error)

        blocks = [
            ("关节角度", 0, 13),
            ("关节速度", 13, 26),
            ("根部线速度", 26, 29),
            ("根部角速度", 29, 32),
            ("重力方向", 32, 35),
            ("脚底相对位置", 35, 41),
            ("根部高度", 41, 42),
        ]

        print("\n========== 特征一致性检查 ==========")
        for name, start, end in blocks:
            print(f"{name}最大误差：{max_error[start:end].max():.8f}")

        # 脚底位置容许2毫米的运动学误差。
        # 其他特征使用各自单位下的数值误差门槛。
        tolerance = np.full(42, 2.0e-4)
        tolerance[35:41] = 2.0e-3

        if np.any(max_error > tolerance):
            raise RuntimeError(
                "特征一致性检查未通过，请保留上面的分项误差。"
            )

        print("在线特征形状：", tuple(extractor.get().shape))
        print("通过：专家与 IsaacLab 在线特征一致。")
        print("本次未启动训练。")

    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()