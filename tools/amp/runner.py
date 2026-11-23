"""RobotcupMini 的第一版 AMP 训练入口。

特点：
- 保留原来的 PPO 实现；
- 风格奖励在写入 PPO storage 前加入；
- 每轮用当前策略采样更新判别器；
- 不连接跨回合特征；
- 保存和恢复判别器及其优化器。

本版本暂不加入历史样本回放池。
"""

import hashlib
import math
import os
import time
from typing import cast

import torch
from torch.utils.tensorboard import SummaryWriter

from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runner import OnPolicyRunner

from tools.amp.discriminator import AmpDiscriminator
from tools.amp.isaac_features import IsaacAmpFeatures
from tools.amp.motion_loader import load_expert_features
from tools.amp.transition_buffer import AmpTransitionBuffer


class AmpOnPolicyRunner(OnPolicyRunner):
    """在原有 PPO 外增加风格奖励与判别器更新。"""

    def __init__(
        self,
        env,
        train_cfg,
        log_dir=None,
        device="cpu",
        style_weight=1.0,
    ):
        super().__init__(
            env,
            train_cfg,
            log_dir=log_dir,
            device=device,
        )

        self.style_weight = float(style_weight)

        if (
            not math.isfinite(self.style_weight)
            or self.style_weight < 0.0
        ):
            raise ValueError("风格奖励权重必须是有限的非负数")

        # 获取包装器内部的原始环境。
        raw_env = getattr(self.env, "unwrapped", None)

        # 运行时检查：必须存在，并且是我们支持的环境类型。
        if raw_env is None:
            raise TypeError("当前环境包装器没有提供 unwrapped 环境")

        if not isinstance(raw_env, ManagerBasedRLEnv):
            raise TypeError("当前 AMP Runner 只支持 ManagerBasedRLEnv")

        # 类型声明：告诉编辑器，经过检查后这是 ManagerBasedRLEnv。
        base_env = cast(ManagerBasedRLEnv, raw_env)

        # 读取环境控制周期，单位为秒。
        self.step_dt = float(base_env.step_dt)

        # 当前示范为50 Hz，相邻特征对的时间间隔必须一致。
        if not math.isclose(
            self.step_dt,
            0.02,
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            raise ValueError(
                f"环境控制周期为 {self.step_dt} s，"
                "与当前示范的0.02 s不一致"
            )

        self.feature_extractor = IsaacAmpFeatures(base_env)

        features_np, transitions_np = load_expert_features()

        # 防止恢复训练时无意中换用了另一份示范。
        self.expert_hash = hashlib.sha256(
            features_np.tobytes()
        ).hexdigest()

        expert_features = torch.as_tensor(
            features_np,
            dtype=torch.float32,
            device=self.device,
        )

        self.expert_transitions = torch.as_tensor(
            transitions_np,
            dtype=torch.float32,
            device=self.device,
        )

        self.discriminator = AmpDiscriminator(
            expert_features
        ).to(self.device)

        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=2.0e-5,
        )

        # 第一版采用固定的小规模判别器更新。
        self.discriminator_updates = 1
        self.discriminator_batch_size = 256

        print(
            "[AMP] 已启用："
            f"专家样本={len(self.expert_transitions)}，"
            "特征对维度=84，"
            f"控制周期={self.step_dt:.3f}s，"
            f"风格权重={self.style_weight}"
        )

    def _read_observations(self, obs_dict):
        """保持原项目的策略、历史、指令与critic观测接口。"""
        return (
            obs_dict["policy"].to(self.device),
            obs_dict["obsHistory"].flatten(
                start_dim=1
            ).to(self.device),
            obs_dict["commands"].to(self.device),
            obs_dict["critic"].to(self.device),
        )

    def _update_discriminator(self, policy_batches):
        """使用本轮采集的有效特征对更新判别器。"""
        if not policy_batches:
            raise RuntimeError(
                "本轮没有有效的AMP样本，请检查环境是否持续重置"
            )

        policy_data = torch.cat(policy_batches, dim=0).detach()

        if not torch.isfinite(policy_data).all().item():
            raise RuntimeError("机器人AMP特征中出现NaN或Inf")

        totals = {}
        self.discriminator.train()

        for _ in range(self.discriminator_updates):
            expert_ids = torch.randint(
                len(self.expert_transitions),
                (self.discriminator_batch_size,),
                device=self.device,
            )

            policy_ids = torch.randint(
                len(policy_data),
                (self.discriminator_batch_size,),
                device=self.device,
            )

            expert_batch = self.expert_transitions[expert_ids]
            policy_batch = policy_data[policy_ids]

            self.discriminator_optimizer.zero_grad(
                set_to_none=True
            )

            loss, metrics = self.discriminator.compute_loss(
                expert_batch,
                policy_batch,
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError("判别器loss出现NaN或Inf")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                max_norm=1.0,
                error_if_nonfinite=True,
            )

            self.discriminator_optimizer.step()

            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value.item()

        return {
            key: value / self.discriminator_updates
            for key, value in totals.items()
        }

    def learn(
        self,
        num_learning_iterations,
        init_at_random_ep_len=False,
    ):
        if self.log_dir is None:
            raise ValueError("AMP训练需要指定日志目录")

        if self.cfg.get("logger", "tensorboard").lower() != "tensorboard":
            raise ValueError("第一版AMP Runner请使用tensorboard日志")

        os.makedirs(self.log_dir, exist_ok=True)

        writer = SummaryWriter(
            log_dir=self.log_dir,
            flush_secs=10,
        )
        self.writer = writer

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        observations = self.env.get_observations()

        obs, history, commands, critic_obs = (
            self._read_observations(observations)
        )

        transition_buffer = AmpTransitionBuffer(
            self.feature_extractor.get().to(self.device)
        )

        self.alg.actor_critic.train()
        self.alg.encoder.train()

        first_iteration = self.current_learning_iteration
        final_iteration = (
            first_iteration + num_learning_iterations
        )

        try:
            for it in range(first_iteration, final_iteration):
                start = time.time()
                policy_batches = []

                # 依次记录任务奖励、原始风格奖励、
                # 加权风格奖励、有效样本比例。
                reward_stats = torch.zeros(
                    4,
                    device=self.device,
                )
                reset_count = 0

                # 这里使用no_grad，采样不构建梯度图；
                # 判别器更新放在这个上下文之外。
                with torch.no_grad():
                    self.discriminator.eval()

                    for _ in range(self.num_steps_per_env):
                        actions = self.alg.act(
                            obs,
                            history,
                            commands,
                            critic_obs,
                        )

                        observations, task_reward, dones, infos = (
                            self.env.step(actions)
                        )

                        task_reward = task_reward.to(self.device)
                        dones = dones.to(self.device)

                        current_features = (
                            self.feature_extractor.get().to(self.device)
                        )

                        pairs, valid_mask = transition_buffer.step(
                            current_features,
                            dones,
                        )

                        style_reward = torch.zeros_like(task_reward)

                        if pairs.shape[0] > 0:
                            style_reward[valid_mask] = (
                                self.discriminator.style_reward(pairs)
                            )
                            policy_batches.append(pairs)

                        # 与当前IsaacLab任务奖励的dt缩放口径对齐。
                        style_bonus = (
                            self.style_weight
                            * self.step_dt
                            * style_reward
                        )

                        total_reward = task_reward + style_bonus

                        if not torch.isfinite(total_reward).all().item():
                            raise RuntimeError(
                                "合并后的训练奖励出现NaN或Inf"
                            )

                        obs, history, commands, critic_obs = (
                            self._read_observations(observations)
                        )

                        # PPO接收到的是合并后的奖励。
                        self.alg.process_env_step(
                            total_reward,
                            dones,
                            infos,
                            obs,
                        )

                        reward_stats += torch.stack(
                            [
                                task_reward.mean(),
                                style_reward.mean(),
                                style_bonus.mean(),
                                valid_mask.float().mean(),
                            ]
                        )

                        reset_count += int(
                            (dones > 0).sum().item()
                        )

                    last_critic_obs = torch.cat(
                        (critic_obs, commands),
                        dim=-1,
                    )

                    if self.alg.critic_take_latent:
                        latent = self.alg.encoder.encode(history)
                        last_critic_obs = torch.cat(
                            (last_critic_obs, latent),
                            dim=-1,
                        )

                    self.alg.compute_returns(last_critic_obs)

                # 先更新机器人策略，再更新判别器。
                ppo_losses = self.alg.update()

                amp_metrics = self._update_discriminator(
                    policy_batches
                )

                averages = (
                    reward_stats / self.num_steps_per_env
                ).cpu().tolist()

                self.current_learning_iteration = it + 1

                for key, value in amp_metrics.items():
                    writer.add_scalar(
                        f"AMP/{key}",
                        value,
                        self.current_learning_iteration,
                    )

                for name, value in zip(
                    (
                        "task_per_step",
                        "style_raw",
                        "style_bonus_per_step",
                        "valid_fraction",
                    ),
                    averages,
                ):
                    writer.add_scalar(
                        f"AMP/{name}",
                        value,
                        self.current_learning_iteration,
                    )

                for name, value in zip(
                    ("value", "encoder", "surrogate", "kl"),
                    ppo_losses,
                ):
                    writer.add_scalar(
                        f"Loss/{name}",
                        float(value),
                        self.current_learning_iteration,
                    )

                writer.add_scalar(
                    "AMP/reset_count",
                    reset_count,
                    self.current_learning_iteration,
                )

                print(
                    f"[AMP] iter={self.current_learning_iteration} "
                    f"task={averages[0]:.4f} "
                    f"style={averages[1]:.4f} "
                    f"bonus={averages[2]:.4f} "
                    f"D_loss={amp_metrics['loss']:.4f} "
                    f"expert={amp_metrics['expert_score']:.3f} "
                    f"policy={amp_metrics['policy_score']:.3f} "
                    f"valid={averages[3]:.3f} "
                    f"resets={reset_count} "
                    f"time={time.time() - start:.2f}s"
                )

                if (
                    (self.current_learning_iteration - first_iteration)
                    % self.save_interval == 0
                ):
                    self.save(
                        os.path.join(
                            self.log_dir,
                            f"model_{self.current_learning_iteration}.pt",
                        )
                    )

            self.save(
                os.path.join(
                    self.log_dir,
                    f"model_{self.current_learning_iteration}.pt",
                )
            )

        finally:
            writer.flush()
            writer.close()

    def save(self, path, infos=None):
        """保留原策略检查点字段，并附加AMP训练状态。"""
        torch.save(
            {
                "model_state_dict": (
                    self.alg.actor_critic.state_dict()
                ),
                "encoder_state_dict": (
                    self.alg.encoder.state_dict()
                ),
                "optimizer_state_dict": (
                    self.alg.optimizer.state_dict()
                ),
                "iter": self.current_learning_iteration,
                "infos": infos,
                "amp": {
                    "version": 1,
                    "model": self.discriminator.state_dict(),
                    "optimizer": (
                        self.discriminator_optimizer.state_dict()
                    ),
                    "style_weight": self.style_weight,
                    "step_dt": self.step_dt,
                    "expert_hash": self.expert_hash,
                },
            },
            path,
        )
        print(f"[AMP] 检查点已保存：{path}")

    def load(self, path, load_optimizer=True):
        """区分AMP恢复训练和普通PPO策略热启动。"""
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )

        amp_state = checkpoint.get("amp")

        if amp_state is not None:
            if amp_state["version"] != 1:
                raise ValueError("AMP检查点版本不兼容")

            for key, expected in (
                ("expert_hash", self.expert_hash),
                ("style_weight", self.style_weight),
                ("step_dt", self.step_dt),
            ):
                if amp_state[key] != expected:
                    raise ValueError(
                        f"AMP恢复失败：{key}与当前配置不一致"
                    )

        # 不允许静默跳过不匹配的策略参数。
        self.alg.actor_critic.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        self.alg.encoder.load_state_dict(
            checkpoint["encoder_state_dict"],
            strict=True,
        )

        if amp_state is None:
            # 普通PPO检查点：沿用已有策略，
            # 但新建判别器，AMP轮数从0开始计数。
            self.current_learning_iteration = 0
            print(
                "[AMP] 已加载普通PPO策略作为热启动；"
                "判别器新建，AMP迭代数从0开始。"
            )
        else:
            self.discriminator.load_state_dict(
                amp_state["model"],
                strict=True,
            )

            if load_optimizer:
                self.alg.optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"]
                )
                self.discriminator_optimizer.load_state_dict(
                    amp_state["optimizer"]
                )

            self.current_learning_iteration = int(
                checkpoint["iter"]
            )

            print(
                "[AMP] 已恢复策略、encoder和判别器；"
                f"当前迭代数={self.current_learning_iteration}。"
            )

        return checkpoint.get("infos")