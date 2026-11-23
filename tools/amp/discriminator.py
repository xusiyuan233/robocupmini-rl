"""AMP 风格判别器。

输入：连续两帧42维特征拼接而成的84维向量。
输出：一个实数评分，不是概率。
"""

import torch
from torch import nn


class AmpDiscriminator(nn.Module):
    """独立于机器人策略网络的风格判别器。"""

    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    
    def __init__(self, expert_features: torch.Tensor):
        super().__init__()

        if (
            expert_features.ndim != 2
            or expert_features.shape[1] != 42
            or expert_features.shape[0] < 2
        ):
            raise ValueError("专家特征必须是 [帧数, 42]")

        features = expert_features.detach().float()

        if not torch.isfinite(features).all().item():
            raise ValueError("专家特征中存在 NaN 或 Inf")

        # 第一版采用固定的专家数据归一化统计。
        # 设置标准差下限，避免近乎不变的特征被过度放大。
        mean = features.mean(dim=0)
        std = features.std(dim=0, unbiased=False).clamp_min(0.05)

        # 两帧使用同一套统计，拼接后为84维。
        # register_buffer 使统计量跟随模型保存和迁移设备。
        self.register_buffer("feature_mean", mean.repeat(2))
        self.register_buffer("feature_std", std.repeat(2))

        self.network = nn.Sequential(
            nn.Linear(84, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def normalize(self, transitions: torch.Tensor) -> torch.Tensor:
        """输入必须是尚未归一化的84维特征对。"""
        if transitions.ndim != 2 or transitions.shape[1] != 84:
            raise ValueError("判别器输入必须是 [样本数, 84]")

        normalized = (
            transitions - self.feature_mean
        ) / self.feature_std

        return normalized.clamp(-10.0, 10.0)

    def forward(self, transitions: torch.Tensor) -> torch.Tensor:
        """返回每个样本的评分，形状为 [样本数]。"""
        normalized = self.normalize(transitions)
        return self.network(normalized).squeeze(-1)

    @staticmethod
    def scores_to_reward(scores: torch.Tensor) -> torch.Tensor:
        """把评分转换为 [0, 1] 范围内的风格奖励。"""
        return (
            1.0 - 0.25 * (scores - 1.0).square()
        ).clamp(0.0, 1.0)

    @torch.no_grad()
    def style_reward(
        self,
        transitions: torch.Tensor,
    ) -> torch.Tensor:
        """计算奖励时，不对判别器或策略进行反向传播。"""
        scores = self(transitions)
        return self.scores_to_reward(scores)

    def compute_loss(
        self,
        expert_transitions: torch.Tensor,
        policy_transitions: torch.Tensor,
        gradient_penalty_weight: float = 10.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算判别器损失。

        特征输入被 detach，梯度仅用于更新判别器参数。
        """
        expert_normalized = (
            self.normalize(expert_transitions.detach())
            .detach()
            .requires_grad_(True)
        )

        policy_normalized = self.normalize(
            policy_transitions.detach()
        )

        expert_scores = self.network(
            expert_normalized
        ).squeeze(-1)

        policy_scores = self.network(
            policy_normalized
        ).squeeze(-1)

        # 专家目标 +1，策略样本目标 -1。
        expert_loss = (expert_scores - 1.0).square().mean()
        policy_loss = (policy_scores + 1.0).square().mean()

        # 对专家归一化特征求梯度。
        gradients = torch.autograd.grad(
            outputs=expert_scores.sum(),
            inputs=expert_normalized,
            create_graph=True,
        )[0]

        gradient_penalty = gradients.square().sum(
            dim=-1
        ).mean()

        loss = (
            expert_loss
            + policy_loss
            + 0.5 * gradient_penalty_weight * gradient_penalty
        )

        metrics = {
            "loss": loss.detach(),
            "expert_score": expert_scores.mean().detach(),
            "policy_score": policy_scores.mean().detach(),
            "gradient_penalty": gradient_penalty.detach(),
        }

        return loss, metrics