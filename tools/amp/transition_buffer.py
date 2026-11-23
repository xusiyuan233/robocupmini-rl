"""收集 AMP 连续帧特征，排除跨回合的样本。"""

import torch


class AmpTransitionBuffer:
    """缓存每个并行环境的上一帧42维特征。

    使用约定：
    1. 初始化时传入环境重置后的特征。
    2. 每次 env.step() 后，传入最新特征和 dones。
    3. 已结束的环境应已自动重置，最新特征属于新回合。
    """

    def __init__(self, initial_features: torch.Tensor):
        if (
            initial_features.ndim != 2
            or initial_features.shape[1] != 42
        ):
            raise ValueError("初始特征必须是 [环境数, 42]")

        if not initial_features.is_floating_point():
            raise TypeError("特征必须是浮点 Tensor")

        # 必须复制，不能让仿真后续更新覆盖缓存。
        self.previous_features = initial_features.detach().clone()

    @torch.no_grad()
    def step(
        self,
        current_features: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回有效特征对，以及它们对应的环境掩码。

        返回：
            valid_transitions: [有效环境数, 84]
            valid_mask: [环境数]，True 表示该环境样本有效。
        """
        if current_features.shape != self.previous_features.shape:
            raise ValueError("当前特征形状与初始特征不一致")

        if (
            current_features.device != self.previous_features.device
            or current_features.dtype != self.previous_features.dtype
        ):
            raise ValueError("当前特征的设备或数据类型发生变化")

        num_envs = current_features.shape[0]

        if dones.shape != (num_envs,):
            raise ValueError("dones 必须是 [环境数]")

        # dones=True：刚结束或超时的环境，排除该步样本。
        valid_mask = ~dones.to(
            device=current_features.device,
            dtype=torch.bool,
        )

        # 顺序必须与专家样本一致：前一帧在前，后一帧在后。
        transitions = torch.cat(
            [
                self.previous_features,
                current_features.detach(),
            ],
            dim=-1,
        )

        valid_transitions = transitions[valid_mask]

        # 即使发生重置，也更新缓存。
        # 重置环境保存的是新回合初始帧，供下一步使用。
        self.previous_features = current_features.detach().clone()

        return valid_transitions, valid_mask