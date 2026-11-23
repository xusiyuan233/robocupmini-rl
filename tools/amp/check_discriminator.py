"""仅在 CPU 上检查判别器。

会进行32次小规模判别器参数更新。
不会启动 IsaacLab，不会更新机器人策略，不保存测试模型。
"""

import torch

from tools.amp.discriminator import AmpDiscriminator
from tools.amp.motion_loader import load_expert_features


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)

    features_np, transitions_np = load_expert_features()

    features = torch.from_numpy(features_np)
    transitions = torch.from_numpy(transitions_np)

    discriminator = AmpDiscriminator(features).to("cpu")

    optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=1.0e-4,
    )

    # 检查奖励转换公式。
    test_scores = torch.tensor([-1.0, 0.0, 1.0])
    expected_rewards = torch.tensor([0.0, 0.75, 1.0])

    torch.testing.assert_close(
        discriminator.scores_to_reward(test_scores),
        expected_rewards,
    )

    initial_parameter = next(
        discriminator.parameters()
    ).detach().clone()

    batch_size = 64

    for step in range(32):
        expert_ids = torch.randint(
            len(transitions),
            (batch_size,),
        )
        expert_batch = transitions[expert_ids]

        # 这里只用打乱时序的样本检查损失和梯度。
        # 正式训练必须换成当前策略产生的真实连续帧。
        first_ids = torch.randint(
            len(features),
            (batch_size,),
        )
        second_ids = torch.randint(
            len(features),
            (batch_size,),
        )

        test_negative_batch = torch.cat(
            [
                features[first_ids],
                features[second_ids],
            ],
            dim=-1,
        )

        optimizer.zero_grad(set_to_none=True)

        loss, metrics = discriminator.compute_loss(
            expert_batch,
            test_negative_batch,
        )

        if not torch.isfinite(loss).item():
            raise RuntimeError("判别器损失出现 NaN 或 Inf")

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            discriminator.parameters(),
            max_norm=1.0,
        )

        if not torch.isfinite(grad_norm).item():
            raise RuntimeError("判别器梯度出现 NaN 或 Inf")

        if step == 0 and grad_norm.item() == 0.0:
            raise RuntimeError("判别器没有获得非零梯度")

        optimizer.step()

        if step in (0, 15, 31):
            print(
                f"step={step + 1:02d} "
                f"loss={loss.item():.4f} "
                f"expert_score={metrics['expert_score'].item():.4f} "
                f"test_negative_score={metrics['policy_score'].item():.4f}"
            )

    final_parameter = next(
        discriminator.parameters()
    ).detach()

    if torch.equal(initial_parameter, final_parameter):
        raise RuntimeError("判别器参数没有更新")

    if not all(
        torch.isfinite(parameter).all().item()
        for parameter in discriminator.parameters()
    ):
        raise RuntimeError("判别器参数出现 NaN 或 Inf")

    discriminator.eval()

    sample = transitions[:16]
    rewards = discriminator.style_reward(sample)

    if rewards.shape != (16,):
        raise RuntimeError("风格奖励形状错误")

    if not torch.isfinite(rewards).all().item():
        raise RuntimeError("风格奖励出现 NaN 或 Inf")

    if not ((rewards >= 0.0) & (rewards <= 1.0)).all().item():
        raise RuntimeError("风格奖励超出 [0, 1]")

    if rewards.requires_grad:
        raise RuntimeError("计算风格奖励时不应保留梯度")

    # 模拟检查点恢复，包含归一化统计。
    restored = AmpDiscriminator(features)
    restored.load_state_dict(discriminator.state_dict())
    restored.eval()

    torch.testing.assert_close(
        restored.style_reward(sample),
        rewards,
    )

    print("\n通过：判别器损失与梯度检查。")
    print("通过：判别器参数更新检查。")
    print("通过：风格奖励范围与形状检查。")
    print("通过：模型参数及归一化统计恢复检查。")
    print("本次仅测试判别器，未训练机器人策略。")


if __name__ == "__main__":
    main()