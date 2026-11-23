"""仅在 CPU 上检查连续帧缓存，不启动机器人训练。"""

import torch

from tools.amp.transition_buffer import AmpTransitionBuffer


def main() -> None:
    # 用3个虚拟环境检查，数字仅表示测试特征。
    initial = torch.zeros(3, 42)
    buffer = AmpTransitionBuffer(initial)

    # 修改原始数据，验证缓存持有的是独立副本。
    initial.fill_(999.0)

    # 第一步：环境1发生重置，其余环境正常。
    frame1 = torch.ones(3, 42)
    frame1[1] = 100.0

    pairs, mask = buffer.step(
        frame1,
        torch.tensor([False, True, False]),
    )

    assert pairs.shape == (2, 84)
    assert mask.tolist() == [True, False, True]

    torch.testing.assert_close(
        pairs[:, :42],
        torch.zeros(2, 42),
    )
    torch.testing.assert_close(
        pairs[:, 42:],
        torch.ones(2, 42),
    )

    print("通过：正常连续帧的维度与先后顺序检查。")
    print("通过：重置环境的跨回合样本排除检查。")

    # 保存预期值，再修改输入，检查缓存没有共享输入存储。
    expected_previous = frame1.clone()
    frame1.fill_(999.0)

    # 第二步：3个环境都正常推进。
    # 环境1应从新回合的100连接到101。
    frame2 = torch.full((3, 42), 2.0)
    frame2[1] = 101.0

    pairs, mask = buffer.step(
        frame2,
        torch.zeros(3, dtype=torch.bool),
    )

    assert pairs.shape == (3, 84)
    assert mask.all().item()

    torch.testing.assert_close(
        pairs[:, :42],
        expected_previous,
    )
    torch.testing.assert_close(
        pairs[:, 42:],
        frame2,
    )

    print("通过：重置后从新回合恢复连续帧。")
    print("通过：缓存独立保存，不受输入后续修改影响。")

    # 第三步：全部环境重置，没有有效样本也不应报错。
    pairs, mask = buffer.step(
        torch.zeros(3, 42),
        torch.ones(3, dtype=torch.bool),
    )

    assert pairs.shape == (0, 84)
    assert not mask.any().item()

    print("通过：全部环境重置时返回空样本。")
    print("本次仅检查连续帧缓存，未启动机器人训练。")


if __name__ == "__main__":
    main()