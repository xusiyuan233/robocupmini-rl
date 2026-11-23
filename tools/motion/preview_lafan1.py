"""预览 LAFAN1 人体行走片段。

只处理当前 LAFAN1 文件使用的通道布局。
输出人体骨架动画，不生成机器人关节动作，不启动训练。
"""

import argparse
import io
import json
from pathlib import Path
from typing import cast

import matplotlib

# 不打开绘图窗口，直接导出图片和动画网页。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "data/motions/robotcupmini/raw/lafan1/walk4_subject1.bvh"
)


def load_bvh(path):
    """读取骨架，并通过正向运动学计算世界坐标中的关节位置。"""

    hierarchy, motion = path.read_text().split("MOTION", 1)

    tokens = iter(
        hierarchy.replace("{", " { ").replace("}", " } ").split()
    )

    names = []
    parents = []
    offsets = []
    channels = []

    def expect(expected):
        actual = next(tokens)
        if actual != expected:
            raise ValueError(
                f"骨架格式错误：预期 {expected}，实际 {actual}"
            )

    def read_joint(parent):
        index = len(names)

        names.append(next(tokens))
        parents.append(parent)
        offsets.append(None)
        channels.append(None)

        expect("{")

        while True:
            tag = next(tokens)

            if tag == "}":
                break

            if tag == "OFFSET":
                offsets[index] = [
                    float(next(tokens)) for _ in range(3)
                ]

            elif tag == "CHANNELS":
                count = int(next(tokens))
                channels[index] = [
                    next(tokens) for _ in range(count)
                ]

            elif tag == "JOINT":
                read_joint(index)

            elif tag == "End":
                # End Site 不作为可旋转关节加入骨架。
                expect("Site")
                expect("{")
                expect("OFFSET")
                for _ in range(3):
                    float(next(tokens))
                expect("}")

            else:
                raise ValueError(f"未支持的骨架字段：{tag}")

    expect("HIERARCHY")
    expect("ROOT")
    read_joint(-1)

    if next(tokens, None) is not None:
        raise ValueError("骨架段存在未解析内容")

    lines = motion.strip().splitlines()

    frame_count = int(lines[0].split(":", 1)[1])
    dt = float(lines[1].split(":", 1)[1])

    if frame_count <= 0 or dt <= 0:
        raise ValueError("帧数或帧间隔无效")

    data = np.loadtxt(
        io.StringIO("\n".join(lines[2:])),
        ndmin=2,
    )

    rotation_channels = [
        "Zrotation",
        "Yrotation",
        "Xrotation",
    ]

    root_channels = [
        "Xposition",
        "Yposition",
        "Zposition",
    ] + rotation_channels

    if channels[0] != root_channels:
        raise ValueError("根部通道布局与当前脚本不一致")

    if any(c != rotation_channels for c in channels[1:]):
        raise ValueError("关节旋转通道布局与当前脚本不一致")

    joint_count = len(names)

    if data.shape != (frame_count, 3 + joint_count * 3):
        raise ValueError(f"运动数据形状不正确：{data.shape}")

    if not np.isfinite(data).all():
        raise ValueError("运动数据包含 NaN 或 Inf")

    # BVH 旋转角以度表示，按照文件声明的 Z、Y、X 顺序组合。
    local_rotations = Rotation.from_euler(
        "ZYX",
        data[:, 3:].reshape(-1, 3),
        degrees=True,
    ).as_matrix().reshape(frame_count, joint_count, 3, 3)

    # 当前 LAFAN1 文件的长度数值由厘米转换为米。
    local_positions = np.broadcast_to(
        np.asarray(offsets) * 0.01,
        (frame_count, joint_count, 3),
    ).copy()

    # 根部位置通道给出根部平移，不重复加根部 OFFSET。
    local_positions[:, 0] = data[:, :3] * 0.01

    world_positions = np.empty_like(local_positions)
    world_rotations = np.empty_like(local_rotations)

    for joint, parent in enumerate(parents):
        if parent < 0:
            world_positions[:, joint] = local_positions[:, joint]
            world_rotations[:, joint] = local_rotations[:, joint]
        else:
            world_positions[:, joint] = (
                world_positions[:, parent]
                + np.einsum(
                    "tij,tj->ti",
                    world_rotations[:, parent],
                    local_positions[:, joint],
                )
            )

            world_rotations[:, joint] = (
                world_rotations[:, parent]
                @ local_rotations[:, joint]
            )

    return names, np.asarray(parents), world_positions, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--start", type=float, default=24.0)
    parser.add_argument("--end", type=float, default=28.0)
    args = parser.parse_args()

    names, parents, positions, dt = load_bvh(args.file)

    if not (
        0 <= args.start < args.end <= len(positions) * dt
    ):
        raise ValueError("起止时间不合法，或超过文件长度")

    all_times = np.arange(len(positions)) * dt
    selected = np.flatnonzero(
        (all_times >= args.start)
        & (all_times < args.end)
    )

    if len(selected) < 2:
        raise ValueError("选择的片段太短")

    # 原始 Y 向上，转换为 Z 向上：
    # 新坐标 = (原 X，-原 Z，原 Y)。
    xyz = positions[selected][..., [0, 2, 1]].copy()
    xyz[..., 1] *= -1

    # 只去除初始水平位置，不改变高度。
    initial_xy = xyz[0, 0, :2].copy()
    xyz[..., :2] -= initial_xy

    # 将片段整体前进方向对齐到图中的 +X。
    # 这只是统一观察方向，不会把弯曲轨迹强行拉直。
    displacement = xyz[-1, 0, :2] - xyz[0, 0, :2]

    if np.linalg.norm(displacement) < 0.1:
        raise ValueError("净位移太小，不适合作为直线前进片段")

    yaw = np.arctan2(displacement[1], displacement[0])
    c, s = np.cos(yaw), np.sin(yaw)
    xyz[..., :2] = xyz[..., :2] @ np.array([
        [c, -s],
        [s, c],
    ])

    times = all_times[selected]
    steps = np.linalg.norm(
        np.diff(xyz[:, 0, :2], axis=0),
        axis=1,
    )
    path_length = float(steps.sum())

    mean_speed = path_length / (times[-1] - times[0])
    straightness = float(
        np.linalg.norm(xyz[-1, 0, :2]) / path_length
    )

    output = (
        PROJECT_ROOT
        / "outputs/style_reference_preview"
        / f"{args.file.stem}_{args.start:g}_{args.end:g}"
    )
    output.mkdir(parents=True, exist_ok=True)

    # 检查图：根部轨迹与脚部骨架点高度。
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(xyz[:, 0, 0], xyz[:, 0, 1])
    axes[0].set(
        title="Human root trajectory",
        xlabel="Forward X (m)",
        ylabel="Lateral Y (m)",
    )
    axes[0].axis("equal")
    axes[0].grid(True)

    for name in ["LeftFoot", "RightFoot", "LeftToe", "RightToe"]:
        joint = names.index(name)
        axes[1].plot(
            times - times[0],
            xyz[:, joint, 2],
            label=name,
        )

    axes[1].set(
        title="Human skeleton point heights",
        xlabel="Time within clip (s)",
        ylabel="Z (m), NOT robot sole clearance",
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    figure.tight_layout()
    figure.savefig(output / "overview.png", dpi=150)
    plt.close(figure)

    # 动画：蓝色左侧，红色右侧，灰色躯干。
    edges = [
        (parent, joint)
        for joint, parent in enumerate(parents)
        if parent >= 0
    ]

    colors = [
        "tab:blue" if names[joint].startswith("Left")
        else "tab:red" if names[joint].startswith("Right")
        else "gray"
        for _, joint in edges
    ]

    figure = plt.figure(figsize=(7, 6), dpi=80)
    # projection="3d" 实际创建 Axes3D，明确类型以免检查器按二维 Axes 报错。
    axis = cast(Axes3D, figure.add_subplot(111, projection="3d"))

    skeleton = Line3DCollection(
        xyz[0][np.asarray(edges)],
        colors=colors,
        linewidths=3,
    )
    axis.add_collection3d(skeleton)

    axis.set(
        xlabel="Forward X (m)",
        ylabel="Lateral Y (m)",
        zlabel="Height Z (m)",
        zlim=(-0.15, 1.9),
    )
    axis.set_box_aspect((1.8, 1.8, 2.05))
    axis.view_init(elev=15, azim=-65)

    def update(frame):
        # 转成嵌套列表以匹配 set_segments 的 Sequence 参数类型，坐标值不变。
        skeleton.set_segments(xyz[frame][np.asarray(edges)].tolist())

        # 摄像机跟随根部，防止人走出画面。
        x, y = xyz[frame, 0, :2]
        axis.set_xlim(x - 0.9, x + 0.9)
        axis.set_ylim(y - 0.9, y + 0.9)

        axis.set_title(
            f"Human reference | source time {times[frame]:.2f}s\n"
            "Blue: left | Red: right"
        )

        return (skeleton,)

    animation = FuncAnimation(
        figure,
        update,
        frames=len(xyz),
        interval=dt * 1000,
        blit=False,
        repeat=False,
    )

    matplotlib.rcParams["animation.embed_limit"] = 80

    # 让动画根据前面设置的 interval 自动计算帧率，避免 float/int 类型冲突。
    html = animation.to_jshtml(
        default_mode="once",
    )

    (output / "preview.html").write_text(html, encoding="utf-8")
    plt.close(figure)

    metadata = {
        "source_file": str(args.file.resolve()),
        "requested_start_s": args.start,
        "requested_end_s": args.end,
        "first_frame": int(selected[0]),
        "last_frame_inclusive": int(selected[-1]),
        "frames": len(selected),
        "fps": 1.0 / dt,
        "human_mean_speed_mps": mean_speed,
        "root_straightness": straightness,
        "status": "human_preview_only_not_robot_validated",
    }

    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"帧数：{len(selected)}")
    print(f"人体平均水平速度：{mean_speed:.3f} m/s")
    print(f"根部轨迹直线度：{straightness:.3f}")
    print(f"输出目录：{output}")
    print("仅完成人体参考预览，尚未生成机器人训练数据。")


if __name__ == "__main__":
    main()