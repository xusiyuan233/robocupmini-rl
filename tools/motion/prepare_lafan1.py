"""检查 LAFAN1 原始数据，并提取第一批行走候选动作。

本脚本不进行动作重定向，也不启动训练。
"""

from pathlib import Path
import io
import re
import zipfile

import numpy as np


# 根据脚本位置定位项目根目录，不依赖终端当前所在位置。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "motions"
    / "robotcupmini"
    / "raw"
    / "lafan1"
)

ZIP_PATH = RAW_DIR / "lafan1.zip"

CANDIDATES = (
    "walk4_subject1.bvh",
    "walk2_subject1.bvh",
    "walk1_subject1.bvh",
)


def check_bvh(name: str, content: bytes) -> None:
    """检查一个 BVH 文件的基础数据结构。"""

    text = content.decode("utf-8")

    if "MOTION" not in text:
        raise ValueError(f"{name} 缺少 MOTION 段")

    hierarchy, motion = text.split("MOTION", 1)

    # 每个关节的 CHANNELS 数相加，得到每帧应有的列数。
    channel_counts = re.findall(
        r"\bCHANNELS\s+(\d+)",
        hierarchy,
    )
    expected_columns = sum(int(x) for x in channel_counts)

    lines = motion.strip().splitlines()

    if len(lines) < 3:
        raise ValueError(f"{name} 的运动数据不完整")

    declared_frames = int(lines[0].split(":", 1)[1])
    frame_time = float(lines[1].split(":", 1)[1])

    if declared_frames <= 0 or frame_time <= 0:
        raise ValueError(f"{name} 的帧数或帧间隔无效")

    data = np.loadtxt(
        io.StringIO("\n".join(lines[2:])),
        ndmin=2,
    )

    if data.shape != (declared_frames, expected_columns):
        raise ValueError(
            f"{name} 数据形状错误："
            f"实际 {data.shape}，"
            f"预期 {(declared_frames, expected_columns)}"
        )

    if not np.isfinite(data).all():
        raise ValueError(f"{name} 含有 NaN 或 Inf")

    print(
        f"[OK] {name} | "
        f"frames={declared_frames} | "
        f"fps={1.0 / frame_time:.3f} | "
        f"columns={expected_columns}"
    )


def main() -> None:
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(
            f"没有找到压缩包，请先完成下载：\n{ZIP_PATH}"
        )

    if not zipfile.is_zipfile(ZIP_PATH):
        raise ValueError(
            "下载文件不是有效 ZIP。"
            "可能下载到了网页、Git LFS 指针或不完整文件。"
        )

    with zipfile.ZipFile(ZIP_PATH) as archive:
        print("正在校验压缩包，请稍等……")

        bad_file = archive.testzip()

        if bad_file is not None:
            raise ValueError(f"压缩包校验失败：{bad_file}")

        print("[OK] 压缩包 CRC 校验通过")

        archive_names = set(archive.namelist())

        # 先全部检查通过，再写出候选文件。
        selected = {}

        for name in CANDIDATES:
            if name not in archive_names:
                raise FileNotFoundError(
                    f"压缩包中没有找到候选动作：{name}"
                )

            content = archive.read(name)
            check_bvh(name, content)
            selected[name] = content

        for name, content in selected.items():
            destination = RAW_DIR / name

            # 不静默覆盖已有的不同内容。
            if destination.exists():
                if destination.read_bytes() != content:
                    raise FileExistsError(
                        f"已有同名但内容不同的文件，请先检查："
                        f"{destination}"
                    )
            else:
                destination.write_bytes(content)

    print(f"\n候选动作目录：{RAW_DIR}")
    print("原始数据准备完成；尚未重定向，尚不能用于 AMP 训练。")


if __name__ == "__main__":
    main()