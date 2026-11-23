"""基于 URDF 检查默认站姿和脚部碰撞网格。

不启动 IsaacLab，不运行物理仿真，不训练策略。
"""

from dataclasses import dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs/style_reference_fk"


@dataclass
class JointSpec:
    """保存单个关节的运动学信息。"""

    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


def origin_transform(element: ET.Element | None) -> np.ndarray:
    """把 URDF origin 转换成 4×4 变换矩阵。"""

    matrix = np.eye(4)

    if element is not None:
        xyz = np.fromstring(
            element.get("xyz", "0 0 0"), sep=" "
        )
        rpy = np.fromstring(
            element.get("rpy", "0 0 0"), sep=" "
        )

        if xyz.shape != (3,) or rpy.shape != (3,):
            raise ValueError(
                "URDF origin 必须包含三个 xyz 和三个 rpy"
            )

        matrix[:3, :3] = Rotation.from_euler(
            "xyz", rpy
        ).as_matrix()
        matrix[:3, 3] = xyz

    return matrix


def read_urdf(path: Path):
    """读取关节结构、根连杆和关节限位。"""

    xml = ET.parse(path).getroot()

    joints: list[JointSpec] = []
    limits: dict[str, tuple[float, float]] = {}

    links = {
        element.attrib["name"]
        for element in xml.findall("link")
    }

    for element in xml.findall("joint"):
        name = element.attrib["name"]
        kind = element.attrib["type"]

        if kind not in ("fixed", "revolute"):
            raise ValueError(
                f"当前工具不支持关节类型：{kind}"
            )

        if element.find("mimic") is not None:
            raise ValueError("当前工具不支持 mimic 关节")

        parent = element.find("parent")
        child = element.find("child")

        if parent is None or child is None:
            raise ValueError(
                f"关节缺少 parent/child：{name}"
            )

        axis_element = element.find("axis")
        axis = np.fromstring(
            axis_element.get("xyz", "1 0 0")
            if axis_element is not None else "1 0 0",
            sep=" ",
        )

        if (
            axis.shape != (3,)
            or np.linalg.norm(axis) < 1e-12
        ):
            raise ValueError(f"关节轴无效：{name}")

        axis = axis / np.linalg.norm(axis)

        if kind == "revolute":
            limit = element.find("limit")

            if limit is None:
                raise ValueError(
                    f"关节缺少限位：{name}"
                )

            limits[name] = (
                float(limit.attrib["lower"]),
                float(limit.attrib["upper"]),
            )

        joints.append(
            JointSpec(
                name=name,
                kind=kind,
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                origin=origin_transform(
                    element.find("origin")
                ),
                axis=axis,
            )
        )

    roots = links - {joint.child for joint in joints}

    if len(roots) != 1:
        raise ValueError(
            f"机器人必须只有一个根连杆：{roots}"
        )

    return xml, roots.pop(), joints, limits


def forward_kinematics(
    root: str,
    joints: list[JointSpec],
    angles: dict[str, float],
    base_height: float,
):
    """根据关节角，计算各连杆相对世界的变换矩阵。"""

    base = np.eye(4)
    base[2, 3] = base_height

    transforms = {root: base}
    pending = list(joints)

    while pending:
        remaining = []

        for joint in pending:
            if joint.parent not in transforms:
                remaining.append(joint)
                continue

            motion = np.eye(4)

            if joint.kind == "revolute":
                motion[:3, :3] = Rotation.from_rotvec(
                    joint.axis * angles[joint.name]
                ).as_matrix()

            transforms[joint.child] = (
                transforms[joint.parent]
                @ joint.origin
                @ motion
            )

        if len(remaining) == len(pending):
            raise ValueError(
                "URDF 存在断开的连杆或循环"
            )

        pending = remaining

    return transforms


def foot_vertices(
    xml: ET.Element,
    urdf_path: Path,
    link_name: str,
):
    """读取脚部碰撞网格，转换到脚部连杆坐标系。"""

    link = xml.find(
        f"./link[@name='{link_name}']"
    )

    if link is None:
        raise ValueError(
            f"找不到脚部连杆：{link_name}"
        )

    vertices = []

    for collision in link.findall("collision"):
        mesh_element = collision.find("geometry/mesh")

        if mesh_element is None:
            raise ValueError(
                "当前脚部检查工具只支持 mesh 碰撞几何"
            )

        mesh_path = (
            urdf_path.parent
            / mesh_element.attrib["filename"]
        )

        mesh = trimesh.load_mesh(
            str(mesh_path),
            process=False,
        )

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(
                f"不能读取为单个三角网格：{mesh_path}"
            )

        scale = np.fromstring(
            mesh_element.get("scale", "1 1 1"),
            sep=" ",
        )

        points = np.asarray(mesh.vertices) * scale

        origin = origin_transform(
            collision.find("origin")
        )

        vertices.append(
            points @ origin[:3, :3].T
            + origin[:3, 3]
        )

    if not vertices:
        raise ValueError(
            f"脚部没有碰撞网格：{link_name}"
        )

    return np.concatenate(vertices)


def main():
    config_path = (
        PROJECT_ROOT
        / "data/motions/robotcupmini/retarget_config.json"
    )
    config = json.loads(config_path.read_text())

    urdf_path = (
        PROJECT_ROOT / config["target"]["urdf_file"]
    )

    xml, root, joints, limits = read_urdf(urdf_path)
    names = config["joint_names"]

    if len(names) != 13 or set(names) != set(limits):
        raise ValueError(
            "配置中的 13 个关节与 URDF 不一致"
        )

    # 与当前 robotcupmini_cfg.py 的默认关节姿态一致。
    # 这里直接设置数值，避免导入 IsaacLab。
    angles = {name: 0.0 for name in names}

    for side in ("left", "right"):
        angles[f"{side}_hip_pitch_joint"] = -0.075
        angles[f"{side}_knee_joint"] = 0.15
        angles[f"{side}_ankle_pitch_joint"] = -0.075

    for name, value in angles.items():
        lower, upper = limits[name]

        if not lower <= value <= upper:
            raise ValueError(
                f"默认关节角越界：{name}"
            )

    # 与当前机器人配置中的初始根部高度一致。
    base_height = 0.571

    transforms = forward_kinematics(
        root, joints, angles, base_height
    )

    report = {
        "root_link": root,
        "base_height_m": base_height,
        "joint_angles_rad": angles,
        "landmarks_m": {},
        "feet": {},
        "status": "kinematics_only_not_dynamics_validated",
    }

    print(
        f"根连杆：{root}；活动关节：{len(limits)}"
    )

    for label, mapping in config["landmarks"].items():
        position = transforms[
            mapping["robot_link"]
        ][:3, 3]

        report["landmarks_m"][label] = position.tolist()

        print(
            f"{label:12s} xyz = "
            f"{np.round(position, 5)} m"
        )

    figure, axes = plt.subplots(
        1, 2, figsize=(10, 5)
    )

    # 绘制各连杆原点之间的连接线。
    for joint in joints:
        points = np.stack([
            transforms[joint.parent][:3, 3],
            transforms[joint.child][:3, 3],
        ])

        color = (
            "tab:blue"
            if joint.name.startswith("left")
            else "tab:red"
            if joint.name.startswith("right")
            else "gray"
        )

        for axis, horizontal in zip(axes, (0, 1)):
            axis.plot(
                points[:, horizontal],
                points[:, 2],
                "o-",
                color=color,
            )

    # 把脚部碰撞网格转换到世界坐标。
    for side, link_name in (
        config["foot_orientation_links"].items()
    ):
        local = foot_vertices(
            xml, urdf_path, link_name
        )

        transform = transforms[link_name]

        world = (
            local @ transform[:3, :3].T
            + transform[:3, 3]
        )

        minimum = float(world[:, 2].min())

        report["feet"][side] = {
            "link": link_name,
            "mesh_min_local_m": local.min(
                axis=0
            ).tolist(),
            "mesh_max_local_m": local.max(
                axis=0
            ).tolist(),
            "mesh_lowest_world_z_m": minimum,
        }

        print(
            f"{side:5s} 脚碰撞网格最低点 "
            f"z = {minimum:.6f} m"
        )

        for axis, horizontal in zip(axes, (0, 1)):
            axis.scatter(
                world[:, horizontal],
                world[:, 2],
                s=2,
                alpha=0.3,
            )

    for axis, label in zip(
        axes,
        ("Side view: X-Z", "Front view: Y-Z"),
    ):
        axis.axhline(
            0, color="black", linestyle="--"
        )
        axis.set_title(label)
        axis.set_ylabel("Z (m)")
        axis.set_aspect(
            "equal", adjustable="box"
        )
        axis.grid(True)

    axes[0].set_xlabel("X (m)")
    axes[1].set_xlabel("Y (m)")

    figure.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    figure.savefig(
        OUTPUT_DIR / "robot_fk.png",
        dpi=150,
    )
    plt.close(figure)

    (OUTPUT_DIR / "robot_fk.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"输出目录：{OUTPUT_DIR}")
    print(
        "仅检查几何关系；不代表已经通过"
        "站立、碰撞或动力学验证。"
    )


if __name__ == "__main__":
    main()