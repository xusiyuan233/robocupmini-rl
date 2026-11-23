"""带关节限位的双腿 IK。仅做离线几何计算，不训练策略。"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from check_robot_fk import (
    read_urdf,
    forward_kinematics,
    foot_vertices,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs/style_reference_ik"


class RobotLegIK:
    """固定骨盆姿态和腰部，求解两条腿的 12 个关节角。"""

    def __init__(self, project_root: Path):
        self.project_root = project_root

        # 读取上一阶段创建的重定向配置。
        config_path = (
            project_root
            / "data/motions/robotcupmini/retarget_config.json"
        )
        self.config = json.loads(config_path.read_text())

        # 直接读取自己机器人的 URDF。
        path = project_root / self.config["target"]["urdf_file"]
        xml, self.root, self.joints, limits = read_urdf(path)

        # 输出顺序使用配置中的 13 个关节顺序。
        self.names = self.config["joint_names"]

        # 腰部固定，只有两条腿的 12 个角度参与优化。
        self.leg_names = [
            name
            for name in self.names
            if name != "waist_yaw_joint"
        ]

        if len(self.names) != 13 or len(self.leg_names) != 12:
            raise ValueError("需要 12 个腿部关节和 1 个腰部关节")

        if (
            len(set(self.names)) != 13
            or set(self.names) != set(limits)
        ):
            raise ValueError("关节配置与 URDF 不一致")

        # 优化过程中必须遵守 URDF 中的关节角度限位。
        self.lower = np.array([
            limits[name][0] for name in self.leg_names
        ])
        self.upper = np.array([
            limits[name][1] for name in self.leg_names
        ])

        # 与前面的默认站姿保持一致，角度单位为 rad。
        self.default = {name: 0.0 for name in self.names}

        for side in ("left", "right"):
            self.default[f"{side}_hip_pitch_joint"] = -0.075
            self.default[f"{side}_knee_joint"] = 0.15
            self.default[f"{side}_ankle_pitch_joint"] = -0.075

        self.default["waist_yaw_joint"] = float(
            self.config["initial_waist_yaw_rad"]
        )

        for name, value in self.default.items():
            if (
                not np.isfinite(value)
                or not limits[name][0] <= value <= limits[name][1]
            ):
                raise ValueError(f"默认关节角无效：{name}")

        self.foot_links = self.config["foot_orientation_links"]

        if set(self.foot_links) != {"left", "right"}:
            raise ValueError("必须配置 left 和 right 两只脚")

        # 为每只脚选取一个一致的几何定位点。
        self.anchors: dict[str, np.ndarray] = {}

        for side, link in self.foot_links.items():
            vertices = foot_vertices(xml, path, link)
            low = vertices.min(axis=0)
            high = vertices.max(axis=0)

            # 脚部碰撞网格在 link 坐标系中的包围盒底面中心。
            # 这是几何定位点，不是物理仿真测得的实际接触点。
            self.anchors[side] = np.array([
                (low[0] + high[0]) / 2,
                (low[1] + high[1]) / 2,
                low[2],
            ])

    def foot_poses(
        self,
        angles: dict[str, float],
        base_height=0.571,
    ):
        """根据关节角计算左右脚定位点的位置和脚部朝向。"""

        transforms = forward_kinematics(
            self.root,
            self.joints,
            angles,
            base_height,
        )

        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for side, link in self.foot_links.items():
            transform = transforms[link]
            rotation = transform[:3, :3]

            position = (
                rotation @ self.anchors[side]
                + transform[:3, 3]
            )

            poses[side] = (position, rotation)

        return poses

    def solve(
        self,
        targets: dict[str, tuple[np.ndarray, np.ndarray]],
        initial: dict[str, float] | None = None,
        base_height=0.571,
    ):
        """给定左右脚目标位姿，求解完整的 13 个关节角。"""

        if set(targets) != {"left", "right"}:
            raise ValueError("目标必须同时包含左右脚")

        if not np.isfinite(base_height):
            raise ValueError("骨盆高度无效")

        # 检查输入位置和旋转矩阵。
        for position, rotation in targets.values():
            if position.shape != (3,) or rotation.shape != (3, 3):
                raise ValueError("脚位置必须为 3 维，旋转必须为 3×3")

            if (
                not np.isfinite(position).all()
                or not np.isfinite(rotation).all()
            ):
                raise ValueError("目标中存在 NaN 或 Inf")

            if (
                not np.allclose(
                    rotation.T @ rotation,
                    np.eye(3),
                    atol=1e-6,
                )
                or not np.isclose(
                    np.linalg.det(rotation),
                    1.0,
                    atol=1e-6,
                )
            ):
                raise ValueError("目标旋转矩阵无效")

        # 首帧可以从默认站姿开始。
        # 后续处理动作序列时，可以传入上一帧的结果。
        seed = self.default if initial is None else initial

        if set(seed) != set(self.names):
            raise ValueError("初始角度必须包含完整的 13 个关节")

        if not np.isclose(
            seed["waist_yaw_joint"],
            self.default["waist_yaw_joint"],
        ):
            raise ValueError("当前求解器要求腰部保持配置中的固定值")

        x0 = np.array([
            seed[name] for name in self.leg_names
        ])

        if (
            not np.isfinite(x0).all()
            or np.any(x0 < self.lower)
            or np.any(x0 > self.upper)
        ):
            raise ValueError("初始腿部关节角无效或越界")

        def unpack(vector):
            """将优化器的 12 维向量转换成完整关节角字典。"""

            angles = self.default.copy()
            angles.update({
                name: float(value)
                for name, value in zip(self.leg_names, vector)
            })
            return angles

        def residual(vector):
            """计算当前双脚与目标双脚之间的位置和朝向误差。"""

            poses = self.foot_poses(
                unpack(vector),
                base_height,
            )

            errors = []

            for side in ("left", "right"):
                position, rotation = poses[side]
                target_position, target_rotation = targets[side]

                # 用 1 cm 对位置误差做数值尺度归一化。
                errors.extend(
                    (position - target_position) / 0.01
                )

                # 用旋转向量表示朝向误差，单位为 rad。
                rotation_error = Rotation.from_matrix(
                    target_rotation.T @ rotation
                ).as_rotvec()

                # 用 0.05 rad 对朝向误差做尺度归一化。
                errors.extend(rotation_error / 0.05)

            return np.asarray(errors)

        # 在关节限位内寻找误差尽可能小的解。
        result = least_squares(
            residual,
            x0,
            bounds=(self.lower, self.upper),
            method="trf",
            max_nfev=250,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )

        angles = unpack(result.x)
        poses = self.foot_poses(angles, base_height)

        position_errors = []
        orientation_errors = []

        for side in ("left", "right"):
            position, rotation = poses[side]
            target_position, target_rotation = targets[side]

            position_errors.append(
                float(np.linalg.norm(position - target_position))
            )

            orientation_errors.append(
                float(np.degrees(
                    Rotation.from_matrix(
                        target_rotation.T @ rotation
                    ).magnitude()
                ))
            )

        max_position = max(position_errors)
        max_orientation = max(orientation_errors)

        within_limits = bool(
            np.all(result.x >= self.lower - 1e-10)
            and np.all(result.x <= self.upper + 1e-10)
        )

        # 优化器结束不等于目标真正可达。
        # 还必须检查位置误差、朝向误差和关节限位。
        diagnostics = {
            "accepted": bool(
                result.success
                and within_limits
                and max_position <= 0.001
                and max_orientation <= 0.5
            ),
            "solver_success": bool(result.success),
            "max_position_error_m": max_position,
            "max_orientation_error_deg": max_orientation,
            "within_joint_limits": within_limits,
            "function_evaluations": int(result.nfev),
        }

        return angles, diagnostics


def main():
    ik = RobotLegIK(PROJECT_ROOT)

    # 测试一：用已知的合法关节角生成确定可达的脚部目标。
    known = ik.default.copy()

    for side, values in (
        ("left", (-0.18, 0.025, 0.04, 0.36, -0.18, -0.025)),
        ("right", (-0.12, -0.02, -0.03, 0.24, -0.12, 0.02)),
    ):
        for suffix, value in zip(
            (
                "hip_pitch",
                "hip_roll",
                "hip_yaw",
                "knee",
                "ankle_pitch",
                "ankle_roll",
            ),
            values,
        ):
            known[f"{side}_{suffix}_joint"] = value

    targets = ik.foot_poses(known)
    recovered, good = ik.solve(targets)

    print("可达目标自检：")
    print(json.dumps(good, indent=2, ensure_ascii=False))

    if not good["accepted"]:
        raise RuntimeError("可达目标自检失败，请先不要处理人体动作")

    # 测试二：将左脚目标抬高 2 m，构造明显不可达的目标。
    impossible = {
        side: (position.copy(), rotation.copy())
        for side, (position, rotation) in targets.items()
    }
    impossible["left"][0][2] += 2.0

    _, bad = ik.solve(impossible)

    print("不可达目标是否被拒绝：", not bad["accepted"])

    if bad["accepted"]:
        raise RuntimeError("不可达目标被误接受")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "status": "ik_selfcheck_only_not_retargeted_motion",
        "reachable_test": good,
        "unreachable_test": bad,
        "known_joint_angles_rad": known,
        "recovered_joint_angles_rad": recovered,
    }

    report_path = OUTPUT_DIR / "ik_selfcheck.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"报告：{report_path}")
    print("IK 自检通过；尚未生成人体重定向动作，也没有开始训练。")


if __name__ == "__main__":
    main()