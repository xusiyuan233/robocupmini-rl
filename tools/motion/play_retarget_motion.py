"""在 MuJoCo 中按记录姿态回放候选动作；没有策略推理或物理积分。"""

import argparse
import json
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any, cast

import mujoco as _mujoco
import numpy as np

# MuJoCo 的 C 扩展导出未被当前静态检查器完整识别。
# 只在此外部库边界使用 Any，不关闭整个文件的类型检查。
mujoco = cast(Any, _mujoco)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOTION = (PROJECT_ROOT / "data/motions/robotcupmini/retargeted"
                  / "walk1_candidate_v0/candidate_motion.npz")
MODEL_PATH = (PROJECT_ROOT / "exts/bipedal_locomotion/bipedal_locomotion"
              / "assets/usd/ROBOCUPMINI/mjcf/robotcupmini.xml")
VIEWS = {"side": 90.0, "front": 0.0, "oblique": 45.0}


def load_motion(path: Path):
    """读取动作，并检查时间、数组形状和四元数约定。"""
    with np.load(path, allow_pickle=False) as archive:
        motion = {name: archive[name].copy() for name in archive.files}
    names = motion["joint_names"].tolist()
    times = motion["time_s"]
    count = len(times)
    dt = float(motion["sample_dt_s"].item())
    if (len(names) != 13 or len(set(names)) != 13
            or not all(isinstance(name, str) for name in names)):
        raise ValueError("动作文件必须包含 13 个不重复的关节名称")
    if count < 2 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("帧数或采样间隔无效")
    expected = {
        "time_s": (count,),
        "joint_positions_rad": (count, 13),
        "root_positions_m": (count, 3),
        "root_quaternions_xyzw": (count, 4),
    }
    for name, shape in expected.items():
        values = motion[name]
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"数据形状或数值无效：{name}")
    if not np.allclose(np.diff(times), dt, atol=1e-8, rtol=1e-6):
        raise ValueError("时间轴与采样间隔不一致")
    norms = np.linalg.norm(motion["root_quaternions_xyzw"], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6, rtol=0):
        raise ValueError("骨盆四元数不是单位四元数")
    return motion, names, dt


def build_model(path: Path):
    """只在内存中添加回放用地面与灯光，不修改原始 XML。"""
    xml = ET.parse(path).getroot()
    compiler = xml.find("compiler")
    if compiler is None:
        raise ValueError("MJCF 缺少 compiler")
    # from_xml_string 不再以原文件目录为基准，因此使用绝对网格路径。
    mesh_directory = (path.parent / compiler.get("meshdir", ".")).resolve()
    compiler.set("meshdir", str(mesh_directory))
    world = xml.find("worldbody")
    assets = xml.find("asset")
    if world is None or assets is None:
        raise ValueError("MJCF 缺少 worldbody 或 asset")
    ET.SubElement(assets, "texture", {
        "name": "playback_grid", "type": "2d", "builtin": "checker",
        "width": "256", "height": "256",
        "rgb1": "0.22 0.25 0.29", "rgb2": "0.36 0.39 0.43",
    })
    ET.SubElement(assets, "material", {
        "name": "playback_floor_material", "texture": "playback_grid",
        "texrepeat": "12 12", "texuniform": "true", "reflectance": "0.05",
    })
    ET.SubElement(world, "geom", {
        "name": "playback_floor", "type": "plane", "size": "6 6 0.1",
        "material": "playback_floor_material", "contype": "0", "conaffinity": "0",
    })
    ET.SubElement(world, "light", {
        "pos": "0 -2 4", "dir": "0 0 -1", "directional": "true",
        "diffuse": "0.8 0.8 0.8", "ambient": "0.3 0.3 0.3",
    })
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    for geom in range(model.ngeom):
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or ""
        # 隐藏碰撞近似体，避免覆盖外观网格；只改内存中的显示分组。
        if model.geom_contype[geom] != 0 or model.geom_conaffinity[geom] != 0:
            model.geom_group[geom] = 3
        elif body_name.startswith("left_"):
            model.geom_rgba[geom] = [0.25, 0.60, 0.95, 1.0]
        elif body_name.startswith("right_"):
            model.geom_rgba[geom] = [0.95, 0.55, 0.20, 1.0]
    return model


def joint_mapping(model, names):
    """逐个按名称查找地址，不假设动作列顺序等于 MJCF 顺序。"""
    actual_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        for joint in range(model.njnt)
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
    }
    if set(names) != actual_names:
        raise ValueError("动作关节名称与机器人模型不一致")
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
           for name in names]
    addresses = np.array([model.jnt_qposadr[joint] for joint in ids], dtype=int)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_freejoint")
    if root < 0 or model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("模型中找不到骨盆自由关节")
    return ids, addresses, int(model.jnt_qposadr[root])


def apply_frame(model, data, motion, addresses, root_address, frame):
    """直接放置一帧姿态，再刷新运动学；不调用 mj_step。"""
    data.qpos[:] = model.qpos0
    data.qpos[addresses] = motion["joint_positions_rad"][frame]
    data.qpos[root_address:root_address + 3] = motion["root_positions_m"][frame]
    quaternion = motion["root_quaternions_xyzw"][frame]
    # 文件是 XYZW，MuJoCo 的自由关节需要 WXYZ。
    data.qpos[root_address + 3:root_address + 7] = quaternion[[3, 0, 1, 2]]
    data.qvel[:] = 0
    data.time = float(motion["time_s"][frame])
    mujoco.mj_forward(model, data)


def check_frames(model, data, motion, names, ids, addresses, root_address):
    """检查全部帧能否正确映射；这不是动力学稳定性检查。"""
    positions = motion["joint_positions_rad"]
    max_limit_error = 0.0
    for column, joint in enumerate(ids):
        if model.jnt_limited[joint]:
            lower, upper = model.jnt_range[joint]
            error = max(0.0, float(lower - positions[:, column].min()),
                        float(positions[:, column].max() - upper))
            max_limit_error = max(max_limit_error, error)
    if max_limit_error > 1e-5:
        raise ValueError(f"候选角度超过 MJCF 限位：{max_limit_error:.6g} rad")
    for frame in range(len(positions)):
        apply_frame(model, data, motion, addresses, root_address, frame)
        if not np.isfinite(data.xpos).all():
            raise ValueError(f"第 {frame} 帧连杆位置无效")
    print(f"[CHECK] {len(positions)} 帧映射通过；最大限位误差={max_limit_error:.3g} rad")
    print("[MAP] " + ", ".join(f"{name} -> qpos[{address}]"
                                   for name, address in zip(names, addresses)))


def periodic_clip_translation(motion, dt):
    """只对已核验的周期动作启用空间延拓；旧候选维持原行为。"""
    if not bool(motion.get("periodic_reference", np.array(False))):
        return None
    count = len(motion["time_s"])
    cycle = round(float(motion["cycle_duration_s"]) / dt)
    shift = np.asarray(motion["cycle_translation_m"], dtype=float)
    if cycle < 3 or count < 2*cycle or count % cycle or shift.shape != (3,):
        raise ValueError("周期元数据无效")
    if not np.allclose(motion["joint_positions_rad"][cycle:], motion["joint_positions_rad"][:-cycle], atol=1e-7):
        raise ValueError("关节姿态不满足声明的周期")
    if not np.allclose(motion["root_quaternions_xyzw"][cycle:], motion["root_quaternions_xyzw"][:-cycle], atol=1e-7):
        raise ValueError("根部旋转不满足声明的周期")
    if not np.allclose(motion["root_positions_m"][cycle:]-motion["root_positions_m"][:-cycle], shift, atol=1e-7):
        raise ValueError("根部位置不满足声明的周期平移")
    return shift * (count // cycle)


def play(model, data, motion, addresses, root_address, dt, args):
    import glfw
    from mujoco import viewer as mj_viewer

    state = {"paused": False, "step": 0}
    state_lock = threading.Lock()

    def on_key(key):
        with state_lock:
            if key == glfw.KEY_SPACE:
                state["paused"] = not state["paused"]
            elif state["paused"] and key == glfw.KEY_RIGHT:
                state["step"] += 1
            elif state["paused"] and key == glfw.KEY_LEFT:
                state["step"] -= 1

    count = len(motion["time_s"])
    duration = count * dt
    clip_shift = periodic_clip_translation(motion, dt)
    cursor = 0.0
    final_notice = False
    with mj_viewer.launch_passive(
            model, data, key_callback=on_key,
            show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.opt.geomgroup[3] = 0
            viewer.cam.azimuth = VIEWS[args.view]
            viewer.cam.elevation = -12
            viewer.cam.distance = 1.8
        previous = time.perf_counter()
        while viewer.is_running():
            now = time.perf_counter()
            elapsed = now - previous
            previous = now
            with state_lock:
                paused, step = state["paused"], state["step"]
                state["step"] = 0
            if not paused:
                cursor += elapsed * args.speed
            if step:
                cursor = (int(cursor / dt) + step) * dt
            clip_iteration = 0
            local_cursor = cursor
            if args.loop:
                if clip_shift is not None:
                    clip_iteration = int(np.floor(cursor / duration))
                    local_cursor = cursor - clip_iteration * duration
                else:
                    cursor %= duration
                    local_cursor = cursor
            else:
                cursor = float(np.clip(cursor, 0.0, (count - 1) * dt))
                if cursor >= (count - 1) * dt and not final_notice:
                    print("播放结束，停在最后一帧。关闭窗口退出。")
                    final_notice = True
                local_cursor = cursor
            frame = min(int((local_cursor + 1e-10) / dt), count - 1)
            with viewer.lock():
                apply_frame(model, data, motion, addresses, root_address, frame)
                if clip_shift is not None and clip_iteration:
                    data.qpos[root_address:root_address+3] += clip_iteration * clip_shift
                    data.time = cursor
                    mujoco.mj_forward(model, data)
                viewer.cam.lookat[:] = data.qpos[root_address:root_address + 3]
                viewer.cam.lookat[2] -= 0.08
                if args.fixed_lateral_camera:
                    # Follow forward travel only; don't hide lateral drift/sway.
                    viewer.cam.lookat[1] = float(motion['root_positions_m'][:, 1].mean())
            viewer.sync()
            time.sleep(0.005)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--view", choices=VIEWS, default="side")
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--fixed-lateral-camera", action="store_true", help="相机只跟随前进方向，不跟随骨盆横摆")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed 必须是正数")
    config = json.loads((PROJECT_ROOT / "data/motions/robotcupmini/retarget_config.json").read_text())
    motion, names, dt = load_motion(args.motion)
    if names != config["joint_names"]:
        raise ValueError("候选动作关节顺序与项目重定向配置不一致")
    model = build_model(MODEL_PATH)
    data = mujoco.MjData(model)
    ids, addresses, root_address = joint_mapping(model, names)
    check_frames(model, data, motion, names, ids, addresses, root_address)
    apply_frame(model, data, motion, addresses, root_address, 0)
    print(f"[PLAY] {len(motion['time_s'])} 帧，采样 {1 / dt:g} Hz；仅姿态回放，没有物理积分")
    print("左腿蓝色，右腿橙色。SPACE 暂停/继续；暂停后 ←/→ 逐帧；关闭窗口退出。")
    if args.loop:
        if periodic_clip_translation(motion, dt) is not None:
            print("周期姿态已校验；循环累计根部前进距离，不跳回起点。")
        else:
            print("注意：循环边界会直接回到首帧；尚未做首尾衔接。")
    if not args.check_only:
        play(model, data, motion, addresses, root_address, dt, args)


if __name__ == "__main__":
    main()
