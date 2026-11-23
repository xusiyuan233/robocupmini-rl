"""RobotcupMini奖励补充，只用于当前平地训练。"""
# 【阅读约定】本次只增加 # 注释，不修改原来的函数、公式、参数和文档字符串。
# 本文件定义 7 个普通奖励函数，以及 1 个需要保存历史状态的奖励计算类。
# N 表示并行环境数；F=2 表示左右脚；J 表示被选择的关节数。
# 例如 [N, F, 3] 可理解为：N 个机器人，每个有 F 只脚，每只脚有 xyz 三个数。
# 切片中的 : 表示全选；... 表示省略前面的维度；索引从 0 开始。
# _w 表示世界坐标系；下面使用的 yaw 水平坐标系只随骨盆偏航旋转，不随倾斜旋转。
# 误差函数通常返回 [N] 非负数，由环境配置中的负权重转成惩罚；前进评分函数除外。
# 本地 RewardManager 的计算方式：函数返回值 × 配置权重 × env.step_dt，然后累加。
# 定义函数不等于启用奖励：是否启用、权重和参数以环境配置中的 RewTerm 为准。
# 下面函数参数中的默认值可被 RewTerm.params 覆盖，不一定是当前训练实际采用的值。
# from __future__ 导入影响本模块类型标注的求值方式；不会创建任何机器人或启动训练。
from __future__ import annotations

# 导入 PyTorch，并使用简写 torch；后续张量可以在 CPU 或 GPU 上批量计算。
import torch
# 从 IsaacLab 导入奖励类的基类，以及选择场景实体/关节/连杆的配置类。
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
# 导入四元数工具：quat_apply 做正向旋转，quat_apply_inverse 做逆旋转；
# yaw_quat 只保留朝向中的偏航部分。这里四元数采用 (w, x, y, z) 顺序。
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat


# ==================== 1 前进速度评分函数 ====================
# def 定义普通函数；这里的参数会在奖励管理器调用时传入。
def track_forward_velocity_linear(
    # env 是实际运行的环境对象，不是 EnvCfg 配置对象。
    env,
    # command_name 是速度命令项的名字；: str 是类型标注。
    command_name: str,
    # velocity_scale 是速度误差的归一化尺度，单位 m/s，应当大于 0。
    velocity_scale: float,
    # asset_cfg 指定要读取的机器人；其 name 常为 "robot"。
    asset_cfg: SceneEntityCfg,
    # -> torch.Tensor 是返回类型标注；本函数实际返回每个环境一个分数 [N]。
) -> torch.Tensor:
    """按绝对速度误差评分，允许负分，不做截断。"""
    # 按名字从场景中取得机器人对象；robot 只是本函数使用的局部变量名。
    robot = env.scene[asset_cfg.name]
    # 将世界系速度逆旋转到骨盆的水平朝向坐标系。
    velocity = quat_apply_inverse(
        # 取机器人根部姿态，并去掉 roll/pitch，仅保留 yaw 偏航。
        yaw_quat(robot.data.root_quat_w),
        # 读取所有环境的根部世界系线速度 xyz，形状 [N, 3]。
        robot.data.root_lin_vel_w[:, :3],
    # 旋转后的结果仍为 [N, 3]；[:, 0] 只取前向 x 速度，得到 [N]。
    )[:, 0]
    # 读取命令向量第一列，即每个环境的目标前进速度，得到 [N]。
    command = env.command_manager.get_command(command_name)[:, 0]
    # 绝对误差为 0 时得 1；误差等于 scale 时得 0；更大时可得负分，不做截断。
    return 1.0 - torch.abs(command - velocity) / velocity_scale


# ==================== 2 摆动脚离地高度误差 ====================
def feet_phase_height_error(
    # env 提供机器人状态、命令管理器和各环境的地面原点。
    env,
    # command_name 在这里指步态命令项，而不是速度命令项。
    command_name: str,
    # 速度命令项名字，用来判断当前环境是否处于站立命令模式。
    velocity_command_name: str,
    # asset_cfg 应按左脚、右脚的顺序选择两只脚的连杆。
    asset_cfg: SceneEntityCfg,
    # rest_height 是足底相对脚连杆原点的向下偏移距离，单位 m。
    rest_height: float,
    # lift_height 是参考摆动轨迹的峰值离地高度，单位 m。
    lift_height: float,
    # height_scale 是高度误差的归一化尺度，单位 m，应大于 0。
    height_scale: float,
    # 返回 [N]；值越大，表示摆动期的足底近似高度越偏离目标。
) -> torch.Tensor:
    """左右脚分别跟踪相位高度，返回正误差，由负权重施加惩罚。"""
    # 取出机器人对象。
    robot = env.scene[asset_cfg.name]
    # get_term 取得步态命令对象，可以进一步读取它的相位和参数。
    gait = env.command_manager.get_term(command_name)
    # 步态命令表 [N, 4]：频率、左右相位偏移、支撑比例、摆动高度。
    params = gait.command
    # gait_indices 是每个环境在一个步态周期中的归一化相位 [N]，范围 [0, 1)。
    phase = gait.gait_indices

    # 顺序必须为左脚、右脚，与官方接触相位定义一致。
    # stack 把两条 [N] 相位序列合并成 [N, 2]，两列对应左右脚。
    foot_phase = torch.stack(
        # 右脚相位=左脚相位+偏移，再对 1 取余，使周期在 [0, 1) 内循环。
        (phase, torch.remainder(phase + params[:, 1], 1.0)),
        # dim=1 表示新增第二个维度作为左右脚维度。
        dim=1,
    )
    # 取第三列支撑比例；2:3 保留列维度 [N, 1]，便于与 [N, 2] 广播运算。
    stance_ratio = params[:, 2:3]
    # 相位达到支撑比例后视为摆动；swing 是 [N, 2] 的布尔掩码。
    swing = foot_phase >= stance_ratio
    # 将摆动阶段的相位映射为 0～1 的进度。
    progress = (
        # 减去支撑阶段长度，再除以摆动阶段长度；假设支撑比例介于 0 与 1 之间。
        (foot_phase - stance_ratio) / (1.0 - stance_ratio)
    # clamp 将结果限制在 [0, 1]；支撑期得到的负进度会变成 0。
    ).clamp(0.0, 1.0)
    # 半个正弦波：起脚和落脚目标为 0，中间目标为 lift_height。
    target = lift_height * torch.sin(torch.pi * progress)
    # where 按位置选择：摆动期用正弦目标，支撑期用同形状的 0。
    target = torch.where(swing, target, torch.zeros_like(target))

    # 当前机器人足底的包围矩形，顺序为左脚、右脚。
    # 这是保守几何近似，单位 m，仅用于当前平地模型。
    # 当前本地 IsaacLab 中 body_pos_w 是连杆原点位置；选择两只脚后为 [N, 2, 3]。
    foot_pos = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    # 两只脚的世界姿态四元数，形状 [N, 2, 4]。
    foot_quat = robot.data.body_quat_w[:, asset_cfg.body_ids, :]
    # 以 foot_pos 相同的数据类型和设备创建四角表，形状 [2, 4, 3]。
    corners = foot_pos.new_tensor([
        # 左脚局部坐标第 1 个角点：x=-0.05、y=-0.04、z=-rest_height。
        [[-0.05, -0.04, -rest_height],
         # 左脚第 2 个角点：负 x、正 y。
         [-0.05,  0.04, -rest_height],
         # 左脚第 3 个角点：正 x、负 y。
         [ 0.15, -0.04, -rest_height],
         # 左脚第 4 个角点：正 x、正 y。
         [ 0.15,  0.04, -rest_height]],
        # 右脚局部坐标第 1 个角点；左右脚几何范围不完全相同。
        [[-0.06, -0.04, -rest_height],
         # 右脚第 2 个角点。
         [-0.06,  0.04, -rest_height],
         # 右脚第 3 个角点。
         [ 0.14, -0.04, -rest_height],
         # 右脚第 4 个角点；这四角是几何近似，不是实时网格接触点。
         [ 0.14,  0.04, -rest_height]],
    ])
    # unsqueeze(0) 新增环境维；expand 扩展成 [N, 2, 4, 3]；-1 表示该维不变。
    points = corners.unsqueeze(0).expand(foot_pos.shape[0], -1, -1, -1)
    # 为每只脚的四个角点配上同一姿态，得到 [N, 2, 4, 4]。
    rotations = foot_quat.unsqueeze(2).expand(-1, -1, 4, -1)
    # 先旋转局部角点，再平移到脚连杆原点，得到世界坐标中的角点。
    points_w = quat_apply(
        # reshape(-1, 4/3) 暂时展平批次维；-1 让 PyTorch 自动推算行数。
        rotations.reshape(-1, 4), points.reshape(-1, 3)
    # reshape_as 恢复 points 的形状；原点扩成 [N, 2, 1, 3] 后广播相加。
    ).reshape_as(points) + foot_pos.unsqueeze(2)

    # 每个环境原点的世界 z 坐标 [N, 1]；此处将其当作平地高度，不适用于任意起伏地形。
    ground_z = env.scene.env_origins[:, 2:3]
    # ... 取所有批次与脚；取角点 z，再沿四角维取最小值，减地面高度得 [N, 2]。
    clearance = points_w[..., 2].amin(dim=2) - ground_z

    # 只在各自的摆动相位检查足底离地高度。
    # 对两只脚的误差取平均，输出每个环境的误差 [N]。
    error = torch.mean(
        # 先归一化高度差，再平方，使正负偏差都会产生非负代价。
        torch.square((clearance - target) / height_scale)
        # 将布尔 swing 转成浮点数：摆动脚乘 1，支撑脚乘 0。
        * swing.to(foot_pos.dtype),
        # dim=1 在固定的两只脚维度求均值，不是只对当前摆动脚数量求均值。
        dim=1,
    )
    # 取得速度命令对象，用其站立标记屏蔽本项。
    standing = env.command_manager.get_term(
        # 传入速度命令项的名字。
        velocity_command_name
    # 读取 [N] 布尔标记；True 表示命令生成器把该环境标为站立。
    ).is_standing_env
    # ~ 对布尔张量逐项取反；移动环境保留误差，站立环境返回 0。
    return error * (~standing).to(error.dtype)

# ==================== 3 左右脚横向间距误差 ====================
def feet_lateral_spacing_error(
    # env 是当前环境。
    env,
    # asset_cfg 必须按左、右顺序选脚，否则下面间距的符号会颠倒。
    asset_cfg: SceneEntityCfg,
    # 允许的最小左右连杆间距，默认 0.16 m。
    min_width: float = 0.16,
    # 允许的最大间距，默认 0.28 m。
    max_width: float = 0.28,
    # 额外温和引导的目标间距，默认 0.22 m。
    target_width: float = 0.22,
    # 归一化距离尺度，默认 0.10 m，应大于 0。
    error_scale: float = 0.10,
    # 返回每个环境的间距误差 [N]。
) -> torch.Tensor:
    """惩罚交叉、过宽落脚，并温和引导目标步宽。"""
    # 按配置选择机器人。
    robot = env.scene[asset_cfg.name]

    # 顺序必须为左脚、右脚。
    # 读取左右脚连杆原点的世界坐标 [N, 2, 3]，不是脚掌内缘间距。
    feet = robot.data.body_pos_w[:, asset_cfg.body_ids, :]

    # 将左脚相对右脚的位移转到骨盆水平朝向坐标系。
    delta = quat_apply_inverse(
        # 只取骨盆偏航，避免直接把躯干倾斜当作左右方向变化。
        yaw_quat(robot.data.root_quat_w),
        # 左脚位置减右脚位置，得到 [N, 3] 的相对位移。
        feet[:, 0, :] - feet[:, 1, :],
    )

    # 保留符号，交叉后为负。
    # 取 y 分量 [N]；左脚在骨盆坐标的左侧时通常为正，左右关系反转后为负。
    width = delta[:, 1]

    # relu(x)=max(x, 0)：仅在宽度小于最小值时得到正的过窄量。
    too_narrow = torch.relu(min_width - width)
    # 仅在宽度超过最大值时得到正的过宽量。
    too_wide = torch.relu(width - max_width)
    # 计算相对目标间距的归一化偏差，可能为正或负。
    target_error = (width - target_width) / error_scale

    # 将三项代价相加，不在此处设置最终奖励权重。
    return (
        # 过窄量归一化后平方。
        torch.square(too_narrow / error_scale)
        # 过宽量归一化后平方。
        + torch.square(too_wide / error_scale)
        # 另外叠加目标间距平方误差，系数为 0.25；区间内也不一定完全无惩罚。
        + 0.25 * torch.square(target_error)
    )

# ==================== 4 脚尖水平朝向误差 ====================
def feet_forward_heading_error(
    # env 是当前环境。
    env,
    # asset_cfg 按顺序选择左右脚连杆。
    asset_cfg: SceneEntityCfg,
    # 可容忍的偏角默认约 10°；代码使用弧度，配置可覆盖此值。
    tolerance: float = 0.174533,
    # 偏角归一化尺度默认约 20°，同样以弧度表示，应大于 0。
    angle_scale: float = 0.349066,
    # 输出左右脚平均的超出容差部分的平方误差 [N]。
) -> torch.Tensor:
    """惩罚脚尖相对骨盆前进方向的过大偏角。"""
    # 取得机器人。
    robot = env.scene[asset_cfg.name]
    # 读取两脚姿态 [N, 2, 4]。
    foot_quat = robot.data.body_quat_w[:, asset_cfg.body_ids, :]

    # 脚掌连杆的 +X 轴为脚尖方向。
    # 只借用形状创建 [N, 2, 3] 的全零方向表；不是把四元数截断来表示方向。
    forward = torch.zeros_like(foot_quat[..., :3])
    # 将 x 分量设为 1，形成每只脚局部坐标下的单位方向向量 (1, 0, 0)。
    forward[..., 0] = 1.0

    # 把脚局部的前向单位向量旋转到世界系。
    forward_w = quat_apply(
        # 将两脚四元数展平为 [N*2, 4]。
        foot_quat.reshape(-1, 4),
        # 将方向向量展平为 [N*2, 3]，逐行对应四元数。
        forward.reshape(-1, 3),
    # 恢复为 [N, 2, 3]。
    ).reshape_as(forward)

    # 转换到只随骨盆朝向旋转的水平坐标系。
    # 取骨盆偏航四元数 [N, 4]。
    base_yaw = yaw_quat(robot.data.root_quat_w)
    # None 增加脚这一维，再扩展成 [N, 2, 4]，给两脚使用相同骨盆朝向。
    base_yaw = base_yaw[:, None, :].expand_as(foot_quat)

    # 将世界系脚尖方向逆旋转到骨盆水平朝向系。
    forward_b = quat_apply_inverse(
        # 展平骨盆朝向。
        base_yaw.reshape(-1, 4),
        # 展平世界系脚尖方向。
        forward_w.reshape(-1, 3),
    # 恢复为每环境、每只脚、xyz 三分量。
    ).reshape_as(forward)

    # atan2(y, x) 由水平投影求有符号偏角，单位 rad；不是直接读取某个关节角。
    angle = torch.atan2(
        # 脚尖方向的水平 y 分量。
        forward_b[..., 1],
        # 脚尖方向的水平 x 分量。
        forward_b[..., 0],
    )

    # 先取偏角绝对值，再扣容差；未超过容差的部分用 relu 变为 0。
    excess = torch.relu(torch.abs(angle) - tolerance)

    # 对左右脚求平均，不区分支撑或摆动，也未在本函数中屏蔽站立。
    return torch.mean(
        # 将超容差偏角归一化并平方。
        torch.square(excess / angle_scale),
        # 沿脚维度求平均，输出 [N]。
        dim=1,
    )
# ==================== 5 摆动和支撑中段接触误差 ====================
def feet_mid_swing_contact_error(
    # env 提供步态命令、速度命令和接触传感器。
    env,
    # 步态命令名字。
    command_name: str,
    # 速度命令名字，用于站立屏蔽。
    velocity_command_name: str,
    # sensor_cfg 从接触传感器中按左、右顺序选择两只脚。
    sensor_cfg: SceneEntityCfg,
    # 摆动期允许的接触合力阈值，默认 5 N。
    force_threshold: float = 5.0,
    # 摆动接触误差的归一化力尺度，默认 50 N，应大于 0。
    force_scale: float = 50.0,
    # 支撑中段期望至少达到的竖直力，默认 40 N，同时用作归一化分母。
    stance_force_threshold: float = 40.0,
    # 返回左右脚平均的接触误差 [N]；该函数既检查摆动也检查支撑。
) -> torch.Tensor:
    """约束摆动中段离地，以及支撑中段保持最低竖直接触力。"""
    # 取得步态命令对象。
    gait = env.command_manager.get_term(command_name)
    # 读取 [N, 4] 的步态参数。
    params = gait.command
    # 读取 [N] 的当前周期相位。
    phase = gait.gait_indices
    # 按左、右顺序组合两脚相位。
    foot_phase = torch.stack(
        # 相位加左右偏移后取余，保证每只脚相位处于 [0, 1)。
        (phase, torch.remainder(phase + params[:, 1], 1.0)),
        # 在维度 1 上合并为 [N, 2]。
        dim=1,
    )
    # 支撑比例保留为 [N, 1]，使左右脚共享该环境的比例。
    stance_ratio = params[:, 2:3]
    # 将摆动部分映射成进度；支撑阶段会得到负值，后面用窗口排除。
    swing_progress = (foot_phase - stance_ratio) / (1.0 - stance_ratio)
    # 将支撑部分映射成进度；摆动阶段会大于或等于 1，后面用窗口排除。
    stance_progress = foot_phase / stance_ratio
    # 只在摆动进度 20%～80% 的窗口计摆动代价；& 是布尔张量逐项“与”。
    mid_swing = (swing_progress >= 0.2) & (swing_progress <= 0.8)
    # 只在支撑进度 20%～80% 的窗口计支撑代价。
    mid_stance = (stance_progress >= 0.2) & (stance_progress <= 0.8)

    # 从场景的 sensors 容器中选择接触传感器对象。
    sensor = env.scene.sensors[sensor_cfg.name]
    # 读取两脚世界系接触合力 [N, 2, 3]，单位 N。
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    # 沿 xyz 最后一维求欧氏长度，得到接触合力大小 [N, 2]。
    force_norm = torch.linalg.vector_norm(forces, dim=-1)
    # 合力超过阈值才计代价；clamp 限制到 [0, 1]，因此此项会饱和。
    swing_error = ((force_norm - force_threshold) / force_scale).clamp(0.0, 1.0)

    # 取世界 z 接触力，并把负值当作 0；这里与上面的三轴合力大小不同。
    vertical_force = forces[..., 2].clamp_min(0.0)
    # 计算支撑脚竖直力不足比例。
    stance_error = (
        # 竖直力低于期望时得到正值；高于期望时为负，随后被截断。
        (stance_force_threshold - vertical_force) / stance_force_threshold
    # 将不足程度限制为 [0, 1]。
    ).clamp(0.0, 1.0)
    # 合并摆动接触和支撑力不足两项，再对左右脚求平均。
    error = torch.mean(
        # 摆动代价仅在 mid_swing 对应位置生效。
        swing_error * mid_swing.to(force_norm.dtype)
        # 支撑力不足代价仅在 mid_stance 生效，并额外乘 0.5。
        + 0.5 * stance_error * mid_stance.to(force_norm.dtype),
        # dim=1 对左右两脚求平均，输出 [N]。
        dim=1,
    )
    # 查询当前环境是否被命令生成器指定为站立。
    standing = env.command_manager.get_term(
        # 速度命令项名字。
        velocity_command_name
    # 得到 [N] 的站立布尔表。
    ).is_standing_env
    # 站立环境把本项代价归零；这里不通过实际速度是否接近 0 来判断站立。
    return error * (~standing).to(error.dtype)

# ==================== 6 有历史记忆的摆动连续性计算类 ====================
# class 定义类；括号里的 ManagerTermBase 是父类。
# 奖励管理器创建该类的实例后，实例会持续保存每个环境、每只脚的历史状态。
# 这不是判断单帧姿态是否好看，而是检测一段摆动是否提前触地、又再次起脚。
class FeetSwingContinuityError(ManagerTermBase):
    """Penalize an interrupted swing, with independent state per environment/foot."""

    # 创建实例时执行；self 指当前计算器实例，cfg 是单项奖励配置，env 是实际环境。
    def __init__(self, cfg, env):
        # 先执行父类的初始化，建立框架需要的基础属性；不会另建一个机器人环境。
        super().__init__(cfg, env)
        # 所有历史表都有 N 行、2 列，分别对应每个环境的左脚和右脚。
        shape = (env.num_envs, 2)
        # 初始接触状态设 True；bool 表示每个元素存放真/假，device 与环境保持一致。
        self._contact = torch.ones(shape, dtype=torch.bool, device=env.device)
        # 记录是否已有可信初始化状态；初始均为 False。
        self._initialized = torch.zeros(shape, dtype=torch.bool, device=env.device)
        # 记录上一控制步是否处于摆动阶段；zeros_like 保持形状、类型和设备一致。
        self._was_swing = torch.zeros_like(self._contact)
        # 记录当前周期是否已具备足够相位历史，可用于连续性判断。
        self._valid_cycle = torch.zeros_like(self._contact)
        # 记录这次摆动中是否已确认过离地。
        self._had_air = torch.zeros_like(self._contact)
        # 记录这次摆动是否已经发生提前触地，后续会持续记住该状态。
        self._broken = torch.zeros_like(self._contact)
        # 记录本次摆动累计的离地事件次数；long 是整数类型。
        self._liftoffs = torch.zeros(shape, dtype=torch.long, device=env.device)
        # 连续达到接触阈值的时间表，默认浮点数，单位 s。
        self._contact_time = torch.zeros(shape, device=env.device)
        # 连续低于离地阈值的时间表，单位 s。
        self._air_time = torch.zeros_like(self._contact_time)

    # 回合重置时由框架调用；env_ids=None 表示重置全部环境。
    def reset(self, env_ids=None):
        # 条件表达式：若未给编号就选全部行 slice(None)，否则只选指定环境行。
        ids = slice(None) if env_ids is None else env_ids
        # 将选中的脚接触状态恢复为 True。
        self._contact[ids] = True
        # 循环访问这些布尔状态表；buffer 引用原表，不是独立副本。
        for buffer in (self._initialized, self._was_swing, self._valid_cycle,
                       self._had_air, self._broken):
            # 把选中环境的这些布尔记录逐一清为 False。
            buffer[ids] = False
        # 再遍历整数计数表和浮点时间表。
        for buffer in (self._liftoffs, self._contact_time, self._air_time):
            # 将选中环境的离地次数、接触时间、离地时间统一清零。
            buffer[ids] = 0

    # 定义 __call__，使计算器对象可像函数一样调用；管理器每个环境步计算奖励时调用。
    def __call__(
        # self 是实例；env 是环境；后两个字符串是步态、速度命令项名字。
        self, env, command_name: str, velocity_command_name: str,
        # sensor_cfg 选择脚部传感器；contact_on 默认达到 15 N 才开始确认接触。
        sensor_cfg: SceneEntityCfg, contact_on: float = 15.0,
        # contact_off 默认低到 5 N 才开始确认离地；debounce_s 是持续确认时间，默认 0.04 s。
        contact_off: float = 5.0, debounce_s: float = 0.04,
        # 核心摆动窗口默认是摆动进度 10%～90%，两端为起脚和落脚留出空间。
        core_start: float = 0.1, core_end: float = 0.9,
        # 再次起脚事件的时间尺度；会除以 dt，避免单次事件惩罚随步长直接缩放。
        repeat_event_cost_s: float = 0.10,
        # 返回每个环境的连续性误差 [N]。
    ) -> torch.Tensor:
        # 取得当前步态命令对象。
        gait = env.command_manager.get_term(command_name)
        # 读取频率、相位偏移、支撑比例、摆动高度参数表。
        params = gait.command
        # 把左脚相位与偏移后的右脚相位组合成 [N, 2]。
        phase = torch.stack((gait.gait_indices,
                             # % 1.0 是取余，效果与前面 torch.remainder(..., 1.0) 对这里的正周期用法一致。
                             (gait.gait_indices + params[:, 1]) % 1.0), dim=1)
        # 取支撑比例 [N, 1]。
        stance_ratio = params[:, 2:3]
        # 相位大于或等于支撑比例时标为摆动 [N, 2]。
        swing = phase >= stance_ratio
        # 换算摆动进度；只在后续有效窗口内使用，不在此行截断。
        progress = (phase - stance_ratio) / (1.0 - stance_ratio)
        # 对站立标记取反，找出执行移动命令的环境。
        moving = (~env.command_manager.get_term(
            # [:, None] 将 [N] 扩为 [N, 1]，方便给左右脚使用同一个移动标记。
            velocity_command_name).is_standing_env)[:, None]
        # 把移动标记广播为 [N, 2]；这是按命令启用，不是按实际速度启用。
        active = moving.expand_as(swing)
        # dt 是一次环境/策略控制步的时长，不是单个物理子步时长，单位 s。
        dt = env.step_dt
        # 选择两脚接触力，并沿 xyz 求模；这里采用合力大小，而非只取竖直力。
        force = env.scene.sensors[sensor_cfg.name].data.net_forces_w[
            :, sensor_cfg.body_ids, :].norm(dim=-1)

        # 接触滞回与去抖：超过 contact_on 连续 debounce_s 才确认接触；
        # 低于 contact_off 连续 debounce_s 才确认离地。两个阈值间保持上次状态。
        # 默认持续时间为 40 ms，可由奖励参数覆盖。
        # Hysteresis and 40 ms confirmation suppress brief contact chatter.
        # fresh=True 表示尚未建立有效历史，例如刚重置或刚从站立切到移动。
        fresh = ~self._initialized
        # 对 fresh 的脚先用当前合力直接初始化；否则保留旧接触状态。copy_ 原地写回。
        self._contact.copy_(torch.where(fresh, force >= contact_on, self._contact))
        # clone 创建独立副本，保存“本步更新之前”的接触标记，用于检测状态切换。
        previous_contact = self._contact.clone()
        # 更新连续达到接触阈值的计时器。
        self._contact_time.copy_(torch.where(
            # 力足够则在原值上加 dt；否则立即把本次连续计时清零。
            force >= contact_on, self._contact_time + dt, 0.0))
        # 更新连续低于离地阈值的计时器。
        self._air_time.copy_(torch.where(
            # 力足够小则累计 dt；否则将连续离地计时清零。
            force <= contact_off, self._air_time + dt, 0.0))
        # 接触计时达到确认时间的位置，写为 True；其余保留当前值。
        self._contact.copy_(torch.where(
            self._contact_time >= debounce_s, True, self._contact))
        # 离地计时达到确认时间的位置，写为 False；其余保留当前值。
        self._contact.copy_(torch.where(
            self._air_time >= debounce_s, False, self._contact))

        # 下面避免把重置后只观察到一半的摆动误判为一个完整周期。
        # Ignore a partial swing immediately after reset or leaving stand mode.
        # 当前摆动、上一帧非摆动、且不是首次初始化，才算刚进入一次新摆动。
        entering_swing = swing & ~self._was_swing & ~fresh
        # 非摆动、新摆动开始、缺少历史或未启用时，清理本次摆动的事件记录。
        clear = ~swing | entering_swing | fresh | ~active
        # &= 是逐项与后原地赋回；只在不需要 clear 的位置保留“曾离地”标记。
        self._had_air &= ~clear
        # 同样清掉该新周期不应继承的“摆动被打断”标记。
        self._broken &= ~clear
        # masked_fill_ 在 clear=True 的位置把离地次数写为 0。
        self._liftoffs.masked_fill_(clear, 0)
        # |= 是逐项或后原地赋回；已观察支撑或有效的新摆动开始，才建立有效周期标记。
        self._valid_cycle |= (~swing | entering_swing) & active
        # 未执行移动命令的脚不能保留有效周期标记。
        self._valid_cycle &= active
        # 必须同时满足摆动、周期有效、移动命令启用，才检查连续性。
        valid = swing & self._valid_cycle & active
        # 再限制到核心进度区间；此处含起点，不含终点。
        core = valid & (progress >= core_start) & (progress < core_end)

        # 即使起脚发生在核心区间之前，也要记住它是本次摆动的第一次离地。
        # Airborne at swing start also counts as the first flight of this swing.
        # 有效摆动中，接触标记为 False 且离地计时已达确认阈值，才算确认离地。
        airborne = valid & ~self._contact & (self._air_time >= debounce_s)
        # 本周期以前没确认过离地、现在已确认，得到首次离地事件。
        first_flight = airborne & ~self._had_air
        # 上一接触状态为 True、本次变 False 时，得到接触→离地的切换事件。
        liftoff = valid & previous_contact & ~self._contact
        # 将首次离地和接触切换合并；同一位置布尔或，不重复算两次。
        new_flight = first_flight | liftoff
        # 核心区间再次发生离地事件，且之前至少已离地一次，才视为重复起脚。
        repeat = core & new_flight & (self._liftoffs >= 1)
        # 将布尔事件转整数再原地累加，每个位置最多在本步加 1。
        self._liftoffs += new_flight.to(torch.long)
        # 核心区间从非接触变接触，且此前已离地，判断为摆动中提前落脚。
        touchdown = core & ~previous_contact & self._contact & self._had_air
        # |= 让“曾被打断”保持为 True，直到周期清理；不只是惩罚触地那一帧。
        self._broken |= touchdown
        # 更新“本次摆动曾确认离地”的历史标记。
        self._had_air |= airborne

        # 以下分为持续误差和单次事件误差两种收费方式。
        # Keep charging for an interrupted swing until its landing allowance.
        # RewardManager multiplies by dt: dividing only the event by dt keeps
        # its integrated cost independent of the policy frequency.
        # 已打断且仍在核心摆动窗口时，持续误差为 1，否则为 0。
        error = (self._broken & core).to(force.dtype)
        # 重复起脚事件加 repeat_event_cost_s/dt；管理器随后乘 dt，抵消事件项的显式步长因子。
        error += repeat.to(force.dtype) * (repeat_event_cost_s / dt)
        # 将当前摆动状态存为下一步的“上一帧状态”；copy_ 修改原张量内容。
        self._was_swing.copy_(swing)
        # 用 active 更新初始化标记；站立环境写 False，下次移动时重新初始化。
        self._initialized.copy_(active)
        # 对左右脚误差取平均，得到 [N]；这一步也会作用于事件项。
        return error.mean(dim=1)
    
# ==================== 7 踝侧摆目标越界误差 ====================
def ankle_roll_target_limit_error(
    # env 是实际环境对象。
    env,
    # asset_cfg 选择需要检查的关节；当前配置选择左右 ankle_roll 两个关节。
    asset_cfg: SceneEntityCfg,
    # 返回 [N] 的非负目标越界误差，不会在函数内裁剪动作。
) -> torch.Tensor:
    """惩罚踝侧摆目标超出软限位的程度，不直接裁剪动作。"""
    # 取得机器人对象。
    robot = env.scene[asset_cfg.name]
    # 读取已由框架按名字解析的关节编号；关节名称与索引不是同一概念。
    joint_ids = asset_cfg.joint_ids

    # 读取 PD 使用的关节位置目标 [N, J]，不是实际关节角度。
    target = robot.data.joint_pos_target[:, joint_ids]
    # 读取每个环境、每个选中关节的软上下限 [N, J, 2]。
    soft_limits = robot.data.soft_joint_pos_limits[:, joint_ids, :]
    # 读取硬上下限 [N, J, 2]，后面用来确定归一化尺度。
    hard_limits = robot.data.joint_pos_limits[:, joint_ids, :]

    # 分别计算低于下界和高于上界的量，区间内两项都为 0。
    excess = (
        # [..., 0] 取所有选中关节的软下界；relu 只保留低于下界的正差值。
        torch.relu(soft_limits[..., 0] - target)
        # [..., 1] 取软上界；relu 只保留超过上界的正差值。
        + torch.relu(target - soft_limits[..., 1])
    )

    # 以硬限位总范围的一半作为每个关节的归一化尺度。
    half_range = (
        # 先计算硬上界减硬下界，再乘 0.5。
        0.5 * (hard_limits[..., 1] - hard_limits[..., 0])
    # clamp_min 保证分母至少为 1e-6，避免除以 0。
    ).clamp_min(1.0e-6)

    # 逐关节计算归一化越界量的平方，再对 J 个关节平均。
    # 目标越界不等于实际关节已经越界；本函数只报告目标误差，是否启用由配置决定。
    return torch.mean(torch.square(excess / half_range), dim=1)

# ==================== 8 支撑脚绕竖直轴转动误差 ====================
def feet_contact_yaw_motion(
    # env 提供机器人和接触传感器。
    env,
    # sensor_cfg 的 body_ids 是接触传感器内部编号。
    sensor_cfg: SceneEntityCfg,
    # asset_cfg 的 body_ids 是机器人内部编号，两套编号不保证相同。
    asset_cfg: SceneEntityCfg,
    # 竖直接触力阈值，默认 15 N；这是瞬时阈值判断，没有连续去抖。
    force_threshold: float = 15.0,
    # 输出 [N]；这是实际刚体转动量的代价，不是某个关节角的误差。
) -> torch.Tensor:
    """平地支撑脚绕世界竖直方向的转动惩罚。"""
    # 取得机器人对象，用于读取两脚角速度。
    robot = env.scene[asset_cfg.name]
    # 取得接触传感器对象，用于判断当前承重的脚。
    sensor = env.scene.sensors[sensor_cfg.name]

    # 接触传感器与机器人使用各自解析后的 body_ids。
    # 读取脚部世界系竖直接触力。
    vertical_force = sensor.data.net_forces_w[
        # 选所有环境、指定传感器脚编号、z 分量，得到 [N, 2]。
        :, sensor_cfg.body_ids, 2
    ]
    # 逐位置比较阈值，得到 True/False 接触表 [N, 2]；不是整个张量只判断一次。
    contact = vertical_force > force_threshold

    # 读取脚刚体世界系角速度，并取竖直轴分量；单位 rad/s。
    yaw_speed = robot.data.body_ang_vel_w[
        # 使用机器人的脚连杆编号取角速度 z 分量，而不是复用传感器编号。
        :, asset_cfg.body_ids, 2
    # abs 取绝对值，让向左转和向右转都产生非负代价；不是四元数偏角的时间差分。
    ].abs()

    # 对左右脚求和，因此不是 mean；同一数值下总量与平均的尺度不同。
    return torch.sum(
        # 接触表转成角速度的浮点类型；触地脚乘 1，未触地脚乘 0。
        yaw_speed * contact.to(yaw_speed.dtype),
        # 沿左右脚维度求和，得到每个环境一个数 [N]。
        # 本项不测量全部接触点的切向滑移，也不在函数中屏蔽站立或转向。
        # 作为惩罚使用时，由配置中的负权重赋予负号。
        dim=1,
    )