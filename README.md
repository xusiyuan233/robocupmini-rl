# RobotcupMini：IsaacLab PPO / AMP 风格训练

本仓库是基于 [limxdynamics/tron2_rl_lab](https://github.com/limxdynamics/tron2_rl_lab) 的 **RobotcupMini 13 关节机器人训练项目**，保存机器人接入、站立、速度任务及 AMP 风格奖励训练的当前可迁移快照。

> **范围与状态**
> - 本次迁移只包含 IsaacLab / PPO / AMP；**不包含本项目的 MuJoCo Sim2Sim 仓库、部署程序或实机驱动**。
> - 当前已有可运行策略，但步态、横向摆动和方向保持仍在优化，**不是实机部署验收通过版本**。
> - 代码在 Git 仓库；训练检查点在配套 Release 中。**只下载代码不等于恢复训练成果。**
> - 机器人资产和参考数据包含第三方来源，默认按私有研究项目保存；未确认权限前不要改成公开分发。

## 1. 项目文件导航

| 位置 | 用途 |
|---|---|
| `exts/bipedal_locomotion/.../assets/config/robotcupmini_cfg.py` | 机器人 USD、13 关节顺序、默认角度、PD / 执行器配置 |
| `exts/bipedal_locomotion/.../assets/usd/ROBOCUPMINI/` | 当前使用的 URDF、STL 网格及 USD 层 |
| `exts/bipedal_locomotion/.../tasks/locomotion/robots/robotcupmini_env_cfg.py` | 站立环境 |
| `exts/bipedal_locomotion/.../tasks/locomotion/robots/robotcupmini_velocity_env_cfg.py` | 速度任务、动作与奖励配置 |
| `exts/bipedal_locomotion/.../tasks/locomotion/robots/robotcupmini_style_env_cfg.py` | 固定 0.13 m/s 风格任务，步宽及根高目标覆盖 |
| `exts/bipedal_locomotion/.../tasks/locomotion/mdp/robotcupmini_rewards.py` | 自定义奖励函数与有状态奖励类 |
| `exts/bipedal_locomotion/.../tasks/locomotion/agents/robotcupmini_ppo_cfg.py` | PPO 超参数 |
| `exts/bipedal_locomotion/.../tasks/locomotion/robots/__init__.py` | Gym 任务名称注册 |
| `rsl_rl/` | 本项目携带的 PPO / encoder 实现；不要随意替换成其他版本 |
| `tools/amp/` | AMP 特征、判别器、连续帧缓存、Runner 及测试 |
| `tools/motion/verified_style_data.py` | 校验选定示范和证书 SHA-256；AMP 必需 |
| `data/motions/robotcupmini/style_reference_manifest.json` | 指定当前示范及其哈希绑定 |
| `reproducibility/` | 依赖锁、IsaacLab 提交与补丁、文件校验、成果清单 |
| `tools/repro/` | 安装、完整性检查、回放、续训辅助脚本 |
| `docs/upstream/` | 原官方 README 和原 CI，保留来源说明 |

表格中的 `...` 表示 `bipedal_locomotion` 包的嵌套路径，实际完整路径可直接在 VS Code 搜索文件名。

## 2. 运行环境基线

| 组件 | 记录版本 |
|---|---|
| 系统 | Ubuntu 22.04.5 LTS / x86_64 |
| Python | 3.11.16 |
| Isaac Sim | 5.1.0.0 |
| IsaacLab 源码 | `5c2ec81cb17532d32f7922dd7fcaae40d123b71a`，`VERSION=2.3.1`，加本仓库补丁 |
| torch / torchvision / torchaudio | `2.7.0+cu128` / `0.22.0+cu128` / `2.7.0+cu128` |
| NumPy / tensordict | `1.26.0` / `0.8.3` |
| 原验证 GPU | RTX 4070 Ti SUPER，16 GB；驱动 580.82.07 |

完整运行依赖记录在 `reproducibility/requirements-runtime.lock`（238 个非 editable 包）。其他已安装包的版本备查表为 `package_versions.json`；后者**不是需要全部安装的清单**。

`nvidia-smi` 中的 CUDA 数字是驱动能力信息，不等于必须另装同版本 CUDA Toolkit。不要为迁移随意升级 torch、tensordict、packaging 或 IsaacLab。

不同 GPU / 驱动 / PhysX 执行可能产生不同的训练轨迹；本仓库旨在恢复同一代码、数据与运行设置，不承诺训练逐位一致。4096 环境是当前续训设置，不代表所有电脑都能承载。

## 3. 新电脑安装

### 3.1 下载代码、创建独立环境

先安装 Git、Miniconda/Conda 和适合 NVIDIA GPU 的驱动。私有仓库需先通过自己的 GitHub 账号授权。本仓库地址如下：

```bash
mkdir -p ~/robocup
cd ~/robocup
git clone https://github.com/xusiyuan233/robocupmini-rl.git tron2_rl_lab
cd tron2_rl_lab
conda env create -f reproducibility/environment.yml
conda activate env_isaaclab
```

### 3.2 安装锁定依赖和项目

```bash
bash tools/repro/bootstrap.sh
```

默认将 IsaacLab 放到 `~/robocup/IsaacLab`。如需独立位置：

```bash
ISAACLAB_DIR="$HOME/robocup/IsaacLab_robotcupmini" bash tools/repro/bootstrap.sh
```

脚本会固定 IsaacLab 提交、应用两处本地差异、安装锁定包，并以 editable 方式安装本地源码。发现已有 IsaacLab 提交不匹配时会停止，**不会 reset 或覆盖已有项目**。脚本不会启动训练，也不会替用户接受 NVIDIA 许可。

安装需下载较大的 Isaac Sim / PyTorch 包，并访问 GitHub、NVIDIA PyPI 和 PyTorch 下载站。首次打开仿真器可能需要确认许可、下载资产或编译着色器。若超时，保留报错排查网络，不要用无约束的包升级来碰运气。

> 依赖清单来自工作机的可运行环境；尚不能把源码机器上的检查当作另一台电脑已安装验证。新电脑仍需完成以下 CPU / GPU 验收。

## 4. 恢复训练成果

下载本仓库 Release 中的：

- `robotcupmini-training-artifacts-20261123.tar.gz`
- `robotcupmini-training-artifacts-20261123.tar.gz.sha256`

放到仓库根目录，再执行：

```bash
sha256sum -c robotcupmini-training-artifacts-20261123.tar.gz.sha256
tar -xzf robotcupmini-training-artifacts-20261123.tar.gz
```

解压后，已选定续训基线在：

```text
artifacts/checkpoints/amp_align_v2/model_150.pt
```

该文件不是零起点：它保留策略、encoder、PPO 优化器、判别器、判别器优化器、迭代号和示范哈希。只加载本项目可信来源的 `.pt`；它不是适合随意打开的普通数据文件。

成果包中的 `artifact_manifest.json` 记录每个模型原始实验名、轮数、文件校验值和用途。若包含新续训检查点，只说明已成功保存，**不自动代表回放效果已通过验收**。本仓库不打包全部历史日志、所有中间模型和视频。

## 5. 先检查，再回放，再续训

### 5.1 CPU 检查（不启动机器人训练）

在仓库根目录、激活环境后执行：

```bash
python -m pip check
python -m tools.repro.verify_snapshot
python -m tools.amp.motion_loader
python -m tools.amp.check_transition_buffer
python -m tools.amp.check_discriminator
python -m tools.repro.verify_snapshot \
  --checkpoint artifacts/checkpoints/amp_align_v2/model_150.pt
```

示范应读出 `(420, 42)` 单帧特征、`(419, 84)` 连续帧对；不连接最后一帧到第一帧。判别器测试只更新临时测试判别器，不训练机器人策略，不写正式检查点。

### 5.2 GPU 特征一致性检查

```bash
python -m tools.amp.check_isaac_features --headless --device cuda:0
```

这一步启动 IsaacLab 并检查在线特征与示范口径，**不是 PPO 训练**。首次启动需更久；不要在共享 GPU 上停止别人的进程。

### 5.3 回放选定基线

```bash
bash tools/repro/play_amp.sh
```

等价的直接命令：

```bash
python scripts/rsl_rl/play.py \
  --task=Robocup-Style-Prep-Play-v0 \
  --device=cuda:0 --num_envs=1 \
  --checkpoint_path=artifacts/checkpoints/amp_align_v2/model_150.pt
```

回放不需要 `--amp`。当前 `play.py` 同时会在检查点旁的 `exported/` 写出部署用的 `encoder.onnx`、`policy.onnx` 和 TorchScript 导出文件。导出不是继续训练，也不是 Sim2Sim 验收。

### 5.4 从150轮继续训练：追加2000轮，4096环境，每500轮保存

**只有准备正式训练时才执行：**

```bash
bash tools/repro/resume_amp.sh
```

等价命令：

```bash
python scripts/rsl_rl/train.py \
  --task=Robocup-Style-Prep-v0 \
  --amp --amp_weight=1.0 --resume=True \
  --checkpoint_path=artifacts/checkpoints/amp_align_v2/model_150.pt \
  --device=cuda:0 --num_envs=4096 \
  --max_iterations=2000 --save_interval=500 \
  --logger=tensorboard \
  --run_name=amp_align_v2_from150_2000 --headless
```

- 当前 AMP Runner 的 `max_iterations=2000` 表示**追加2000轮**；最终累计2150轮。
- 保存判断相对于本次起始轮数，得到 `model_650.pt`、`model_1150.pt`、`model_1650.pt`、`model_2150.pt`。
- 使用新时间戳日志目录，不覆盖基线。
- 若显存不足，可以 `NUM_ENVS=1024 bash tools/repro/resume_amp.sh`；这改变采样批量，不再是4096环境的同一训练设置。
- 检查点会校验风格权重、控制步长和示范哈希。不要改过示范后强行续训。

查看日志：

```bash
tensorboard --logdir logs/rsl_rl/robotcupmini_style_prep --host 127.0.0.1 --port 6006
```

浏览器打开 `http://127.0.0.1:6006`。日志保存在新电脑本地，不会自动上传 GitHub。

## 6. 任务与控制接口

| 训练任务 | 回放任务 | 内容 |
|---|---|---|
| `Robocup-Stand-v0` | `Robocup-Stand-Play-v0` | 站立 |
| `Robocup-Vel-v0` | `Robocup-Vel-Play-v0` | 速度指令行走 |
| `Robocup-Style-Prep-v0` | `Robocup-Style-Prep-Play-v0` | 风格任务，只有训练时加 `--amp` 才启用 AMP |

根连杆为 `pelvis`。动作13维，显式顺序为：

```text
0  left_hip_pitch_joint      6  right_hip_pitch_joint
1  left_hip_roll_joint       7  right_hip_roll_joint
2  left_hip_yaw_joint        8  right_hip_yaw_joint
3  left_knee_joint           9  right_knee_joint
4  left_ankle_pitch_joint   10  right_ankle_pitch_joint
5  left_ankle_roll_joint    11  right_ankle_roll_joint
12 waist_yaw_joint
```

当前动作目标为默认关节角度加 `0.15 × action`；具体限幅/处理应以环境和 Action 配置为准。IsaacLab 内部关节排列不应被默认当成策略顺序。当前控制周期0.02s（50Hz），物理步长0.005s（200Hz）。

AMP 特征单帧42维（13关节角 + 13关节速度 + 3根线速度 + 3根角速度 + 3重力投影 + 6脚底相对位置 + 1根高），相邻帧84维。示范四元数为 `xyzw`，IsaacLab接口为 `wxyz`，特征桥接代码处理相应口径。

## 7. 当前进展与已知局限

- 完成 URDF / USD 接入、站立和速度任务、AMP 特征与判别器接入。
- 已选择第二轮 AMP `model_150.pt` 作为风格续训折中基线；它不是所有指标最优。
- 之前同布局60秒、4个初态测试中该版本4/4完成；仍有速度超调、横滚、偏航、直腿及脚朝向问题。这不是概率安全保证。
- 新增训练轮数不保证步态必然改善；需回放比较中间检查点，不能默认最后一份最好。
- 旧检查点对应的历史训练源代码并未逐次提交；当前快照不能被表述为所有旧实验的精确历史源码。
- `tools/motion/` 中保留早期处理工具供研究追溯；除当前示范校验桥外，部分旧工具需要未打包的原始动作、MuJoCo环境或历史草稿。**它们不是当前训练的前置步骤。**详见 [参考数据说明](docs/REFERENCE_DATA.md)。
- 原官方任务与图文保留用于来源追溯，不代表本次逐一验证。当前复现验收以本项目 `Robocup-*` 任务为准。
- `reproducibility/verification_report.json` 记录本次实际检查范围；没有执行的检查不应被视为已通过。

## 8. 常见问题

- **`No module named isaaclab`**：确认环境正确，重新按安装步骤安装固定源码，不是换一个 `cd` 目录就能解决。
- **pip依赖冲突**：核对锁文件，不直接升级 torch / tensordict；本项目使用携带的 `rsl_rl`，不要替换为 IsaacLab 可选的其他 RSL-RL 算法版本。
- **找不到检查点**：下载配套 Release 并解压；`.gitignore` 有意排除了 `.pt/.onnx` 和日志。
- **AMP恢复提示示范不一致**：先跑完整性检查，恢复原示范，不修改证书和哈希来绕过检查。
- **首次窗口停留很久**：可能在编译着色器或下载资源，结合日志检查；如出现“已杀死”，需检查显存/内存，不直接归因策略。
- **默认账号/路径不同**：示例用 `$HOME` 和仓库相对路径，不要求用户名为 `ruiz`。模型转换配置与旧报告里的原机器路径属于历史记录，不是新机器依赖。

## 9. 来源与权限

保留原 `LICENSE`、`NOTICE`、`THIRD_PARTY_NOTICES.md`，原说明见 `docs/upstream/`。参考示范来自 Berkeley humanoid 已发布策略轨迹的适配；来源提交与哈希见 `reproducibility/reference_source.json`，对应来源许可证副本见 `docs/licenses/`。

原仓库代码许可不应被理解为自动获得机器人CAD/网格、动作源数据或训练策略的额外公开分发权限。不要上传账号令牌、私钥、环境密码，或其他使用者的代码和模型。
