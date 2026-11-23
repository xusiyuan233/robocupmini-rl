# 当前参考示范与迁移范围

## 当前训练需要的数据

唯一选择入口为 `data/motions/robotcupmini/style_reference_manifest.json`。

它用SHA-256绑定：
- `retargeted/berkeley_v12_physics/candidate_motion.npz`：420帧，50Hz；
- 同目录 `controller_commands.npz`；
- `validated/berkeley_20261120/v12_full_audit/decision.json`；
- `validated/berkeley_20261120/v12_style_quality.json`。

这些文件保留原字节；证书中记录的原机器绝对路径是历史溯源信息，不用于AMP数据定位。不要为迁移改写证书，否则会破坏哈希绑定。

`v12_full_audit/physics_replay.npz` 和 `generation_report.json` 是此参考的历史物理记录，不是本次重新执行的Sim2Sim，不代表真机安全许可。证书中的 `amp_learner_integrated: false` 等字段是生成证书时的阶段状态；当前AMP接入状态看 `train.py --amp` 与 `tools/amp/runner.py`，不通过修改旧证书更新阶段。

## 来源

示范来自 google-deepmind/mujoco_playground 中 Berkeley humanoid 的已发布策略轨迹，经目标机器人13关节适配与本项目物理/风格检查得到。源仓库提交：`81dfe512c9f2f03107fda1e31de585d04bb30bc4`。

来源文件哈希列表见 `reproducibility/reference_source.json`（仅移除原机器本地存放路径），来源许可证见 `docs/licenses/mujoco_playground-LICENSE.txt`。这不意味着原策略在任意硬件/机器人上都能直接运行，也不表示当前示范等同真人动作。

## 有意不迁移的内容

- 原始LAFAN等动作下载包、Berkeley第三方原始权重与资产下载缓存；
- 早期walk1/berkeley候选、优化中间缓存及未选择的失败草稿；
- 本项目独立的 `robocup_sim2sim` 仓库、MuJoCo部署模型和部署环境；
- 整套日志、视频、备份和Conda目录；
- 其他使用者的项目、代码、环境或模型。

现成的经过哈希绑定的示范足以恢复当前AMP训练；不要求新电脑先重新生成示范。`tools/motion/` 其他研究工具作为源码留档，但它们可能依赖上述未打包数据、MuJoCo和历史实验文件，不属于本次“新电脑直接复现训练”的可运行接口。

后续若要从源策略重新生成全部示范，需要单独整理原始依赖、下载授权、物理模型与命令，并重新验收，而不是修改当前已选定数据。
