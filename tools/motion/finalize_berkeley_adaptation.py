"""Publish an honest Berkeley-adaptation review; select data only on all gates.

An artifact or a rendered walking animation is not acceptance. Failed versions
stay isolated, and the old selected demonstration is not silently replaced.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from datetime import date
from verified_style_data import load_verified_motion

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'data/motions/robotcupmini'
VALIDATED = BASE/'validated/berkeley_20261120'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(version, visual_review=False, created_date="2026-11-20"):
    date.fromisoformat(created_date)
    if not re.fullmatch(r'v[1-9][0-9]*', version):
        raise ValueError('Expected an own-project version such as v6')
    cert = VALIDATED/f'{version}_full_audit/decision.json'
    quality = VALIDATED/f'{version}_style_quality.json'
    video = VALIDATED/f'{version}_review/berkeley_to_robotcupmini_physics.mp4'
    if not video.is_file():
        raise FileNotFoundError('Render the full recorded-motion review first')
    c = json.loads(cert.read_text()); q = json.loads(quality.read_text())
    passed = bool(c['dynamics_certified_in_this_mujoco_contract'] and
                  all(c['checks'].values()) and q['status'] == 'STRAIGHT_STYLE_TARGETS_MET' and
                  all(q['checks'].values()) and q['physics_certificate_sha256'] == digest(cert))
    protected = json.loads((ROOT/'backups/berkeley_adaptation_20261120/protected_hashes.json').read_text())
    changed = [p for p, sha in protected.items() if digest(ROOT/p) != sha]
    allowed = ['tools/motion/optimize_reference_sampling.py']
    if any(p not in allowed for p in changed):
        raise ValueError(f'Unexpected protected model/training changes: {changed}')
    result = {
        'status': 'PASSED_LOCAL_MUJOCO_AND_STYLE_CHECKS' if passed else 'DRAFT_NOT_APPROVED_FOR_STYLE_TRAINING',
        'source': 'Published MuJoCo Playground Berkeley Humanoid policy, pinned v0.0.5 commit',
        'same_as_selected_real_video_checkpoint': False,
        'version': version,
        'updated_date': created_date,
        'comparison_video': str(video.relative_to(ROOT)),
        'certificate': str(cert.relative_to(ROOT)),
        'style_quality_report': str(quality.relative_to(ROOT)),
        'ready_for_style_dataset_selection': passed,
        'selected': False,
        'physics_failed_checks': [k for k, v in c['checks'].items() if not v],
        'style_failed_checks': q['failed_checks'],
        'physics_measured': c['measured'],
        'style_measured': q['measured'],
        'protected_files_changed': changed,
        'training_configs_robot_xml_mass_pd_unchanged': True,
        'training_launched': False,
        'amp_learner_integrated': False,
        'isaaclab_replay_certified': False,
        'real_robot_certified': False,
        'assistant_whole_clip_visual_review': bool(visual_review),
        'assistant_visual_review_method': 'Whole-duration contact sheet sampled every 0.4s, plus full-step numerical screening; not human real-time video approval' if visual_review else None,
        'user_visual_acceptance': False,
    }
    if passed and visual_review:
        motion = BASE/f'retargeted/berkeley_{version}_physics/candidate_motion.npz'
        controls = motion.parent/'controller_commands.npz'
        def entry(p): return dict(path=str(p.relative_to(ROOT)), sha256=digest(p))
        manifest = {
            'status': 'APPROVED_MUJOCO_PHYSICAL_DEMONSTRATION',
            'created_date': created_date, 'source_kind': 'berkeley_published_policy_rollout',
            'motion': entry(motion), 'certificate': entry(cert),
            'style_quality_report': entry(quality), 'control_targets': entry(controls),
            'reference_mean_forward_speed_m_s': q['measured']['mean_forward_speed_m_s'],
            'control_hz': 50, 'physics_hz': 1000, 'nonperiodic': True,
            'amp_learner_integrated': False, 'isaaclab_replay_certified': False,
            'human_visual_review_accepted': False, 'human_equivalence_certified': False,
            'notes': [
                'Same Berkeley robot as chosen candidate, NOT the real-video policy checkpoint.',
                '12 source leg joints adapted to 13 target joints; no synthetic arms/spine motion.',
                'Actual native free-root target physics, no root forces or filtered saved states.',
                'All local project checks passed; not universal gait, IsaacLab, or hardware approval.',
                'Nonperiodic: never use the final-to-first pair.',
                'Demo selection alone does not implement AMP or start training.',
            ],
        }
        selected = VALIDATED/f'{version}_style_reference_manifest.json'
        selected.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        verified = load_verified_motion(selected)
        result['verified_frames'] = len(verified.arrays['time_s'])
        result['verified_adjacent_pairs'] = len(verified.valid_pair_indices)
        active = BASE/'style_reference_manifest.json'
        backup = ROOT/f'backups/berkeley_adaptation_20261120/before_selecting_{version}.json'
        if backup.exists():
            raise FileExistsError('Refusing to replace a previous selection backup')
        backup.write_bytes(active.read_bytes())
        temporary = active.with_suffix('.berkeley.tmp')
        temporary.write_bytes(selected.read_bytes()); temporary.replace(active)
        result['selected'] = True
    out = VALIDATED/'adaptation_result.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    lines = ['# Berkeley Humanoid → ROBOCUPMINI 示范适配结果', '',
             f"状态：**{result['status']}**", '',
             '## 来源与适配方法',
             '- 来源：MuJoCo Playground 公开 Berkeley Humanoid ONNX 策略，固定版本与 SHA256。',
             '- 不是所选实机短视频背后的同一检查点；不声称复刻论文演示。',
             '- 源机器人 12 个腿部关节 → 本机器人 12 个腿部关节＋腰部关节。',
             '- 按腿长、脚底位置、接触相位和关节命名重定向，不按数组下标硬拷贝。',
             '- 适配版采用 1.5 倍时间伸展、0.55 倍步幅，再进行本机原生 PD 下的自由根物理优化。',
             '- 腰部没有直接源自由度，因此以中性腰部为约束；不伪造人类摆臂或脊柱动作。',
             '', '## 验收结果',
             '| 检查 | 结果 |', '|---|---|']
    for key, value in c['checks'].items():
        lines.append(f"| 物理：{key} | {'通过' if value else '未通过'} |")
    for key, value in q['checks'].items():
        lines.append(f"| 风格：{key} | {'通过' if value else '未通过'} |")
    metrics = c['measured']; style_metrics = q['measured']
    lines += ['', '### 实测数值',
              f"- 完整时长：{metrics['duration_s']:.2f} 秒。",
              f"- 实际平均前进速度：{metrics['mean_forward_speed_m_s']:.3f} m/s；不是原策略的 0.4 m/s 指令。",
              f"- 关节速度峰值：{metrics['peak_actual_joint_speed_rad_s']:.3f} rad/s；门槛 5.001 rad/s。",
              f"- 支撑接触点滑动峰值：{metrics['support_contact_point_slip_max_m_s']:.4f} m/s；门槛 0.05 m/s。",
              f"- 脚底穿透最大值：{metrics['maximum_mesh_penetration_m']*1000:.3f} mm；门槛 2 mm。",
              f"- 摆动相内部最小离地：{metrics['minimum_interior_swing_clearance_m']*1000:.3f} mm；门槛 3 mm。",
              f"- 去趋势侧向摆幅：{style_metrics['detrended_pelvis_sway_range_m']*100:.2f} cm；门槛 5.5 cm。",
              f"- 躯干高频抖动：{style_metrics['torso']['highfreq_rms_deg']:.3f}°；门槛 0.2°。"]
    lines += ['', '数值详情见同目录 adaptation_result.json；所有失败版本均保留，未删除失败帧、未放宽门槛。',
              '', '## 本次改动',
              '- 新增公开策略采集、Berkeley 专用重定向、机器人源风格筛查、对照视频和启动支撑目标优化工具。',
              '- 复用原有 IK → 自由根物理优化 → 独立 1kHz 重放的链路，支持机器人源脚底高度。',
              '- 优化器补充前进追踪、初始控制前缀，以及精确脚底凸包、关节速度与支撑滑动的预测峰值约束；旧参数默认值保持兼容。',
              '- 训练配置、奖励代码、机器人 XML、质量惯量、PD 和力矩限制未改动。',
              '- 未运行 PPO/AMP 训练，未涉及其他使用者项目。',
              '', '## 如何查看', '```bash',
              'cd /home/ruiz/robocup/tron2_rl_lab', 'bash tools/motion/play_berkeley_reference.sh', '```',
              '上排是源机器人，下排是本机器人已记录的实际物理状态；不是 IK 预览或新训练的策略。',
              '', '## 训练边界',
              f"- 本次是否替换统一示范选择：{result['selected']}。", 
              '- 没通过检查时保留旧选择，但这不代表旧动作获得用户的视觉认可；不能误把旧回放当成本次 Berkeley 成果。',
              '- 通过本地 MuJoCo 检查也不能保证 IsaacLab 或实机稳定。',
              '- StylePrep 当前仍不是完整 AMP；还需要判别器、专家转换特征、style reward 和 IsaacLab 对齐验证。']
    (VALIDATED/'适配说明.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: result[k] for k in ['status','version','selected','physics_failed_checks','style_failed_checks','comparison_video']}, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--accept-visual-review', action='store_true')
    parser.add_argument('--created-date', default='2026-11-20', help='Explicit project date, YYYY-MM-DD')
    args = parser.parse_args(); finalize(args.version, args.accept_visual_review, args.created_date)
