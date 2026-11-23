"""Stricter straight-gait screen over actual independently replayed states.

No trajectory filtering/export and no dynamics waiver. Thresholds are engineering
acceptance targets for this fix, NOT human gait norms or real-robot safety proof.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from play_retarget_motion import load_motion, PROJECT_ROOT
from reference_coordination_quality import from_audit, measure_motion

TARGETS = dict(maximum_sway_peak_to_peak_m=.055, maximum_sway_ratio_to_v16=.60,
               maximum_lateral_trend_speed_m_s=.003, minimum_duration_s=8.,
               minimum_foot_width_p05_m=.125, minimum_foot_width_median_m=.15,
               maximum_foot_width_median_m=.19, maximum_foot_width_p95_m=.215,
               maximum_torso_jitter_deg=.20, maximum_torso_rate_deg_s=65.,
               maximum_torso_acceleration_p95_deg_s2=400., maximum_swing_side_excursion_p95_m=.025,
               minimum_source_yaw_correlation=.55, minimum_source_roll_correlation=.5, maximum_source_yaw_rmse_deg=5.)


def judge(m:dict, old:dict, lateral_trend:float, physics_ok:bool) -> dict[str,bool]:
    g=TARGETS;w=m['foot_width_p05_median_p95_m'];t=m['torso']
    return dict(physics=physics_ok, full_duration=m['duration_s']>=g['minimum_duration_s'],
                lateral_sway=m['detrended_pelvis_sway_range_m']<=g['maximum_sway_peak_to_peak_m'],
                relative_sway=m['detrended_pelvis_sway_range_m']<=g['maximum_sway_ratio_to_v16']*old['detrended_pelvis_sway_range_m'],
                no_lateral_path_drift=abs(lateral_trend)<=g['maximum_lateral_trend_speed_m_s'],
                foot_spacing=w[0]>=g['minimum_foot_width_p05_m'] and g['minimum_foot_width_median_m']<=w[1]<=g['maximum_foot_width_median_m'] and w[2]<=g['maximum_foot_width_p95_m'],
                torso_jitter=t['highfreq_rms_deg']<=g['maximum_torso_jitter_deg'],
                torso_rate=t['peak_rate_deg_s']<=g['maximum_torso_rate_deg_s'],
                torso_acceleration=t['acceleration_p95_deg_s2']<=g['maximum_torso_acceleration_p95_deg_s2'],
                source_torso_yaw=m['torso_yaw_correlation_to_source']>=g['minimum_source_yaw_correlation'] and m['torso_yaw_rmse_to_source_deg']<=g['maximum_source_yaw_rmse_deg'],
                source_torso_roll=m['torso_roll_correlation_to_source']>=g['minimum_source_roll_correlation'],
                no_sideways_swing=m['swing_side_excursion_p95_m']<=g['maximum_swing_side_excursion_p95_m'])


def audit(audit_dir:Path, baseline:Path, output:Path):
    for p in (audit_dir,baseline,output):
        if not p.resolve().is_relative_to(PROJECT_ROOT) or 'lilin' in str(p.resolve()).lower():
            raise ValueError('Outside user project')
    if output.exists():raise FileExistsError(output)
    a,decision=from_audit(audit_dir);old=measure_motion(load_motion(baseline)[0]);m=measure_motion(a)
    trend=float(np.polyfit(a['time_s'],a['root_positions_m'][:,1],1)[0])
    physics=bool(decision['dynamics_certified_in_this_mujoco_contract'] and all(decision['checks'].values()))
    checks=judge(m,old,trend,physics)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=dict(status='STRAIGHT_STYLE_TARGETS_MET' if all(checks.values()) else 'STRAIGHT_STYLE_TARGETS_FAILED',
           criteria_version='20261120_straight_v2_source_phase',measured=m,baseline=old,lateral_trend_speed_m_s=trend,thresholds=TARGETS,checks=checks,
           failed_checks=[k for k,v in checks.items() if not v],physics_passed=physics,
           physics_certificate_sha256=sha(audit_dir/'decision.json'),
           physics_recording_sha256=sha(audit_dir/'physics_replay.npz'),
           baseline_sha256=sha(baseline),human_equivalence_certified=False,
           isaaclab_physical_replay_certified=False,amp_learner_integrated=False)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(r,ensure_ascii=False,indent=2))
    print(json.dumps(r,ensure_ascii=False,indent=2));return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.audit,a.baseline,a.output)
if __name__=='__main__':main()
