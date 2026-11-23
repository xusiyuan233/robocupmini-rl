"""Independent style screen for robot-policy sources, NOT a human-gait standard.
All thresholds are fixed here before selecting a Berkeley candidate. States are
not filtered or edited; filtering inside rotational_metrics is diagnostic only.
Does not approve failed physics, implement AMP, or certify real-world safety.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from reference_coordination_quality import from_audit,rotational_metrics
from reference_contact_quality import intervals
ROOT=Path(__file__).resolve().parents[2]
GATES={'minimum_duration_s':8.,'maximum_lateral_sway_m':.055,'maximum_lateral_drift_m_s':.003,'minimum_width_p05_m':.125,'minimum_width_median_m':.15,'maximum_width_median_m':.19,'maximum_width_p95_m':.215,'maximum_torso_highfreq_deg':.2,'maximum_torso_rate_deg_s':65.,'maximum_torso_accel_p95_deg_s2':400.,'maximum_swing_side_excursion_p95_m':.025,'maximum_heading_range_deg':10.,'minimum_forward_speed_m_s':.12,'maximum_forward_speed_m_s':.3,'maximum_source_torso_deviation_rmse_deg':5.}

def audit(folder,output):
    for path in (folder,output):
        if not path.resolve().is_relative_to(ROOT) or 'lilin' in str(path.resolve()).lower():raise ValueError('Outside user project')
    if output.exists():raise FileExistsError(output)
    a,decision=from_audit(folder)
    if str(a.get('source_kind',''))!='berkeley_published_policy_rollout':raise ValueError('Not a Berkeley source')
    dt=float(a['sample_dt_s']);t=a['time_s'];p=a['root_positions_m'];root=Rotation.from_quat(a['root_quaternions_xyzw'])
    waist=a['joint_positions_rad'][:,a['joint_names'].tolist().index('waist_yaw_joint')]
    torso=root*Rotation.from_euler('z',waist);rot=rotational_metrics(torso,dt)
    trend=np.polyfit(t,p[:,1],1);sway=float(np.ptp(p[:,1]-np.polyval(trend,t)))
    feet=a['foot_targets_world_m'];width=root.inv().apply(feet[:,0]-feet[:,1])[:,1];wp=np.percentile(width,[5,50,95])
    excursions=[];periods=[]
    for side in range(2):
        contact=a['contact_candidates'][:,side];starts=[lo for lo,hi in intervals(contact) if lo>0]
        periods.extend(np.diff(t[starts]))
        for lo,hi in intervals(~contact):
            if lo==0 or hi==len(t) or hi-lo<4:continue
            y=feet[lo:hi,side,1];line=np.linspace(y[0],y[-1],len(y));excursions.extend(abs(y-line))
    side95=float(np.percentile(excursions,95)) if excursions else float('inf')
    source=Rotation.from_matrix(a['source_robot_torso_rotations'][:len(t)])
    # Constant bind offset only. No amplitude or phase fitting to actual states.
    source_rel=source.mean().inv()*source;torso_rel=torso.mean().inv()*torso
    source_rmse=float(np.degrees(np.sqrt(np.mean((source_rel.inv()*torso_rel).magnitude()**2))))
    speed=float((p[-1,0]-p[0,0])/(t[-1]-t[0]));heading=float(np.degrees(np.ptp(np.unwrap(root.as_euler('xyz')[:,2]))))
    g=GATES
    checks={'physics':bool(decision['dynamics_certified_in_this_mujoco_contract']),'full_duration':t[-1]>=g['minimum_duration_s'],'lateral_sway':sway<=g['maximum_lateral_sway_m'],'lateral_drift':abs(trend[0])<=g['maximum_lateral_drift_m_s'],'foot_width':bool(wp[0]>=g['minimum_width_p05_m'] and g['minimum_width_median_m']<=wp[1]<=g['maximum_width_median_m'] and wp[2]<=g['maximum_width_p95_m']),'torso_jitter':rot['highfreq_rms_deg']<=g['maximum_torso_highfreq_deg'],'torso_rate':rot['peak_rate_deg_s']<=g['maximum_torso_rate_deg_s'],'torso_acceleration':rot['acceleration_p95_deg_s2']<=g['maximum_torso_accel_p95_deg_s2'],'no_sideways_swing':side95<=g['maximum_swing_side_excursion_p95_m'],'heading':heading<=g['maximum_heading_range_deg'],'speed':g['minimum_forward_speed_m_s']<=speed<=g['maximum_forward_speed_m_s'],'source_torso':source_rmse<=g['maximum_source_torso_deviation_rmse_deg']}
    checks={k:bool(v) for k,v in checks.items()}
    r={'status':'STRAIGHT_STYLE_TARGETS_MET' if all(checks.values()) else 'STYLE_SCREEN_FAILED','source_kind':'berkeley_published_policy_rollout','criteria_version':'20261120_robot_source_v1','thresholds':g,'measured':{'duration_s':float(t[-1]),'detrended_pelvis_sway_range_m':sway,'lateral_drift_m_s':float(trend[0]),'width_p05_median_p95_m':wp.tolist(),'torso':rot,'swing_side_excursion_p95_m':side95,'mean_forward_speed_m_s':speed,'heading_range_deg':heading,'source_torso_rmse_deg':source_rmse,'same_foot_period_median_s':float(np.median(periods)) if periods else None,'knee_median_deg':np.degrees(np.median(a['joint_positions_rad'][:,[3,9]],axis=0)).tolist()},'checks':checks,'failed_checks':[k for k,v in checks.items() if not v],'physics_passed':bool(decision['dynamics_certified_in_this_mujoco_contract']),'physics_certificate_sha256':hashlib.sha256((folder/'decision.json').read_bytes()).hexdigest(),'human_equivalence_certified':False,'isaaclab_physical_replay_certified':False,'amp_learner_integrated':False,'user_visual_acceptance':False}
    output.parent.mkdir(exist_ok=True,parents=True);output.write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False,indent=2));return r
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();audit(args.audit,args.output)
