"""Independent source-relative motion-quality screen, NOT a normal-human certificate.

Read actual recorded states, including failed physical rollouts, without changing
or smoothing them. Low-pass decomposition is DIAGNOSTIC ONLY. The reachable
human reference has no synthetic arm/spine DOFs. Physics has its own audit.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import butter,sosfiltfilt
from scipy.ndimage import gaussian_filter1d
from play_retarget_motion import load_motion
from bvh_motion import anatomical_frame
from reference_contact_quality import intervals

# Project-specific gates fixed BEFORE V15 full physical optimization. Not human norms.
QUALITY_GATES={
 'minimum_duration_s':8.,'maximum_detrended_pelvis_sway_range_m':.10,
 'minimum_foot_width_p05_m':.125,'minimum_median_foot_width_m':.15,
 'maximum_median_foot_width_m':.19,'maximum_foot_width_p95_m':.215,
 'maximum_torso_highfreq_rms_deg':.20,'maximum_torso_peak_rate_deg_s':65.,
 'maximum_torso_acceleration_p95_deg_s2':400.,
 'maximum_torso_yaw_rmse_to_reachable_source_deg':5.,
 'minimum_torso_yaw_correlation_to_source':.55,
 'minimum_torso_roll_correlation_to_source':.50,
 'maximum_swing_side_excursion_p95_m':.025,
 'maximum_source_time_stretch':1.6,
}


def quat_product_z(quat_xyzw, angle):
    r=Rotation.from_quat(quat_xyzw)*Rotation.from_euler('z',angle)
    return r


def rotational_metrics(rot: Rotation,dt:float) -> dict:
    """SO(3) local log about first orientation avoids +/-pi Euler artefacts."""
    rv=(rot[0].inv()*rot).as_rotvec()
    filtered=sosfiltfilt(butter(3,2.,fs=1/dt,output='sos'),rv,axis=0)
    high=rv-filtered
    rate=np.gradient(rv,dt,axis=0);acc=np.gradient(rate,dt,axis=0)
    trim=slice(5,-5) if len(rv)>12 else slice(None)
    return {'highfreq_rms_deg':float(np.rad2deg(np.sqrt(np.mean(high[trim]**2)))),
            'peak_rate_deg_s':float(np.rad2deg(np.linalg.norm(rate,axis=1)[trim]).max()),
            'acceleration_p95_deg_s2':float(np.percentile(np.rad2deg(np.linalg.norm(acc,axis=1)[trim]),95)),
            'rpy_range_deg':np.rad2deg(np.ptp(rot.as_euler('xyz'),axis=0)).tolist()}


def measure_motion(arrays:dict[str,np.ndarray]) -> dict:
    a=arrays;dt=float(a['sample_dt_s']);t=a['time_s'];p=a['root_positions_m']
    q=a['root_quaternions_xyzw'];names=a['joint_names'].tolist()
    j=a['joint_positions_rad'];waist=j[:,names.index('waist_yaw_joint')]
    root=Rotation.from_quat(q);torso=quat_product_z(q,waist)
    refq=Rotation.from_matrix(a['source_pelvis_rotations'])
    # Compare yaw only: no independent spine roll/pitch exists on this robot.
    human=a['human_positions_m'];hn=a['human_joint_names'].tolist()
    at=lambda name:human[:,hn.index(name)]
    ht=Rotation.from_matrix(anatomical_frame(at('LeftShoulder'),at('RightShoulder'),at('Neck')-at('Spine2')))
    hy=np.unwrap(ht.as_euler('xyz')[:,2]);ty=np.unwrap(torso.as_euler('xyz')[:,2])
    # Remove only a constant bind yaw offset; don't phase-align or scale actual motion.
    error=ty-hy;error-=np.mean(error)
    torso_roll=torso.as_euler('xyz')[:,0]; human_roll=ht.as_euler('xyz')[:,0]
    roll_corr=float(np.corrcoef(torso_roll,human_roll)[0,1]) if min(np.std(torso_roll),np.std(human_roll))>1e-8 else 0.
    corr=float(np.corrcoef(hy,ty)[0,1]) if min(np.std(hy),np.std(ty))>1e-8 else 0.
    feet=a['foot_targets_world_m'];separation=root.inv().apply(feet[:,0]-feet[:,1])[:,1]
    y=p[:,1]-np.polyval(np.polyfit(t,p[:,1],1),t)
    side_exc=[];stance_starts=[];step_periods=[]
    for side in (0,1):
        contacts=a['contact_candidates'][:,side]
        starts=[lo for lo,hi in intervals(contacts) if lo>0]
        stance_starts.extend([(float(t[k]),side) for k in starts])
        if len(starts)>1:step_periods.extend(np.diff(t[starts]).tolist())
        for lo,hi in intervals(~contacts):
            if lo==0 or hi==len(t) or hi-lo<4:continue
            segment=feet[lo:hi,side]
            u=(t[lo:hi]-t[lo])/(t[hi-1]-t[lo])
            line=segment[0]+u[:,None]*(segment[-1]-segment[0])
            side_exc.extend(abs(segment[:,1]-line[:,1]).tolist())
    duration=float(t[-1]-t[0]);source_duration=float(a['source_time_s'][-1]-a['source_time_s'][0])
    return {'duration_s':duration,'source_time_stretch':duration/source_duration,
            'detrended_pelvis_sway_range_m':float(np.ptp(y)),
            'pelvis_sway_rms_m':float(np.sqrt(np.mean(y*y))),
            'foot_width_p05_median_p95_m':np.percentile(separation,[5,50,95]).tolist(),
            'foot_width_min_m':float(separation.min()),
            'swing_side_excursion_p95_m':float(np.percentile(side_exc,95)) if side_exc else float('inf'),
            'torso':rotational_metrics(torso,dt),'source_torso':rotational_metrics(ht,dt),
            'torso_yaw_rmse_to_source_deg':float(np.rad2deg(np.sqrt(np.mean(error**2)))),
            'torso_yaw_correlation_to_source':corr, 'torso_roll_correlation_to_source':roll_corr,
            'same_foot_stride_period_median_s':float(np.median(step_periods)) if step_periods else None,
            'foot_contact_starts_s':sorted(stance_starts),
            'knee_median_deg':[float(np.rad2deg(np.median(j[:,names.index(s+'_knee_joint')]))) for s in ('left','right')],
            'waist_range_deg':np.rad2deg([waist.min(),waist.max()]).tolist()}


def from_audit(audit:Path):
    decision=json.loads((audit/'decision.json').read_text())
    source=Path(decision['reference_motion']);a,names,dt=load_motion(source)
    with np.load(audit/'physics_replay.npz',allow_pickle=False) as ar:
        states=ar['states_50hz'];feet=ar['foot_anchor_positions_m'];phase=ar['contact_phase']
    n=len(states);Q=states[:,1:21];V=states[:,21:]
    # Audit's tested model records canonical joint order, but explicitly verify it.
    if names!=decision['tested_model']['joint_names']:raise ValueError('Audit joint order mismatch')
    source_count=len(a['time_s'])
    for k,v in list(a.items()):
        if v.ndim>0 and len(v)==source_count:a[k]=v[:n]
    initial_feet=a['foot_targets_world_m'][0]
    a.update(time_s=states[:,0],root_positions_m=Q[:,:3],root_quaternions_xyzw=Q[:,[4,5,6,3]],
             joint_positions_rad=Q[:,7:],joint_velocities_rad_s=V[:,6:],
             foot_targets_world_m=np.concatenate([initial_feet[None],feet]),
             contact_candidates=np.vstack([phase[:1],phase[round(dt/.001)-1::round(dt/.001)]]))
    return a,decision


def audit_quality(audit:Path,output:Path):
    for p in (audit,output):
        if 'lilin' in str(p.resolve()).lower():raise ValueError('Outside user project')
    if output.exists():raise FileExistsError(output)
    a,decision=from_audit(audit);m=measure_motion(a);g=QUALITY_GATES;t=m['torso'];w=m['foot_width_p05_median_p95_m']
    checks={'full_duration':m['duration_s']>=g['minimum_duration_s'],
      'lateral_sway':m['detrended_pelvis_sway_range_m']<=g['maximum_detrended_pelvis_sway_range_m'],
      'foot_spacing':w[0]>=g['minimum_foot_width_p05_m'] and g['minimum_median_foot_width_m']<=w[1]<=g['maximum_median_foot_width_m'] and w[2]<=g['maximum_foot_width_p95_m'],
      'torso_high_frequency_jitter':t['highfreq_rms_deg']<=g['maximum_torso_highfreq_rms_deg'],
      'torso_rate':t['peak_rate_deg_s']<=g['maximum_torso_peak_rate_deg_s'],
      'torso_acceleration':t['acceleration_p95_deg_s2']<=g['maximum_torso_acceleration_p95_deg_s2'],
      'source_torso_yaw':m['torso_yaw_rmse_to_source_deg']<=g['maximum_torso_yaw_rmse_to_reachable_source_deg'] and m['torso_yaw_correlation_to_source']>=g['minimum_torso_yaw_correlation_to_source'],
      'source_torso_roll_phase':m['torso_roll_correlation_to_source']>=g['minimum_torso_roll_correlation_to_source'],
      'no_large_sideways_swing':m['swing_side_excursion_p95_m']<=g['maximum_swing_side_excursion_p95_m'],
      'not_excessively_slowed':m['source_time_stretch']<=g['maximum_source_time_stretch']}
    report={'status':'PROJECT_COORDINATION_CHECKS_MET' if all(checks.values()) else 'COORDINATION_CHECKS_FAILED',
            'audit':str(audit.resolve()),'physics_passed':decision['dynamics_certified_in_this_mujoco_contract'],
            'physics_certificate_sha256':hashlib.sha256((audit/'decision.json').read_bytes()).hexdigest(),
            'physics_recording_sha256':hashlib.sha256((audit/'physics_replay.npz').read_bytes()).hexdigest(),
            'measured':m,'thresholds':g,'checks':checks,'failed_checks':[k for k,v in checks.items() if not v],
            'human_equivalence_certified':False,'visual_review_performed_by_this_script':False,
            'notes':['Engineering screen based on stated problems, NOT a universal normal-human gait test.',
                     'No filtering of saved states; 2Hz decomposition only measures jitter.',
                     'Source human has arm/spine/toe motion unavailable to this 13-joint robot.',
                     'Source contact labels are inferred, not force ground truth.']}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit_quality(a.audit,a.output)
if __name__=='__main__':main()
