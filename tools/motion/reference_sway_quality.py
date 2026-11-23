"""Lateral-motion planning and comparison, separate from dynamics certification.

World Y diagnostics assume this clip walks primarily along world X. They are
not a universal naturalness score and never constitute a training certificate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


def walking_centerline(lateral_com: np.ndarray, dt: float, smoothing_s: float) -> np.ndarray:
    """Remove step-frequency oscillation without forcing a drifting path to Y=0."""
    y = np.asarray(lateral_com, dtype=float)
    if y.ndim != 1 or len(y) < 3 or not np.isfinite(y).all():
        raise ValueError('Finite one-dimensional lateral track with >=3 frames required')
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(smoothing_s) or smoothing_s <= 0:
        raise ValueError('Positive finite time constants required')
    t = np.arange(len(y)) * dt
    trend = np.polyval(np.polyfit(t, y, 1), t)
    return trend + gaussian_filter1d(y - trend, smoothing_s / dt, mode='nearest')


def sway_metrics(motion: Path) -> dict:
    with np.load(motion, allow_pickle=False) as a:
        t = a['time_s']; p = a['root_positions_m']
        q = a['root_quaternions_xyzw']; dt = float(a['sample_dt_s'])
        center = walking_centerline(p[:, 1], dt, 1.)
        osc = p[:, 1] - center
        straight = p[:, 1] - np.polyval(np.polyfit(t, p[:, 1], 1), t)
        rpy = Rotation.from_quat(q).as_euler('xyz', degrees=True)
        feet = a['foot_targets_world_m']; delta = feet[:, 0] - feet[:, 1]
        yaw = np.deg2rad(rpy[:, 2])
        width = -np.sin(yaw) * delta[:, 0] + np.cos(yaw) * delta[:, 1]
        result = {'path': str(motion.resolve()), 'sha256': hashlib.sha256(motion.read_bytes()).hexdigest(),
                  'duration_s': float(t[-1]-t[0]),
                  'root_lateral_range_m': float(np.ptp(p[:, 1])),
                  'root_lateral_detrended_range_m': float(np.ptp(straight)),
                  'root_lateral_oscillation_range_m': float(np.ptp(osc)),
                  'root_lateral_oscillation_rms_m': float(np.sqrt(np.mean(osc**2))),
                  'root_roll_minmax_deg': [float(rpy[:, 0].min()), float(rpy[:, 0].max())],
                  'foot_separation_min_median_max_m': [float(width.min()), float(np.median(width)), float(width.max())],
                  'mean_forward_speed_m_s': float((p[-1, 0]-p[0, 0])/(t[-1]-t[0])),
                  'peak_joint_speed_rad_s': float(abs(a['joint_velocities_rad_s']).max())}
        if 'reference_com_world_m' in a:
            result['com_lateral_range_m'] = float(np.ptp(a['reference_com_world_m'][:, 1]))
        return result


def compare(baseline: Path, candidate: Path) -> dict:
    old, new = sway_metrics(baseline), sway_metrics(candidate)
    ratio = new['root_lateral_detrended_range_m']/max(old['root_lateral_detrended_range_m'], 1e-12)
    # Project-specific improvement target declared before physics optimization.
    # Not claimed as a human naturalness standard. Dynamics must pass separately.
    checks = {'at_least_35_percent_less_lateral_range': ratio <= .65,
              'detrended_lateral_range_at_most_10cm': new['root_lateral_detrended_range_m'] <= .10,
              'no_crossed_feet': new['foot_separation_min_median_max_m'][0] >= .03,
              'reference_joint_speed_at_most_5rad_s': new['peak_joint_speed_rad_s'] <= 5.001,
              'at_least_8s': new['duration_s'] >= 8.}
    return {'status': 'LATERAL_IMPROVEMENT_TARGET_MET' if all(checks.values()) else 'LATERAL_TARGET_NOT_MET',
            'baseline': old, 'candidate': new, 'lateral_range_ratio': ratio,
            'checks': checks, 'dynamics_approved_by_this_report': False,
            'note': 'Straight walking clip only; relative improvement criteria, not proof of naturalness, dynamics or AMP readiness.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True); p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    for path in (a.baseline, a.candidate, a.output):
        if 'lilin' in str(path.resolve()).lower(): raise ValueError('Outside user project scope')
    if a.output.exists(): raise FileExistsError('Refusing report overwrite')
    result = compare(a.baseline, a.candidate)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
