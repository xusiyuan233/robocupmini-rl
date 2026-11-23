"""NumPy-only, fail-closed bridge from audited demonstrations to future AMP.

No MuJoCo or IsaacLab import is needed. A boolean in an arbitrary NPZ is not
approval: the selected manifest must bind both the exported data and independent
physics certificate by SHA-256. This module does NOT implement a discriminator.
"""
from __future__ import annotations
import argparse,json,hashlib
from dataclasses import dataclass
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]


def file_sha256(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class VerifiedStyleMotion:
    """One nonperiodic clip; adjacent pairs exclude the final->first transition."""
    arrays:dict[str,np.ndarray]
    joint_names:tuple[str,...]
    dt_s:float
    certificate:dict

    @property
    def valid_pair_indices(self)->np.ndarray:
        return np.arange(len(self.arrays['time_s'])-1,dtype=np.int64)


def load_verified_motion(manifest_path:Path,expected_joint_names:list[str]|None=None)->VerifiedStyleMotion:
    """Refuse partial/failed/changed motion and inconsistent joint/time mapping."""
    if 'lilin' in str(manifest_path.resolve()).lower():raise ValueError('Another user project is outside scope')
    manifest=json.loads(manifest_path.read_text())
    if manifest['status']!='APPROVED_MUJOCO_PHYSICAL_DEMONSTRATION':raise ValueError('No approved demonstration selected')
    paths=[]
    for key in ['motion','certificate']:
        path=(ROOT/manifest[key]['path']).resolve()
        if not path.is_relative_to(ROOT) or 'lilin' in str(path).lower():raise ValueError('Manifest points outside user project')
        if file_sha256(path)!=manifest[key]['sha256']:raise ValueError(f'{key} changed after approval')
        paths.append(path)
    motion_path,certificate_path=paths
    if 'style_quality_report' in manifest:
        entry=manifest['style_quality_report'];path=(ROOT/entry['path']).resolve()
        if not path.is_relative_to(ROOT) or 'lilin' in str(path).lower():
            raise ValueError('Quality report outside user project')
        if file_sha256(path)!=entry['sha256']:raise ValueError('Style quality report changed')
        quality=json.loads(path.read_text())
        if (quality['status']!='STRAIGHT_STYLE_TARGETS_MET' or not all(quality['checks'].values())
                or not quality['physics_passed']
                or quality['physics_certificate_sha256']!=file_sha256(certificate_path)):
            raise ValueError('Straight walking quality/physics binding failed')
    certificate=json.loads(certificate_path.read_text())
    if not certificate['dynamics_certified_in_this_mujoco_contract'] or not all(certificate['checks'].values()):
        raise ValueError('Physics certificate failed')
    with np.load(motion_path,allow_pickle=False) as ar:arrays={key:ar[key].copy() for key in ar.files}
    names=tuple(arrays['joint_names'].tolist());n=len(arrays['time_s']);dt=float(arrays['sample_dt_s'].item())
    if len(names)!=13 or len(set(names))!=13 or (expected_joint_names is not None and list(names)!=expected_joint_names):
        raise ValueError('Joint order mismatch')
    expected={'joint_positions_rad':(n,13),'joint_velocities_rad_s':(n,13),'root_positions_m':(n,3),'root_quaternions_xyzw':(n,4),'root_linear_velocities_world_m_s':(n,3),'root_angular_velocities_world_rad_s':(n,3),'foot_targets_world_m':(n,2,3)}
    for key,shape in expected.items():
        if arrays[key].shape!=shape or not np.isfinite(arrays[key]).all():raise ValueError(f'Invalid state data: {key}')
    if n<3 or abs(dt-.02)>1e-10 or not np.allclose(np.diff(arrays['time_s']),dt,atol=1e-10):raise ValueError('Time/control frequency mismatch')
    if bool(arrays['periodic_reference'].item()):raise ValueError('This dataset has no certified cyclic seam')
    for flag in ['valid_for_training','dynamics_certified']:
        if not bool(arrays[flag].item()):raise ValueError('Unapproved NPZ')
    if str(arrays['dynamics_certificate_sha256'].item())!=file_sha256(certificate_path):raise ValueError('NPZ/certificate binding mismatch')
    if not np.allclose(np.linalg.norm(arrays['root_quaternions_xyzw'],axis=1),1.,atol=1e-7):raise ValueError('Invalid root quaternion')
    return VerifiedStyleMotion(arrays,names,dt,certificate)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);args=p.parse_args();motion=load_verified_motion(args.manifest)
    print(json.dumps({'verified':True,'frames':len(motion.arrays['time_s']),'adjacent_training_pairs':len(motion.valid_pair_indices),'control_hz':1/motion.dt_s,'joint_names':motion.joint_names,'cyclic_seam_allowed':False,'amp_discriminator_implemented_by_this_module':False},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
