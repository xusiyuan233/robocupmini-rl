"""Package trusted training checkpoints without editing or stopping training.

Run from the user's project. Output is ignored by Git and intended for a private
GitHub Release. Copies are validated on CPU; no simulation or training occurs.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    import torch
    from tools.amp.motion_loader import load_expert_features
    features, _ = load_expert_features()
    expert_hash = hashlib.sha256(features.tobytes()).hexdigest()
    destination = ROOT / 'reproducibility/local/release'
    staging = destination / 'staging'
    destination.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        raise FileExistsError(f'Choose/remove this tool-owned staging directory before rebuilding: {staging}')
    staging.mkdir()
    entries = []
    specs = [
        ('robotcupmini_velocity', '2026-11-09_16-30-32_forward_gait_v18_stance_yaw', 'ppo_velocity_v18', [15000]),
        ('robotcupmini_style_prep', '2026-11-23_15-04-37_amp_walk_pilot_v1', 'amp_walk_pilot_v1', [50, 100, 150, 200]),
        ('robotcupmini_style_prep', '2026-11-23_15-24-04_amp_align_v2', 'amp_align_v2', [50, 100, 150, 200]),
    ]
    # Include only checkpoints already atomically stable at collection time.
    new_run = '2026-11-23_15-55-02_amp_align_v2_from150_2000'
    existing = sorted((ROOT / 'logs/rsl_rl/robotcupmini_style_prep' / new_run).glob('model_*.pt'))
    new_iterations = [int(p.stem.split('_')[-1]) for p in existing
                      if time.time() - p.stat().st_mtime > 10]
    if new_iterations:
        specs.append(('robotcupmini_style_prep', new_run, 'amp_align_v2_continued_unreviewed', new_iterations))
    for experiment, run, alias, iterations in specs:
        for iteration in iterations:
            src = ROOT / 'logs/rsl_rl' / experiment / run / f'model_{iteration}.pt'
            if not src.is_file():
                raise FileNotFoundError(src)
            before = (src.stat().st_size, src.stat().st_mtime_ns)
            relative = Path('artifacts/checkpoints') / alias / src.name
            dest = staging / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            after = (src.stat().st_size, src.stat().st_mtime_ns)
            if before != after or sha256(src) != sha256(dest):
                raise RuntimeError(f'Checkpoint changed during copy: {src.name}; no training process was interrupted')
            checkpoint = torch.load(dest, map_location='cpu', weights_only=False)
            if int(checkpoint['iter']) != iteration:
                raise ValueError(f'Iteration mismatch for {src.name}')
            if 'amp' in checkpoint:
                amp = checkpoint['amp']
                if amp['expert_hash'] != expert_hash or amp['step_dt'] != .02 or amp['style_weight'] != 1.0:
                    raise ValueError(f'Expert/runtime mismatch: {src.name}')
            entries.append({'path':relative.as_posix(),'sha256':sha256(dest),'bytes':dest.stat().st_size,
                            'original_experiment':experiment,'original_run':run,'iteration':iteration,
                            'contains_amp_state':'amp' in checkpoint,
                            'selected_resume_baseline':alias=='amp_align_v2' and iteration==150,
                            'playback_review':'not reviewed after new continuation' if 'unreviewed' in alias else 'see README; not hardware-certified'})
    document = {'snapshot_date':'2026-11-23','scope':'Trusted local IsaacLab training checkpoints only; no Sim2Sim',
                'expert_features_sha256':expert_hash,'files':entries}
    (staging / 'artifacts/artifact_manifest.json').write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n')
    (ROOT / 'reproducibility/artifact_catalog.json').write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n')
    archive = destination / 'robotcupmini-training-artifacts-20261123.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as output:
        output.add(staging / 'artifacts',arcname='artifacts')
    checksum = sha256(archive)
    archive.with_suffix(archive.suffix+'.sha256').write_text(f'{checksum}  {archive.name}\n')
    print(json.dumps({'checkpoints':len(entries),'archive':str(archive),'bytes':archive.stat().st_size,'sha256':checksum},indent=2))


if __name__ == '__main__':
    main()
