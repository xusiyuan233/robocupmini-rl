"""Reuse old PD targets as NUMERICAL guesses for a new reference (never states).

Reference hash identifies the new optimization contract, NOT old-control
feasibility. Explicitly record old/new source hashes and require equal names/dt.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from play_retarget_motion import load_motion,PROJECT_ROOT


def prepare(reference:Path, previous_run:Path, nominal_seed:Path, output:Path):
    for p in (reference,previous_run,nominal_seed,output):
        if not p.resolve().is_relative_to(PROJECT_ROOT) or 'lilin' in str(p.resolve()).lower():raise ValueError('Outside user project')
    if output.exists():raise FileExistsError(output)
    m=json.loads((previous_run/'run_manifest.json').read_text());old=Path(m['input_reference'])
    if not old.resolve().is_relative_to(PROJECT_ROOT) or 'lilin' in str(old).lower():raise ValueError('Old source outside project')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(old)!=m['input_reference_sha256']:raise ValueError('Old reference changed')
    a,names,dt=load_motion(reference);b,old_names,old_dt=load_motion(old)
    if names!=old_names or abs(dt-old_dt)>1e-10 or len(a['time_s'])!=len(b['time_s']):raise ValueError('Warm joint/time mismatch')
    archive=previous_run/'rollout.npz'
    if not archive.exists():raise ValueError('Old diagnostic optimization must finish first')
    with np.load(archive,allow_pickle=False) as z:controls=z['controls'].copy()
    with np.load(nominal_seed,allow_pickle=False) as z:
        if str(z['reference_sha256'].item())!=sha(reference):raise ValueError('Nominal seed reference mismatch')
        u=z['controls'].copy()
    if controls.shape!=(len(a['time_s'])-1,13) or not np.isfinite(controls).all():raise ValueError('Incomplete/invalid old controls')
    u[:len(controls)]=controls
    np.savez_compressed(output,controls=u,reference_sha256=np.array(sha(reference)),
                        physical_reference=np.array(False),warm_only=np.array(True),
                        previous_rollout_sha256=np.array(sha(archive)))
    output.with_suffix('.json').write_text(json.dumps(dict(status='NUMERICAL_GUESS_ONLY_NOT_PHYSICS_OR_TRAINING_DATA',
        new_reference=str(reference.resolve()),new_reference_sha256=sha(reference),previous_reference=str(old),
        previous_reference_sha256=sha(old),previous_run=str(previous_run.resolve()),previous_rollout_sha256=sha(archive),
        copied_states=False,zero_root_forces=True),indent=2))
    print(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('reference','previous-run','nominal-seed','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();prepare(a.reference,a.previous_run,a.nominal_seed,a.output)
if __name__=='__main__':main()
