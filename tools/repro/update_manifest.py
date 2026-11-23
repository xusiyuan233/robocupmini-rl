"""Refresh hashes after intentional source updates; never use to hide failed audits.

This does NOT rewrite the approved demonstration manifest or certificates.
It only records the portable source/asset snapshot for migration checks.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    raw = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT)
    files = sorted(set(filter(None,raw.decode().split('\0'))))
    selected = {}
    for relative in files:
        p = Path(relative)
        # Do not bind unused upstream LFS assets or generated verification outputs.
        include = p.suffix in ('.py','.sh') and p.parts[0] in ('exts','rsl_rl','scripts','tools')
        include |= relative.startswith('exts/bipedal_locomotion/bipedal_locomotion/assets/usd/ROBOCUPMINI/')
        include |= relative.startswith('data/motions/robotcupmini/')
        include |= relative.startswith('reproducibility/') and not relative.startswith('reproducibility/local/')
        if relative in ('reproducibility/file_manifest.json','reproducibility/verification_report.json'):
            include=False
        path=ROOT/p
        if not include or not path.is_file():continue
        if 'lilin' in relative.lower() or path.is_symlink():
            raise ValueError(f'Unapproved source: {relative}')
        selected[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    output={'snapshot_date':'2026-11-23','scope':'Training source, RobotcupMini URDF/USD, curated expert data and reproduction metadata',
            'notes':['Unused upstream robot LFS assets are outside the RobotcupMini integrity contract.',
                     'Historical audit files retain their original hashes; this tool does not approve modified demonstrations.'],
            'sha256':selected}
    (ROOT/'reproducibility/file_manifest.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(f'Recorded {len(selected)} source/asset hashes.')


if __name__=='__main__':
    main()
