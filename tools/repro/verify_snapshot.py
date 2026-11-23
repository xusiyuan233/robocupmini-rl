"""Verify the portable snapshot without launching Isaac Sim or using the GPU."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='Also inspect a trusted local AMP checkpoint (requires torch).')
    args = parser.parse_args()
    integrity = json.loads((ROOT / 'reproducibility/file_manifest.json').read_text())
    files = integrity['sha256']
    for relative, expected in files.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f'Outside snapshot: {relative}')
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Missing or changed snapshot file: {relative}')
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=relative)
    print(f'PASS: {len(files)} snapshot files match SHA-256; recorded Python syntax is valid.')
    asset = ROOT / 'exts/bipedal_locomotion/bipedal_locomotion/assets/usd/ROBOCUPMINI'
    urdf = asset / 'urdf/robotcupmini.urdf'
    tree = ET.parse(urdf)
    joints = [j.get('name') for j in tree.findall('joint') if j.get('type') not in ('fixed', 'floating')]
    if len(joints) != 13:
        raise ValueError(f'Expected 13 movable joints, found {len(joints)}')
    for mesh in tree.findall('.//mesh'):
        name = mesh.attrib['filename']
        if Path(name).is_absolute() or not (urdf.parent / name).is_file():
            raise ValueError(f'Nonportable/missing mesh: {name}')
    print('PASS: URDF has 13 movable joints; relative mesh references exist.')
    manifest = json.loads((ROOT / 'data/motions/robotcupmini/style_reference_manifest.json').read_text())
    for key in ('motion', 'certificate', 'style_quality_report', 'control_targets'):
        entry = manifest[key]
        path = (ROOT / entry['path']).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != entry['sha256']:
            raise ValueError(f'Expert data binding failed: {key}')
    print('PASS: selected demonstration, reports and control targets retain their approved hashes.')
    if args.checkpoint:
        import torch
        checkpoint = torch.load(args.checkpoint.resolve(), map_location='cpu', weights_only=False)
        required = {'model_state_dict', 'encoder_state_dict', 'optimizer_state_dict', 'iter', 'amp'}
        if not required.issubset(checkpoint):
            raise ValueError('Not a complete AMP training checkpoint')
        amp = checkpoint['amp']
        if not {'model', 'optimizer', 'style_weight', 'step_dt', 'expert_hash'}.issubset(amp):
            raise ValueError('Incomplete discriminator state')
        from tools.amp.motion_loader import load_expert_features
        features, _ = load_expert_features()
        expected = hashlib.sha256(features.tobytes()).hexdigest()
        if amp['expert_hash'] != expected or abs(amp['step_dt'] - .02) > 1e-8:
            raise ValueError('AMP checkpoint does not match this expert data / control step')
        print(f'PASS: trusted AMP checkpoint iter={checkpoint["iter"]}, weight={amp["style_weight"]}, dt={amp["step_dt"]}.')
    print('No simulator, policy optimizer or training process was started.')


if __name__ == '__main__':
    main()
