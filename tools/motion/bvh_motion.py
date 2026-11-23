"""Read the project's BVH dialect without GUI/matplotlib; preserve global rotations.

BVH channels are intrinsic, in their declared order. Input lengths are centimetres.
The reader rejects other layouts rather than silently guessing their semantics.
"""
from dataclasses import dataclass
import io
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class HumanMotion:
    names: list[str]
    parents: np.ndarray
    offsets_m: np.ndarray
    positions_m: np.ndarray
    rotations: np.ndarray
    dt: float

    def sample(self, times: np.ndarray):
        source_times = np.arange(len(self.positions_m)) * self.dt
        if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError('Sample times must be strictly increasing')
        if times[0] < 0 or times[-1] > source_times[-1]:
            raise ValueError('Sample times outside BVH')
        flat = self.positions_m.reshape(len(source_times), -1)
        pos = np.stack([np.interp(times, source_times, c) for c in flat.T], axis=1)
        rot = np.stack([Slerp(source_times, Rotation.from_matrix(self.rotations[:, j]))(times).as_matrix()
                        for j in range(len(self.names))], axis=1)
        return pos.reshape(len(times), len(self.names), 3), rot


def read_bvh(path: Path) -> HumanMotion:
    hierarchy, motion = path.read_text().split('MOTION', 1)
    tokens = iter(hierarchy.replace('{', ' { ').replace('}', ' } ').split())
    names, parents, offsets, channels = [], [], [], []

    def expect(value):
        actual = next(tokens)
        if actual != value:
            raise ValueError(f'Expected {value}, got {actual}')

    def joint(parent):
        index = len(names)
        names.append(next(tokens)); parents.append(parent)
        offsets.append(None); channels.append(None)
        expect('{')
        while True:
            tag = next(tokens)
            if tag == '}':
                break
            if tag == 'OFFSET':
                offsets[index] = [float(next(tokens)) for _ in range(3)]
            elif tag == 'CHANNELS':
                channels[index] = [next(tokens) for _ in range(int(next(tokens)))]
            elif tag == 'JOINT':
                joint(index)
            elif tag == 'End':
                expect('Site'); expect('{'); expect('OFFSET')
                for _ in range(3):
                    float(next(tokens))
                expect('}')
            else:
                raise ValueError(f'Unsupported BVH tag {tag}')

    expect('HIERARCHY'); expect('ROOT'); joint(-1)
    if next(tokens, None) is not None:
        raise ValueError('Unparsed hierarchy tokens')
    rotations = ['Zrotation', 'Yrotation', 'Xrotation']
    if channels[0] != ['Xposition', 'Yposition', 'Zposition'] + rotations or any(
            c != rotations for c in channels[1:]):
        raise ValueError('Unsupported BVH channel layout')
    lines = motion.strip().splitlines()
    count = int(lines[0].split(':')[1]); dt = float(lines[1].split(':')[1])
    values = np.loadtxt(io.StringIO('\n'.join(lines[2:])), ndmin=2)
    if values.shape != (count, 3 + 3*len(names)) or not np.isfinite(values).all() or dt <= 0:
        raise ValueError('Invalid BVH values')
    offset_array = np.asarray(offsets, dtype=float)*.01
    local_rot = Rotation.from_euler('ZYX', values[:, 3:].reshape(-1, 3), degrees=True
                                   ).as_matrix().reshape(count, len(names), 3, 3)
    world_pos = np.zeros((count, len(names), 3)); world_rot = np.zeros_like(local_rot)
    for j, parent in enumerate(parents):
        if parent < 0:
            world_pos[:, j] = values[:, :3]*.01  # root OFFSET is not added twice
            world_rot[:, j] = local_rot[:, j]
        else:
            world_pos[:, j] = world_pos[:, parent] + np.einsum('tij,j->ti', world_rot[:, parent], offset_array[j])
            world_rot[:, j] = world_rot[:, parent] @ local_rot[:, j]
    return HumanMotion(names, np.asarray(parents), offset_array, world_pos, world_rot, dt)


def anatomical_frame(left, right, up):
    def unit(x):
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError('Degenerate anatomical frame')
        return x/norm
    y = unit(left-right)
    z = unit(up - np.sum(up*y, axis=-1, keepdims=True)*y)
    x = unit(np.cross(y, z))
    return np.stack([x, y, z], axis=-1)
