"""Pure numerical planning features. Never filter or overwrite physical states."""
import numpy as np


def torso_rotation_vector(q, waist_address):
    """World torso principal rotation vector; free root plus a local waist Z."""
    q = np.asarray(q)
    half = q[..., waist_address]/2
    c, s = np.cos(half), np.sin(half)
    w, x, y, z = np.moveaxis(q[..., 3:7], -1, 0)
    quat = np.stack([w*c-z*s, x*c+y*s, y*c-x*s, z*c+w*s], axis=-1)
    quat *= np.where(quat[..., :1] < 0, -1., 1.)
    sine = np.linalg.norm(quat[..., 1:], axis=-1)
    angle = 2*np.arctan2(sine, quat[..., 0])
    scale = np.divide(angle, sine, out=np.full_like(sine, 2.), where=sine>1e-10)
    return quat[..., 1:]*scale[..., None]
