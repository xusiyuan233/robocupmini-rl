"""Exact world-frame torso angular velocity for this free-root + waist-Z robot."""
from __future__ import annotations
import numpy as np


def torso_world_angular_velocity(q:np.ndarray,v:np.ndarray,waist_dof:int)->np.ndarray:
    """q=root xyz,wxyz,joints; v=root linear,LOCAL angular,joint rates.

    Supports leading batch/time dimensions. No numerical derivative or smoothing.
    """
    w=q[...,3];xyz=q[...,4:7];local=v[...,3:6]
    cross=2*np.cross(xyz,local)
    world=local+w[...,None]*cross+np.cross(xyz,cross)
    z=np.zeros_like(local);z[...,2]=1.
    zcross=2*np.cross(xyz,z)
    axis=z+w[...,None]*zcross+np.cross(xyz,zcross)
    return world+axis*v[...,waist_dof,None]
