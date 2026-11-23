"""Shared, explicit physics contract for the CPU reference teacher and its audit.

Ground/contact filtering and 13 native PD actuators are added IN MEMORY.
Optional contact/joint-stop solver calibration is explicit and recorded in the
run manifest; it does NOT alter nominal effort/velocity/position limits.
The on-disk MJCF, masses, inertias, joints, armatures and collision shapes are not
changed. Robot self-collision is off, as in the user's IsaacLab training asset.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, cast
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as _mj
from play_retarget_motion import MODEL_PATH, joint_mapping
from audit_style_pretraining import mesh_points
mj=cast(Any,_mj)


def build_physical_model(names: list[str], contact_time_constant: float | None = None, sensors: bool = False, joint_limit_time_constant: float | None = None):
    """Return (model, joint qpos/dof indices, feet, sole meshes, kp,kd,effort)."""
    if len(names)!=13 or len(set(names))!=13:
        raise ValueError('Exactly 13 uniquely named joints required')
    xml=ET.parse(MODEL_PATH).getroot()
    compiler=xml.find('compiler')
    if compiler is None:
        raise ValueError('Missing compiler')
    compiler.set('meshdir',str((MODEL_PATH.parent/compiler.get('meshdir','.')).resolve()))
    world=xml.find('worldbody')
    if world is None:
        raise ValueError('Missing worldbody')
    ET.SubElement(world,'geom',name='floor',type='plane',size='8 8 .1',
                  friction='.8 .005 .0001',contype='1',conaffinity='2')
    for g in xml.findall('.//geom'):
        if g.get('name')=='floor':
            continue
        if g.get('contype','1')!='0' or g.get('conaffinity','1')!='0':
            g.set('contype','2');g.set('conaffinity','1')
            g.set('friction','.8 .005 .0001')
    kp=np.array([30. if 'ankle' in n else 40. if 'waist' in n else 60. for n in names])
    kd=kp*.05
    effort=np.array([36. if 'ankle' in n else 60. for n in names])
    actuators=ET.SubElement(xml,'actuator')
    for i,name in enumerate(names):
        ET.SubElement(actuators,'position',name='teacher_'+name,joint=name,
                      kp=str(kp[i]),kv=str(kd[i]),forcelimited='true',
                      forcerange=f'{-effort[i]} {effort[i]}')
    model=mj.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
    if contact_time_constant is not None:
        if not .004 <= contact_time_constant <= .02:
            raise ValueError('Contact time constant outside explicit calibration range')
        model.geom_solref[:]=[contact_time_constant,1.]
        model.geom_solimp[:]=[.95,.99,.001,.5,2.]
    if joint_limit_time_constant is not None:
        if not .002 <= joint_limit_time_constant <= .02:
            raise ValueError('Invalid joint-stop solver time constant')
        model.jnt_solref[:]=[joint_limit_time_constant,1.]
        model.jnt_solimp[:]=[.95,.99,.001,.5,2.]
    model.opt.timestep=.001
    model.opt.integrator=mj.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.iterations=60
    model.opt.tolerance=1e-10
    ids,addresses,_=joint_mapping(model,names)
    dofs=model.jnt_dofadr[ids]
    feet=[model.body(s+'_ankle_roll_link').id for s in ['left','right']]
    vertices=[]
    anchors=[]
    for body in feet:
        mesh=np.concatenate([mesh_points(model,g) for g in range(model.ngeom)
                             if model.geom_bodyid[g]==body
                             and model.geom_type[g]==mj.mjtGeom.mjGEOM_MESH
                             and model.geom_contype[g]])
        vertices.append(mesh)
        anchors.append(np.r_[(mesh.min(0)[:2]+mesh.max(0)[:2])/2,mesh[:,2].min()])
    if sensors:
        sensor=ET.SubElement(xml,'sensor')
        for side,anchor in zip(['left','right'],anchors):
            name=side+'_ankle_roll_link'
            body=xml.find(f".//body[@name='{name}']")
            if body is None:raise ValueError('Foot body missing')
            sitename=side+'_teacher_sole'
            ET.SubElement(body,'site',name=sitename,pos=' '.join(map(str,anchor)),size='.001',rgba='0 0 0 0')
            for kind in ['framepos','framequat','framelinvel','frameangvel']:
                ET.SubElement(sensor,kind,objtype='site',objname=sitename)
        instrumented=mj.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
        instrumented.opt.timestep=model.opt.timestep
        instrumented.opt.integrator=model.opt.integrator
        instrumented.opt.iterations=model.opt.iterations
        instrumented.opt.tolerance=model.opt.tolerance
        instrumented.geom_solref[:]=model.geom_solref
        instrumented.geom_solimp[:]=model.geom_solimp
        instrumented.jnt_solref[:]=model.jnt_solref
        instrumented.jnt_solimp[:]=model.jnt_solimp
        model=instrumented
    return model,ids,addresses,dofs,feet,vertices,np.asarray(anchors),kp,kd,effort


def check_transition(initial: np.ndarray, final: np.ndarray, dt: float):
    """Numerical resets/nonfinite states are failures, never a successful rollout."""
    if (initial.ndim!=1 or final.shape!=initial.shape
            or not np.isfinite(initial).all() or not np.isfinite(final).all()
            or abs(float(final[0]-initial[0])-dt)>1e-8):
        raise FloatingPointError('Nonfinite state or numerical reset in physics rollout')


def oriented_box_min_z(positions: np.ndarray, quaternions_wxyz: np.ndarray,
                       local_lower: np.ndarray, local_upper: np.ndarray) -> np.ndarray:
    """Conservative sole-mesh clearance at the sole-anchor site (batchable).

    The physical audit still uses every mesh vertex. This cheap bounding-box
    proxy is used only inside sampling optimization; it cannot certify contact.
    """
    qw,qx,qy,qz=np.moveaxis(quaternions_wxyz,-1,0)
    row_z=np.stack([2*(qx*qz-qw*qy),2*(qy*qz+qw*qx),1-2*(qx*qx+qy*qy)],axis=-1)
    return positions[...,2]+np.minimum(row_z*local_lower,row_z*local_upper).sum(-1)


def contact_speed_bound(linear_velocity: np.ndarray, angular_velocity: np.ndarray,
                        maximum_site_to_contact_radius: float=.14) -> np.ndarray:
    """An upper bound for horizontal rigid-sole point speed, not contact force."""
    if maximum_site_to_contact_radius<=0:
        raise ValueError('Contact radius must be positive')
    return np.linalg.norm(linear_velocity[...,:2],axis=-1)+maximum_site_to_contact_radius*np.linalg.norm(angular_velocity,axis=-1)
