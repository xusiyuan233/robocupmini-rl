"""Collect a published Berkeley policy on its OWN model, never on ROBOCUPMINI.
Observation, action scale, timing and phase follow the pinned DeepMind example.
Only density-zero noncolliding visual meshes are omitted if unavailable; all
inertials, joints, actuators, collision geometry and solver settings are retained.
This policy is NOT the checkpoint behind the Berkeley paper's real-robot video.
"""
from pathlib import Path
import argparse,json,hashlib,xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'data/motions/robotcupmini/raw/berkeley_humanoid/playground_v0_0_5'

def build_teacher():
    xml=ET.parse(SOURCE/'berkeley_humanoid_mjx_feetonly.xml').getroot()
    omitted=[]
    for parent in xml.iter():
        for child in list(parent):
            if child.tag=='mesh' or (child.tag=='geom' and child.get('class')=='visual'):
                omitted.append(ET.tostring(child,encoding='unicode'));parent.remove(child)
    scene=ET.parse(SOURCE/'scene_mjx_feetonly_flat_terrain.xml').getroot()
    for tag in ['asset','worldbody','keyframe','visual','statistic']:
        element=scene.find(tag)
        if element is None:continue
        dest=xml.find(tag)
        if dest is None:xml.append(element)
        else:dest.extend(list(element))
    model=mj.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
    return model,xml,omitted

def collect(args):
    if not (np.isfinite(args.duration) and 1 <= args.duration <= 60 and np.isfinite(args.speed) and abs(args.speed) <= .6 and 0 <= args.heading_kp <= 4):
        raise ValueError('Invalid source rollout parameters')
    for path in [args.output,SOURCE]:
        if not path.resolve().is_relative_to(ROOT) or 'lilin' in str(path).lower():raise ValueError('Outside user project')
    if args.output.exists():raise FileExistsError(args.output)
    m,xml,omitted=build_teacher();d=mj.MjData(m)
    mj.mj_resetDataKeyframe(m,d,m.key('home').id);mj.mj_forward(m,d)
    default=m.key('home').qpos[7:].copy();phase=np.array([0.,np.pi]);last=np.zeros(12,dtype=np.float32)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(SOURCE/'bh_policy.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    dt=.02;sub=round(dt/m.opt.timestep);frequency=1.5
    rows={key:[] for key in ['time_s','qpos','qvel','action','command','position_targets_rad','foot_positions_m','foot_rotations','contact_load_N','root_velocity_body_m_s','root_torso_rotations']}
    feet=[m.site('l_foot').id,m.site('r_foot').id];footbody=[m.site_bodyid[f] for f in feet]
    for step in range(round(args.duration/dt)+1):
        yaw=float(np.arctan2(d.site_xmat[m.site('imu').id].reshape(3,3)[1,0],d.site_xmat[m.site('imu').id].reshape(3,3)[0,0]))
        command=np.array([args.speed,0.,np.clip(-args.heading_kp*yaw,-.5,.5)])
        mat=d.site_xmat[m.site('imu').id].reshape(3,3)
        obs=np.concatenate([d.sensor('local_linvel').data,d.sensor('gyro').data,mat.T@np.array([0.,0.,-1.]),command,d.qpos[7:]-default,d.qvel[6:],last,np.cos(phase),np.sin(phase)]).astype(np.float32)
        action=session.run(['continuous_actions'],{'obs':obs[None]})[0][0]
        target=default+.5*action
        loads=np.zeros(2)
        for k in range(d.ncon):
            con=d.contact[k];bs=[m.geom_bodyid[con.geom1],m.geom_bodyid[con.geom2]]
            for side,bid in enumerate(footbody):
                if bid in bs:
                    force=np.zeros(6);mj.mj_contactForce(m,d,k,force);loads[side]+=max(0.,force[0])
        values=[step*dt,d.qpos.copy(),d.qvel.copy(),action.copy(),command.copy(),target.copy(),d.site_xpos[feet].copy(),d.site_xmat[feet].reshape(2,3,3).copy(),loads,d.sensor('local_linvel').data.copy(),mat.copy()]
        for k,v in zip(rows,values):rows[k].append(v)
        tilt=np.degrees(np.arccos(np.clip(mat[2,2],-1,1)))
        if d.qpos[2]<.3 or tilt>45 or not np.isfinite(d.qpos).all():raise RuntimeError(f'Teacher fell t={step*dt:.2f} h={d.qpos[2]:.3f} tilt={tilt:.1f}')
        if step%100==0:print('TEACHER',step*dt,'speed',np.round(d.sensor('local_linvel').data,3),'tilt',round(tilt,2),'height',round(d.qpos[2],3),flush=True)
        if step==round(args.duration/dt):break
        last=action.copy();d.ctrl[:]=target;phase=(phase+2*np.pi*frequency*dt+np.pi)%(2*np.pi)-np.pi
        for _ in range(sub):mj.mj_step(m,d)
    args.output.mkdir(parents=True)
    arrays={k:np.asarray(v) for k,v in rows.items()};arrays.update(joint_names=np.array([m.joint(i).name for i in range(1,m.njnt)]),sample_dt_s=np.array(dt),valid_for_training=np.array(False),source_kind=np.array('published_robot_policy_rollout'),source_policy_sha256=np.array(hashlib.sha256((SOURCE/'bh_policy.onnx').read_bytes()).hexdigest()))
    np.savez_compressed(args.output/'teacher_rollout.npz',**arrays)
    (args.output/'teacher_model.xml').write_text(ET.tostring(xml,encoding='unicode'))
    report={'status':'SOURCE_TEACHER_COLLECTED_NOT_TARGET_ROBOT_APPROVED','source_repository':'google-deepmind/mujoco_playground','source_commit':'81dfe512c9f2f03107fda1e31de585d04bb30bc4','same_robot_but_not_video_checkpoint':True,'command_m_s':args.speed,'source_control_dt_s':dt,'source_physics_dt_s':float(m.opt.timestep),'frames':len(arrays['time_s']),'omitted_density_zero_visual_items':len(omitted),'physical_parameters_changed':False,'mass_kg':float(m.body_mass.sum()),'steady_mean_body_velocity_m_s':arrays['root_velocity_body_m_s'][100:].mean(0).tolist(),'action_scale':.5,'phase_frequency_hz':frequency,'heading_hold_kp':args.heading_kp}
    (args.output/'collection_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--duration',type=float,default=14.);parser.add_argument('--speed',type=float,default=.4);parser.add_argument('--heading-kp',type=float,default=0.);collect(parser.parse_args())
