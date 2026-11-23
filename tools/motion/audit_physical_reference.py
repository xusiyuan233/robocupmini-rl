"""Independent 1 kHz free-root dynamics replay and fail-closed expert export.

The MPC log is NOT trusted as proof. This tool rebuilds the physical model from
its manifest, integrates native PD targets without reference-pose overwrites,
measures actual torques/velocities/contacts, and only exports a demonstration if
ALL declared gates pass. No PPO/AMP learning and no IsaacLab scene are started.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation
from play_retarget_motion import load_motion,MODEL_PATH
from physical_reference_model import build_physical_model,check_transition,mj
from reference_contact_quality import interior_mask,contact_geometry_report

# Declared before the full run; changing these requires a new report/version.
GATES={
    'joint_speed_limit_rad_s':5.,'joint_speed_numerical_tolerance_rad_s':.001,
    'maximum_effort_ratio':1.000001,'hard_joint_limit_tolerance_rad':.005,
    'maximum_root_tilt_deg':25.,'minimum_root_height_m':.40,
    'maximum_mesh_penetration_m':.002,'minimum_swing_clearance_m':.003,
    'minimum_mean_forward_speed_m_s':.08,'minimum_duration_s':8.,
    'maximum_support_contact_slip_m_s':.05,'support_contact_slip_p95_m_s':.02,
    'minimum_contact_phase_agreement':.90,
    'minimum_lateral_foot_separation_m':.03,
    'maximum_replay_state_error':1e-5,
}

def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(run: Path,output: Path,export: Path | None=None):
    """Replay every physical step; archive failures, never overwrite prior output."""
    if output.exists() or (export is not None and export.exists()):
        raise FileExistsError('Refusing to overwrite audit/export')
    for p in [run,output,export]:
        if p is not None and 'lilin' in str(p.resolve()).lower():
            raise ValueError('Another user project is outside scope')
    manifest=json.loads((run/'run_manifest.json').read_text())
    motion_path=Path(manifest['input_reference'])
    if sha(motion_path)!=manifest['input_reference_sha256'] or sha(MODEL_PATH)!=manifest['mjcf_sha256']:
        raise ValueError('Source/model changed after optimization')
    a,names,dt=load_motion(motion_path)
    archive=run/'rollout.npz'
    if not archive.exists():archive=run/'progress.npz'
    with np.load(archive,allow_pickle=False) as ar:
        expected=ar['states'].copy();controls=ar['controls'].copy()
    frames=len(controls)
    if expected.shape!=(frames+1,40) or controls.shape!=(frames,13) or not np.isfinite(controls).all():
        raise ValueError('Invalid trajectory arrays')
    if not np.allclose(expected[:,0],np.arange(frames+1)*dt,atol=1e-8):
        raise ValueError('Trajectory contains resets or inconsistent timestamps')
    m,ids,adr,dofs,feet,vertices,anchors,kp,kd,effort=build_physical_model(names,manifest.get('contact_time_constant_s'),joint_limit_time_constant=manifest.get('joint_limit_time_constant_s'))
    d=mj.MjData(m);work=mj.MjData(m)
    mj.mj_setState(m,d,expected[0],mj.mjtState.mjSTATE_FULLPHYSICS)
    stride=round(dt/m.opt.timestep);jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
    mesh_z_1khz=[];com_50hz=[];full_states=[];torques=[];contact_slip=[];contact_load=[];mesh_z=[];anchor_vel=[];anchor_pos=[];foot_rot=[]
    warnings_initial=np.array([w.number for w in d.warning]);unexpected_contact_count=0;actual_contact=[]
    snapshots=[expected[0].copy()];replay_errors=[];external_force_max=0.
    # Continuous native mj_step; warm starts are numerical solver guesses, not states.
    # The MPC rollout uses a zero solver warm start at each 20 ms boundary.
    for k,u in enumerate(controls):
        d.ctrl[:]=u
        d.qacc_warmstart[:]=0
        start_time=float(d.time)
        for sub in range(stride):
            external_force_max=max(external_force_max,float(np.max(np.abs(d.qfrc_applied))),float(np.max(np.abs(d.xfrc_applied))))
            mj.mj_step(m,d)
            state=np.empty(1+m.nq+m.nv);mj.mj_getState(m,d,state,mj.mjtState.mjSTATE_FULLPHYSICS)
            if not np.isfinite(state).all() or abs(float(d.time)-(start_time+(sub+1)*m.opt.timestep))>1e-8:
                raise FloatingPointError('Numerical reset/nonfinite state: no export permitted')
            full_states.append(state);torques.append(d.actuator_force.copy())
            # Re-evaluate contact geometry on a SEPARATE data object. Never alter plant state.
            work.qpos[:]=d.qpos;work.qvel[:]=d.qvel;work.ctrl[:]=d.ctrl;mj.mj_forward(m,work)
            mesh_z_1khz.append([float((verts@work.xmat[b].reshape(3,3)[2]+work.xpos[b,2]).min())
                                  for b,verts in zip(feet,vertices)])
            slip=[[],[]];loads=np.zeros(2)
            for ci in range(work.ncon):
                con=work.contact[ci];g1,g2=con.geom1,con.geom2
                b1,b2=int(m.geom_bodyid[g1]),int(m.geom_bodyid[g2])
                b=b2 if b1==0 else b1 if b2==0 else -1
                if b not in feet:
                    if con.dist<=0:unexpected_contact_count+=1
                    continue
                side=feet.index(b);f=np.zeros(6);mj.mj_contactForce(m,work,ci,f)
                loads[side]+=max(0.,f[0])
                # Measure the physical contact point, not an ankle/sole centre that
                # legitimately moves during heel/toe rolling.
                if f[0]>5.:
                    mj.mj_jac(m,work,jp,jr,con.pos,b)
                    slip[side].append(float(np.linalg.norm((jp@work.qvel)[:2])))
            contact_load.append(loads)
            contact_slip.append([max(v) if v else np.nan for v in slip])
            actual_contact.append(loads>10.)
        snapshot=full_states[-1];snapshots.append(snapshot.copy());check_transition(snapshots[-2],snapshot,dt)
        replay_errors.append(float(np.max(np.abs(snapshot-expected[k+1]))))
        zs=[];vs=[];pts=[];rs=[]
        for b,verts,an in zip(feet,vertices,anchors):
            r=work.xmat[b].reshape(3,3);point=work.xpos[b]+r@an
            zs.append(float((verts@r.T+work.xpos[b])[:,2].min()))
            mj.mj_jac(m,work,jp,jr,point,b)
            vs.append(jp@work.qvel);pts.append(point);rs.append(r.copy())
        mesh_z.append(zs);anchor_vel.append(vs);anchor_pos.append(pts);foot_rot.append(rs);com_50hz.append(work.subtree_com[1].copy())
    mesh_z_1khz=np.asarray(mesh_z_1khz)
    S=np.asarray(full_states);T=np.asarray(torques);snapshots=np.asarray(snapshots);mesh_z=np.asarray(mesh_z)
    anchor_vel=np.asarray(anchor_vel);anchor_pos=np.asarray(anchor_pos);foot_rot=np.asarray(foot_rot)
    slip=np.asarray(contact_slip);loads=np.asarray(contact_load);actual_contact=np.asarray(actual_contact)
    expected_phase=np.repeat(a['contact_candidates'][:frames],stride,axis=0)
    # Exclude only the established 100 ms transitions, never whole failed swings.
    stance_interior=np.repeat(interior_mask(a['contact_candidates'][:frames],5),stride,axis=0)
    valid_slip=slip[stance_interior & np.isfinite(slip)]
    slip_max=float(valid_slip.max()) if len(valid_slip) else float('inf')
    slip_p95=float(np.quantile(valid_slip,.95)) if len(valid_slip) else float('inf')
    phase_interior=np.repeat(interior_mask(a['contact_candidates'][:frames],5)|interior_mask(~a['contact_candidates'][:frames],5),stride,axis=0)
    phase_agreement=float((actual_contact==expected_phase)[phase_interior].mean()) if phase_interior.any() else 0.
    q=S[:,1:1+m.nq];v=S[:,1+m.nq:];duration=frames*dt
    tilt=np.rad2deg(np.arccos(np.clip(Rotation.from_quat(q[:,[4,5,6,3]]).as_matrix()[:,2,2],-1,1)))
    speed=float(abs(v[:,dofs]).max());ratio=float(abs(T/effort).max())
    lower=m.jnt_range[ids,0];upper=m.jnt_range[ids,1]
    position_error=float(max(0.,np.max(lower-q[:,adr]),np.max(q[:,adr]-upper)))
    geometry=contact_geometry_report(mesh_z,anchor_vel,a['contact_candidates'][1:frames+1],dt)
    phase_swing=interior_mask(~a['contact_candidates'][1:frames+1],5)
    swing_1khz=np.repeat(interior_mask(~a['contact_candidates'][:frames],5),stride,axis=0)
    swing_clear=float(mesh_z_1khz[swing_1khz].min()) if swing_1khz.any() else -float('inf')
    world_to_body=Rotation.from_quat(snapshots[1:,1+np.array([4,5,6,3])]).inv()
    separation=world_to_body.apply(anchor_pos[:,0]-anchor_pos[:,1])[:,1]
    mean_speed=float((q[-1,0]-snapshots[0,1]) / max(duration,1e-9))
    warnings_final=np.array([w.number for w in d.warning]);bad_warnings=(warnings_final-warnings_initial).tolist()
    measured={'duration_s':duration,'full_clip_duration_s':float(a['time_s'][-1]),'physics_steps':len(S),
       'peak_actual_joint_speed_rad_s':speed,'peak_actual_effort_ratio':ratio,
       'maximum_hard_joint_limit_error_rad':position_error,'maximum_root_tilt_deg':float(tilt.max()),
       'minimum_root_height_m':float(q[:,2].min()),'maximum_mesh_penetration_m':float(max(0.,-mesh_z_1khz.min())),
       'minimum_interior_swing_clearance_m':swing_clear,'mean_forward_speed_m_s':mean_speed,
       'support_contact_point_slip_max_m_s':slip_max,'support_contact_point_slip_p95_m_s':slip_p95,
       'contact_phase_agreement':phase_agreement,'minimum_lateral_foot_separation_m':float(separation.min()),
       'maximum_replay_state_error':max(replay_errors),'maximum_external_force':external_force_max,
       'unexpected_nonfoot_ground_contacts':unexpected_contact_count,'new_mujoco_warning_counts':bad_warnings}
    checks={
       'full_clip':frames==len(a['time_s'])-1 and duration>=GATES['minimum_duration_s'],
       'actual_speed':speed<=GATES['joint_speed_limit_rad_s']+GATES['joint_speed_numerical_tolerance_rad_s'],
       'actual_effort':ratio<=GATES['maximum_effort_ratio'],
       'joint_positions':position_error<=GATES['hard_joint_limit_tolerance_rad'],
       'no_fall':tilt.max()<=GATES['maximum_root_tilt_deg'] and q[:,2].min()>=GATES['minimum_root_height_m'],
       'penetration':measured['maximum_mesh_penetration_m']<=GATES['maximum_mesh_penetration_m'],
       'swing_clearance':swing_clear>=GATES['minimum_swing_clearance_m'],
       'forward_progress':mean_speed>=GATES['minimum_mean_forward_speed_m_s'],
       'support_contact_slip':slip_max<=GATES['maximum_support_contact_slip_m_s'] and slip_p95<=GATES['support_contact_slip_p95_m_s'],
       'contact_phase':phase_agreement>=GATES['minimum_contact_phase_agreement'],
       'no_crossed_feet':separation.min()>=GATES['minimum_lateral_foot_separation_m'],
       'independent_replay':max(replay_errors)<=GATES['maximum_replay_state_error'],
       'no_external_forces':external_force_max==0.,'no_nonfoot_ground_contacts':unexpected_contact_count==0,
       'no_numerical_warnings':not any(bad_warnings),
    }
    checks={key:bool(value) for key,value in checks.items()};passed=all(checks.values())
    report={'status':'MUJOCO_PHYSICAL_DEMONSTRATION_PASSED' if passed else 'NOT_APPROVED_FOR_STYLE_TRAINING',
       'dynamics_certified_in_this_mujoco_contract':passed,'amp_learner_integrated':False,
       'isaaclab_physical_replay_verified':False,'sim2real_safe':False,
       'reference_motion':str(motion_path),'reference_sha256':sha(motion_path),
       'optimizer_rollout_sha256':sha(archive),'model_sha256':sha(MODEL_PATH),
       'physics_contract':manifest,'tested_model':{'joint_names':names,'mass_kg':float(m.body_mass.sum()),'kp':kp.tolist(),'kd':kd.tolist(),'effort_limits_Nm':effort.tolist(),'hard_joint_limits_rad':m.jnt_range[ids].tolist(),'joint_armature':m.dof_armature[dofs].tolist(),'physics_dt_s':float(m.opt.timestep),'control_dt_s':dt,'gravity':m.opt.gravity.tolist(),'ground_friction':m.geom_friction[m.geom('floor').id].tolist(),'self_collisions':False,'root_is_free':True},'criteria_version':'20261119_v2_allstep_geometry','thresholds':GATES,
       'measured':measured,'checks':checks,'failed_checks':[k for k,v in checks.items() if not v],
       'anchor_geometry_diagnostic_not_contact_slip_gate':geometry,
       'qualification_scope':'One nonperiodic source-derived clip under stated MuJoCo contract; not a trained policy or robustness/Isaac/real certificate.',
       'source_kind':str(a.get('source_kind',np.array('human_motion_retarget')).item())}
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'physics_replay.npz',states_1khz=S,torques_Nm=T,states_50hz=snapshots,
       controls_50hz=controls,mesh_min_z_m=mesh_z,mesh_min_z_1khz_m=mesh_z_1khz,foot_anchor_velocity_m_s=anchor_vel,
       foot_anchor_positions_m=anchor_pos,foot_rotations=foot_rot,
       contact_slip_m_s=slip,contact_load_N=loads,contact_phase=actual_contact,
       replay_state_errors=replay_errors)
    (output/'decision.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if export is not None and passed:
        export.mkdir(parents=True,exist_ok=False)
        Q=snapshots[:,1:1+m.nq];V=snapshots[:,1+m.nq:]
        work.qpos[:]=Q[0];work.qvel[:]=V[0];work.ctrl[:]=controls[0];mj.mj_forward(m,work)
        first_positions=np.asarray([work.xpos[b]+work.xmat[b].reshape(3,3)@an for b,an in zip(feet,anchors)])
        first_rotations=np.asarray([work.xmat[b].reshape(3,3).copy() for b in feet])
        initial_com=work.subtree_com[1].copy()
        a['source_contact_candidates']=a['contact_candidates'].copy()
        for obsolete in ['planned_cop_world_m','planned_foot_targets_world_m']:
            a.pop(obsolete,None)
        a.update(time_s=np.arange(len(Q))*dt,root_positions_m=Q[:,:3],
            root_quaternions_xyzw=Q[:,[4,5,6,3]],root_rpy_rad=Rotation.from_quat(Q[:,[4,5,6,3]]).as_euler('xyz'),
            joint_positions_rad=Q[:,adr],joint_velocities_rad_s=V[:,dofs],
            root_linear_velocities_world_m_s=V[:,:3],
            root_angular_velocities_world_rad_s=Rotation.from_quat(Q[:,[4,5,6,3]]).apply(V[:,3:6]),
            foot_targets_world_m=np.vstack([first_positions[None],anchor_pos]),
            foot_target_rotations=np.vstack([first_rotations[None],foot_rot]),
            reference_com_world_m=np.vstack([initial_com[None],np.asarray(com_50hz)]),
            robot_root_target_rotations=Rotation.from_quat(Q[:,[4,5,6,3]]).as_matrix(),
            contact_candidates=np.vstack([actual_contact[:1],actual_contact[stride-1::stride]]),
            valid_for_training=np.array(True),valid_for_style_kinematics=np.array(True),
            dynamics_certified=np.array(True),candidate_requires_dynamics_qualification=np.array(False),
            initializer_only=np.array(False),periodic_reference=np.array(False),
            reference_provenance=np.array('Actual free-root, unforced, effort-limited MuJoCo MPC teacher rollout; source provenance and morphology adaptations documented.'),
            dynamics_certificate_sha256=np.array(sha(output/'decision.json')))
        np.savez_compressed(export/'candidate_motion.npz',**a)
        np.savez_compressed(export/'controller_commands.npz',joint_names=names,control_dt_s=dt,position_targets_rad=controls,initial_state_fullphysics=snapshots[0])
        (export/'generation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'status':report['status'],'failed_checks':report['failed_checks'],'measured':measured},ensure_ascii=False,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--export',type=Path)
    args=p.parse_args();replay(args.run,args.output,args.export)

if __name__=='__main__':main()
