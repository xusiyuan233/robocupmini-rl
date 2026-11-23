"""只读检查重定向动作；生成报告，不修改动作/训练配置，不训练策略。

运行环境：env_robocup_sim2sim。MuJoCo 几何检查、简化力可行性筛查和
短时 PD 跟踪仅用于暴露风险，不代表 IsaacLab 或真机可稳定执行。
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from typing import Any, cast

import mujoco as _mujoco
mujoco = cast(Any, _mujoco)
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from play_retarget_motion import build_model, joint_mapping, load_motion, MODEL_PATH
from reference_contact_quality import contact_geometry_report

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'data/motions/robotcupmini/retargeted/walk1_candidate_v3_coordination/candidate_motion.npz'
OUT = ROOT / 'data/motions/robotcupmini/validated/pretraining_audit_20261118'


def summary(x):
    x = np.asarray(x)
    return {k: float(v) for k, v in zip(['min','median','p95','p99','max'],
                                      np.percentile(x, [0,50,95,99,100]))}


def mesh_points(model, geom):
    """真实编译后 mesh 顶点在 body 局部系下的位置。"""
    mid = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    q = model.geom_quat[geom]
    rot = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    return model.mesh_vert[start:start+count] @ rot.T + model.geom_pos[geom]


def runs(mask):
    e = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    motion, names, dt = load_motion(args.motion)
    m = build_model(MODEL_PATH)
    ids, addresses, rootadr = joint_mapping(m, names)
    dofs = m.jnt_dofadr[ids]
    d = mujoco.MjData(m)
    q = motion['joint_positions_rad']
    n = len(q)
    Q = np.tile(m.qpos0, (n, 1))
    Q[:, addresses] = q
    Q[:, rootadr:rootadr+3] = motion['root_positions_m']
    Q[:, rootadr+3:rootadr+7] = motion['root_quaternions_xyzw'][:, [3,0,1,2]]
    V = np.zeros((n, m.nv))
    for i in range(n):
        lo, hi = max(i-1, 0), min(i+1, n-1)
        mujoco.mj_differentiatePos(m, V[i], (hi-lo)*dt, Q[lo], Q[hi])
    A = np.gradient(V, dt, axis=0)
    derivative_method = 'nonperiodic central differences, one-sided endpoints'
    if bool(motion.get('periodic_reference', np.array(False))):
        cycle_frames = round(float(motion['cycle_duration_s']) / dt)
        shift = np.asarray(motion['cycle_translation_m'], dtype=float)
        if cycle_frames < 3 or n % cycle_frames or n < 2*cycle_frames or shift.shape != (3,):
            raise ValueError('Invalid periodic motion metadata')
        if not np.allclose(Q[cycle_frames:, 7:], Q[:-cycle_frames, 7:], atol=1e-7):
            raise ValueError('Joint trajectory does not match declared cycle')
        if not np.allclose(Q[cycle_frames:, :3]-Q[:-cycle_frames, :3], shift, atol=1e-7):
            raise ValueError('Root translation does not match declared cycle')
        if not np.allclose(Q[cycle_frames:, 3:7], Q[:-cycle_frames, 3:7], atol=1e-7):
            raise ValueError('Root rotation does not match declared cycle')
        before, after = Q[-2:].copy(), Q[:2].copy()
        before[:, :3] -= shift * (n // cycle_frames)
        after[:, :3] += shift * (n // cycle_frames)
        extended = np.vstack([before, Q, after])
        ext_v = np.zeros((n+4, m.nv))
        for i in range(1, n+3):
            mujoco.mj_differentiatePos(m, ext_v[i], 2*dt, extended[i-1], extended[i+1])
        V = ext_v[2:-2]
        A = (ext_v[3:-1]-ext_v[1:-3])/(2*dt)
        derivative_method = 'verified periodic pose extension with accumulated root translation; quaternion-aware central differences' 
    contact = motion['contact_candidates'].astype(bool)
    foot_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'{s}_ankle_roll_link')
                for s in ['left','right']]
    points = []
    for b in foot_ids:
        gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and
              m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g]]
        points.append(np.concatenate([mesh_points(m,g) for g in gs]))
    centers = [np.r_[(p.min(0)[:2]+p.max(0)[:2])/2,p.min(0)[2]] for p in points]
    corners = [np.array([[x,y,p.min(0)[2]] for x in [p[:,0].min(),p[:,0].max()]
                         for y in [p[:,1].min(),p[:,1].max()]]) for p in points]
    kp = np.array([30 if 'ankle' in x else 40 if 'waist' in x else 60 for x in names])
    kd = np.array([1.5 if 'ankle' in x else 2 if 'waist' in x else 3 for x in names])
    effort = np.array([36 if 'ankle' in x else 60 for x in names])
    default = np.array([-.075 if 'hip_pitch' in x or 'ankle_pitch' in x else
                        .15 if 'knee' in x else 0 for x in names])
    limits = m.jnt_range[ids]
    report = {'audit_only':True, 'motion':str(args.motion.resolve()),
              'sha256':hashlib.sha256(args.motion.read_bytes()).hexdigest(),
              'frame_count':n,'derivative_method':derivative_method,'sample_dt_s':dt,'duration_s':float((n-1)*dt),
              'valid_for_training_flag':bool(motion['valid_for_training']),
              'joint_order':names,'mujoco_mass_kg':float(m.body_mass.sum()),
              'finite_all_numeric_arrays':all(np.isfinite(motion[k]).all() for k in motion
                  if motion[k].dtype.kind in 'fiu'),
              'hard_limit_max_violation_rad':float(np.maximum(limits[:,0]-q,q-limits[:,1]).clip(0).max()),
              'hard_limit_min_margin_rad':float(np.minimum(q-limits[:,0],limits[:,1]-q).min()),
              'joint_speed_abs_rad_s':summary(np.abs(V[:,dofs])),
              'joint_acceleration_abs_rad_s2':summary(np.abs(A[:,dofs])),
              'speed_limit_5_rad_s_exceeded_frames':int(np.any(abs(V[:,dofs])>5,axis=1).sum()),
              'per_joint_max_speed_rad_s':dict(zip(names,np.max(abs(V[:,dofs]),axis=0).tolist())),
              'saved_joint_velocity_max_error_rad_s':float(abs(V[:,dofs]-motion['joint_velocities_rad_s']).max()),
              'implied_reference_pose_actions':summary(abs((q-default)/.15)),
              'root_height_m':summary(Q[:,2]),
              'root_linear_velocity_world_m_s_mean':np.mean(V[:,:3],axis=0).tolist(),
              'root_linear_velocity_world_m_s_std':np.std(V[:,:3],axis=0).tolist(),
              'no_stance_candidate_frames':int(np.sum(~contact.any(1))),
              'single_stance_frames':int(np.sum(contact.sum(1)==1)),
              'contact_intervals_frames':{s:[[int(a),int(b)] for a,b in runs(contact[:,j])]
                                         for j,s in enumerate(['left','right'])}}
    if 'root_angular_velocities_world_rad_s' in motion:
        # 自由关节角速度为 body-local：必须显式旋转到 world，不能直接比较。
        world_omega=Rotation.from_quat(motion['root_quaternions_xyzw']).apply(V[:,3:6])
        report['saved_root_angular_velocity_world_max_error_rad_s']=float(abs(world_omega-motion['root_angular_velocities_world_rad_s']).max())
    footworld=[]; minz=[]; com=[]; self_penetration=[]; feasible=[]; footerr=[]
    bad_pairs={}; root_feasible=[]; J=np.zeros((3,m.nv)); Jr=np.zeros_like(J); M=np.zeros((m.nv,m.nv))
    # 力筛查不使用 MuJoCo 求解出的约束力：仅 M*a+b 与候选平面接触点的线性平衡。
    # 足底包围盒四角是乐观近似；即使通过，也不证明真实网格/摩擦接触可跟踪。
    for i in range(n):
        d.qpos[:]=Q[i]; d.qvel[:]=V[i]; mujoco.mj_forward(m,d)
        com.append(np.sum(m.body_mass[:,None]*d.xipos,axis=0)/m.body_mass.sum())
        pw=[]; zz=[]; cs=[]
        for side,b in enumerate(foot_ids):
            R=d.xmat[b].reshape(3,3)
            pw.append(R@centers[side]+d.xpos[b])
            zz.append((points[side]@R.T+d.xpos[b])[:,2].min())
            if contact[i,side]: cs.extend([(b,R@c+d.xpos[b]) for c in corners[side]])
        footworld.append(pw); minz.append(zz)
        footerr.append(np.max(abs(np.array(pw)-motion['foot_targets_world_m'][i])))
        maxpen=0.
        for c in d.contact:
            b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
            if b1 and b2 and c.dist < -1e-4:
                # 仅报告不相邻连杆对；相邻关节嵌入不当作肢体碰撞。
                if m.body_parentid[b1]==b2 or m.body_parentid[b2]==b1: continue
                key='|'.join(sorted([m.body(b1).name,m.body(b2).name]))
                bad_pairs[key]=max(bad_pairs.get(key,0),float(-c.dist))
                maxpen=max(maxpen,float(-c.dist))
        self_penetration.append(maxpen)
        mujoco.mj_fullM(m,d,M)
        required=M@A[i]+d.qfrc_bias-d.qfrc_passive
        nj=3*len(cs); nc=nj+len(names)
        eq=np.zeros((m.nv,nc))
        for k,(b,p) in enumerate(cs):
            mujoco.mj_jac(m,d,J,Jr,p,b); eq[:,3*k:3*k+3]=J.T
        eq[dofs,nj+np.arange(len(names))]=1
        ub=[]
        # 方形摩擦锥 |fx|<=mu*fz, |fy|<=mu*fz；比真实圆锥宽，筛查偏乐观。
        for k in range(len(cs)):
            for axis in [0,1]:
                for sign in [-1,1]:
                    row=np.zeros(nc);row[3*k+axis]=sign;row[3*k+2]=-.8;ub.append(row)
        bounds=[(None,None) if j%3!=2 else (0,None) for j in range(nj)] + [(-float(e),float(e)) for e in effort]
        lp=linprog(np.zeros(nc),A_eq=eq,b_eq=required,
                   A_ub=np.array(ub) if ub else None,b_ub=np.zeros(len(ub)) if ub else None,
                   bounds=bounds,method='highs')
        feasible.append(bool(lp.success))
        relaxed=linprog(np.zeros(nc),A_eq=eq,b_eq=required,
                   A_ub=np.array(ub) if ub else None,b_ub=np.zeros(len(ub)) if ub else None,
                   bounds=bounds[:nj]+[(None,None)]*len(names),method='highs')
        root_feasible.append(bool(relaxed.success))
    footworld=np.array(footworld); minz=np.array(minz); com=np.array(com)
    footvel=np.gradient(footworld,dt,axis=0)
    # 排除接触切换差分跨界，单独测量连续支撑段内速度。
    interior=contact.copy();interior[0]=False;interior[-1]=False
    interior[1:-1]&=contact[:-2]&contact[2:]
    stride_intervals={}
    for j,s in enumerate(['left','right']):
        starts=np.array([a for a,b in runs(contact[:,j]) if a>0])
        stride_intervals[s]=(np.diff(starts)*dt).tolist()
    report.update({'foot_mesh_min_z_m':summary(minz),'stance_mesh_abs_z_m':summary(abs(minz[contact])),
                   'stance_anchor_speed_m_s':summary(np.linalg.norm(footvel,axis=2)[interior]),
                   'foot_target_max_error_m':float(max(footerr)),
                   'nonadjacent_self_penetration_frames':int(np.count_nonzero(self_penetration)),
                   'nonadjacent_collision_pairs_max_depth_m':bad_pairs,
                   'same_foot_stride_intervals_s':stride_intervals,
                   'optimistic_contact_force_screen':{'description':'M*a+b = J^T*f + tau; stance candidates; bounding-box corners; square friction pyramid mu=0.8; effort bounds 60/36 Nm. Pass is NOT dynamics certification.',
                       'feasible_frames':int(sum(feasible)), 'unbounded_joint_torque_feasible_frames':int(sum(root_feasible)), 'failed_frames':[i for i,v in enumerate(feasible) if not v]},
                   'root_centered_loop_joint_jump_rad':float(abs(q[0]-q[-1]).max()),
                   'loop_root_rotation_jump_deg':float(np.rad2deg((Rotation.from_quat(motion['root_quaternions_xyzw'][0]).inv()*Rotation.from_quat(motion['root_quaternions_xyzw'][-1])).magnitude())),
                   'loop_cross_boundary_20ms_joint_speed_rad_s':float(abs(q[0]-q[-1]).max()/dt)})
    # COM 静态投影仅诊断，动态走路 COM 可以离开支撑多边形。
    outside=[]
    for i in range(n):
        p=[]
        for j in range(2):
            if contact[i,j]:
                low,high=points[j].min(0),points[j].max(0)
                p.extend(footworld[i,j,:2]+np.array([[low[0],low[1]],[low[0],high[1]],[high[0],low[1]],[high[0],high[1]]])-centers[j][:2])
        h=ConvexHull(np.array(p)); outside.append(np.max(h.equations[:,:2]@com[i,:2]+h.equations[:,2]))
    report['static_com_outside_bbox_support_frames']=int(np.sum(np.array(outside)>0))
    report['static_com_support_halfspace_distance_m']=summary(outside)
    # 新建内存模型：使用 MuJoCo 内建 position 执行器，让隐式积分处理 PD 阻尼。
    # 不用 qfrc_applied 显式阻尼，避免小惯量关节的数值失稳污染筛查。
    xml=ET.parse(MODEL_PATH).getroot()
    compiler=xml.find('compiler');worldbody=xml.find('worldbody')
    if compiler is None or worldbody is None:
        raise ValueError('MJCF missing compiler/worldbody')
    compiler.set('meshdir',str((MODEL_PATH.parent/'../urdf/meshes').resolve()))
    ET.SubElement(worldbody,'geom',{'name':'audit_floor','type':'plane','size':'6 6 .1',
                  'contype':'1','conaffinity':'2','friction':'.8 .005 .0001'})
    for g in xml.findall('.//geom'):
        if g.get('name')=='audit_floor':continue
        if g.get('contype','1')!='0' or g.get('conaffinity','1')!='0':
            g.set('contype','2');g.set('conaffinity','1');g.set('friction','.8 .005 .0001')
    actuator=ET.SubElement(xml,'actuator')
    for name,p,k,e in zip(names,kp,kd,effort):
        ET.SubElement(actuator,'position',{'name':'audit_'+name,'joint':name,'kp':str(p),'kv':str(k),
                      'forcelimited':'true','forcerange':f'{-e} {e}'})
    m=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
    m.opt.timestep=.001;m.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST;m.opt.iterations=100
    tests=[]
    for feedforward in [False,True]:
        d=mujoco.MjData(m);d.qpos[:]=Q[0];d.qvel[:]=V[0]
        errs=[];saturations=[];heights=[];tilts=[];maxspeed=[]
        for step in range(int((n-1)*dt/.001)):
            t=step*.001;frame=min(int(t/dt),n-1)
            desiredvel=V[frame,dofs] if feedforward else np.zeros(len(names))
            tau=kp*(q[frame]-d.qpos[addresses])+kd*(desiredvel-d.qvel[dofs])
            saturations.append(float(np.mean(abs(tau)>effort)))
            d.ctrl[:]=q[frame]+kd/kp*desiredvel
            mujoco.mj_step(m,d)
            R=Rotation.from_quat(d.qpos[rootadr+3:rootadr+7][[1,2,3,0]]).as_matrix()
            tilt=float(np.rad2deg(np.arccos(np.clip(R[2,2],-1,1))))
            errs.append(float(np.sqrt(np.mean((d.qpos[addresses]-q[frame])**2))))
            heights.append(float(d.qpos[rootadr+2]));tilts.append(tilt);maxspeed.append(float(abs(d.qvel[dofs]).max()))
            if not np.isfinite(d.qpos).all() or d.qpos[rootadr+2]<.35 or tilt>60:break
        tests.append({'test':'diagnostic PD with desired velocity' if feedforward else 'training-like position PD with desired velocity zero',
                      'duration_s':float(d.time),'completed_full_clip':bool(d.time>=(n-1)*dt-1e-5),
                      'joint_rmse_rad':summary(errs),'max_torque_saturation_fraction':max(saturations),
                      'min_root_height_m':min(heights),'max_tilt_deg':max(tilts),'max_joint_speed_rad_s':max(maxspeed),
                      'note':'MuJoCo native position actuator implicit PD, 1ms, no velocity governor; NOT identical to PhysX implicit actuators with 5rad/s limit; root initialized from expert velocity; no policy trained.'})
    report['pd_tracking_screen']=tests
    report['contact_geometry_consistency'] = contact_geometry_report(
        np.asarray(minz), footvel, contact, dt)
    report['qualification'] = {
        'necessary_geometry_passed': report['contact_geometry_consistency']['passed'],
        'necessary_optimistic_force_screen_passed_all_frames': bool(all(feasible)),
        'joint_limits_and_speed_passed': bool(
            report['hard_limit_max_violation_rad'] < 1e-5 and
            report['speed_limit_5_rad_s_exceeded_frames'] == 0),
        # A reference is a state trajectory, NOT a position actuator command.
        # Plain-PD failure alone is not a proof that AMP cannot learn it. It IS
        # incorrect to call this dynamics-certified without an actual successful
        # physics tracking rollout and matching IsaacLab/AMP integration tests.
        'plain_pd_diagnostic_completed': bool(any(x['completed_full_clip'] for x in tests)),
        'dynamics_certified': False,
        'amp_training_ready': False,
        'status': 'NOT_APPROVED_FOR_TRAINING',
        'reason': 'Necessary screens are not a physical teacher rollout or an AMP integration test. Do not bypass this gate by flipping the NPZ flag.',
    }
    report['limitations']=['No AMP optimizer/integration tested yet','All force/contact screens are approximations, not full dynamics certification','Clip is not loopable: only sample adjacent frames within the clip','Robot self-collision disabled in training and PD screen; kinematic overlap still reported','Do not reinterpret valid_for_training=false as true automatically']
    runtime_path=args.output/'isaac_runtime_audit.json'
    if runtime_path.exists():
        runtime=json.loads(runtime_path.read_text())
        if 'runtime_robot' in runtime:
            rd=runtime['runtime_robot'];bi={n:j for j,n in enumerate(rd['body_names'])};ji={n:j for j,n in enumerate(rd['joint_names'])}
            comparisons={}
            for b in range(1,m.nbody):
                name=m.body(b).name
                R=Rotation.from_quat(m.body_iquat[b][[1,2,3,0]]).as_matrix()
                inertia=R@np.diag(m.body_inertia[b])@R.T
                comparisons[name]={'mass_abs_difference_kg':float(abs(m.body_mass[b]-rd['default_mass'][0][bi[name]])),
                   'link_frame_inertia_max_abs_difference_kg_m2':float(abs(inertia-np.array(rd['default_inertia'][0][bi[name]]).reshape(3,3)).max())}
            alignment={'body_parameter_comparison':comparisons,
                'max_mass_difference_kg':max(v['mass_abs_difference_kg'] for v in comparisons.values()),
                'max_link_frame_inertia_difference_kg_m2':max(v['link_frame_inertia_max_abs_difference_kg_m2'] for v in comparisons.values()),
                'max_joint_limit_difference_rad':max(float(abs(m.jnt_range[m.joint(name).id]-np.array(rd['joint_pos_limits'][0][ji[name]])).max()) for name in names),
                'explicit_action_order_matches_expert':runtime['action_names']==names,
                'runtime_internal_order_differs_from_action_order':rd['joint_names']!=names,
                'max_sampled_USD_MJCF_foot_anchor_difference_m':max((v['max_abs_foot_anchor_error_m'] for v in runtime.get('usd_vs_mjcf_fk_samples',[])),default=None),
                'note':'Mass/inertia/limits and 7 kinematic poses compared. Collision shapes/solver/friction response are not thereby identical.'}
            (args.output/'model_alignment.json').write_text(json.dumps(alignment,ensure_ascii=False,indent=2))
    (args.output/'motion_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    np.savez_compressed(args.output/'audit_arrays.npz',qpos=Q,qvel=V,qacc=A,foot_anchor_world=footworld,foot_centers_body=centers,foot_mesh_min_z=minz,com_world=com,force_feasible=feasible)
    print(json.dumps({k:report[k] for k in ['joint_speed_abs_rad_s','speed_limit_5_rad_s_exceeded_frames','nonadjacent_self_penetration_frames','optimistic_contact_force_screen','pd_tracking_screen']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
