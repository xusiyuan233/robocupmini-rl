"""仅用于本用户项目的一环境接入检查和参考 PD 筛查；不调用 learn/update。
先运行 audit_style_pretraining.py，再用 env_isaaclab 执行本文件。
"""
import sys, json, os, traceback, argparse
from typing import cast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'rsl_rl'))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--motion',type=Path,default=ROOT/'data/motions/robotcupmini/retargeted/walk1_candidate_v3_coordination/candidate_motion.npz')
parser.add_argument('--output',type=Path,default=ROOT/'data/motions/robotcupmini/validated/pretraining_audit_20261118')
parser.add_argument('--max-reference-steps',type=int,default=150)
ARGS=parser.parse_args()
OUT=ARGS.output
OUT.mkdir(parents=True,exist_ok=True)
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device='cuda:0').app
result={'audit_only':True,'optimizer_steps':0}
try:
 import torch, gymnasium as gym
 import numpy as np
 import bipedal_locomotion
 from isaaclab.envs import ManagerBasedRLEnv
 from isaaclab_tasks.utils import parse_env_cfg
 from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
 from bipedal_locomotion.tasks.locomotion.agents.robotcupmini_ppo_cfg import RobotcupMiniStylePrepPPORunnerCfg
 from rsl_rl.runner import OnPolicyRunner
 import rsl_rl.runner.on_policy_runner as rr
 from pxr import Usd,UsdPhysics
 cfg=parse_env_cfg('Robocup-Style-Prep-v0',device='cuda:0',num_envs=1)
 result['runner_path']=rr.__file__
 result['trial_env']=os.environ.get('ROBOCUP_GAIT_TRIAL','off')
 result['task']='Robocup-Style-Prep-v0'
 cfg.seed=42
 result['config']={'physics_dt':cfg.sim.dt,'decimation':cfg.decimation,'default_num_envs_in_class':256,'tested_num_envs':1,'episode_length_s':cfg.episode_length_s,'gait_ranges':cfg.commands.gait_command.ranges.to_dict(),'velocity':cfg.commands.base_velocity.to_dict(),'rewards':cfg.rewards.to_dict(),'events':cfg.events.to_dict(),'terminations':cfg.terminations.to_dict()}
 stage=Usd.Stage.Open(cfg.scene.robot.spawn.usd_path)
 result['usd']={'path':cfg.scene.robot.spawn.usd_path,'bodies':[],'joints':[]}
 for p in stage.Traverse():
  if p.HasAPI(UsdPhysics.MassAPI):
   ma=UsdPhysics.MassAPI(p)
   result['usd']['bodies'].append({'name':p.GetName(),'mass':ma.GetMassAttr().Get(),'com':list(ma.GetCenterOfMassAttr().Get()),'inertia':list(ma.GetDiagonalInertiaAttr().Get())})
  if p.IsA(UsdPhysics.RevoluteJoint):
   j=UsdPhysics.RevoluteJoint(p)
   result['usd']['joints'].append({'name':p.GetName(),'lower_deg':j.GetLowerLimitAttr().Get(),'upper_deg':j.GetUpperLimitAttr().Get(),'axis':j.GetAxisAttr().Get()})
 env=gym.make('Robocup-Style-Prep-v0',cfg=cfg)
 base=cast(ManagerBasedRLEnv,env.unwrapped)
 robot=base.scene['robot']
 result['runtime_robot']={'joint_names':robot.joint_names,'body_names':robot.body_names}
 for key in ['default_joint_pos','joint_pos_limits','soft_joint_pos_limits','joint_vel_limits','joint_effort_limits','default_mass','default_inertia']:
  val=getattr(robot.data,key,None)
  if val is not None: result['runtime_robot'][key]=val.detach().cpu().tolist()
 result['runtime_rewards']=base.reward_manager.active_terms
 wrapper=RslRlVecEnvWrapper(env)
 obs=wrapper.get_observations()
 result['observation_shapes']={k:list(v.shape) for k,v in obs.items()}
 result['action_names']=base.action_manager.get_term('joint_pos')._joint_names
 cfg_agent=RobotcupMiniStylePrepPPORunnerCfg()
 runner=OnPolicyRunner(wrapper,cfg_agent.to_dict(),log_dir=None,device='cuda:0')
 result['algorithm_class']=type(runner.alg).__name__
 result['actor_shape']=str(runner.alg.actor_critic.actor)
 result['encoder_shape']=str(runner.alg.encoder)
 steps=[]
 for i in range(16):
  obs,rews,dones,info=wrapper.step(torch.zeros((1,13),device='cuda:0'))
  steps.append({'step':i,'finite_obs':all(torch.isfinite(v).all().item() for v in obs.values()),'reward':rews.cpu().tolist(),'done':dones.cpu().tolist()})
 result['zero_action_smoke_steps']=steps
 result['status']='passed_config_environment_runner_smoke_no_learning'
 # 追加：同一 USD 下的有限时长参考关节 PD 筛查，不反向传播。
 motion=np.load(ARGS.motion,allow_pickle=False)
 result['reference_motion']=str(ARGS.motion.resolve())
 arrays=np.load(OUT/'audit_arrays.npz')
 names=motion['joint_names'].tolist()
 idx=torch.tensor([robot.joint_names.index(n) for n in names],device='cuda:0')
 result['reference_control_action_clipping']=str(getattr(wrapper,'clip_actions',None))
 from isaaclab.utils.math import quat_apply
 feetids=torch.tensor([robot.body_names.index(s+'_ankle_roll_link') for s in ['left','right']],device='cuda:0')
 centers=torch.as_tensor(arrays['foot_centers_body'],device='cuda:0',dtype=torch.float32)
 fks=[]
 for frame in np.linspace(0,len(motion['time_s'])-1,7,dtype=int).tolist():
  state=robot.data.default_root_state.clone();state[0,:3]=torch.as_tensor(arrays['qpos'][frame,:3],device='cuda:0')+base.scene.env_origins[0]
  state[0,3:7]=torch.as_tensor(arrays['qpos'][frame,3:7],device='cuda:0');state[:,7:]=0
  robot.write_root_state_to_sim(state)
  joints=robot.data.default_joint_pos.clone();joints[:,idx]=torch.as_tensor(motion['joint_positions_rad'][frame],device='cuda:0',dtype=torch.float32)
  robot.write_joint_state_to_sim(joints,torch.zeros_like(joints))
  base.sim.forward();base.scene.update(base.physics_dt)
  feet=robot.data.body_link_pos_w[0,feetids]+quat_apply(robot.data.body_link_quat_w[0,feetids],centers)-base.scene.env_origins[0]
  err=float(abs(feet.detach().cpu().numpy()-arrays['foot_anchor_world'][frame]).max())
  fks.append({'frame':frame,'max_abs_foot_anchor_error_m':err})
 result['usd_vs_mjcf_fk_samples']=fks
 base.reset(seed=42)
 root_state=robot.data.default_root_state.clone()
 root_state[0,:3]=torch.as_tensor(arrays['qpos'][0,:3],device='cuda:0')+base.scene.env_origins[0]
 root_state[0,3:7]=torch.as_tensor(arrays['qpos'][0,3:7],device='cuda:0')
 root_state[0,7:10]=torch.as_tensor(arrays['qvel'][0,:3],device='cuda:0')
 from isaaclab.utils.math import quat_apply
 root_state[:,10:13]=quat_apply(root_state[:,3:7],torch.as_tensor(arrays['qvel'][0,3:6],device='cuda:0',dtype=torch.float32).unsqueeze(0))
 robot.write_root_state_to_sim(root_state)
 jp=robot.data.default_joint_pos.clone();jv=torch.zeros_like(jp)
 jp[:,idx]=torch.as_tensor(motion['joint_positions_rad'][0],device='cuda:0',dtype=torch.float32)
 jv[:,idx]=torch.as_tensor(arrays['qvel'][0,6:],device='cuda:0',dtype=torch.float32)
 robot.write_joint_state_to_sim(jp,jv)
 base.sim.forward();base.scene.update(base.physics_dt)
 reference_steps=[]
 for i in range(min(ARGS.max_reference_steps,len(motion['time_s']))):
  target=torch.as_tensor(motion['joint_positions_rad'][i],device='cuda:0',dtype=torch.float32).unsqueeze(0)
  action=(target-robot.data.default_joint_pos[:,idx])/.15
  obs,rw,done,info=wrapper.step(action)
  qq=robot.data.root_quat_w[0]
  up=quat_apply(qq.unsqueeze(0),torch.tensor([[0.,0.,1.]],device='cuda:0'))[0,2]
  reference_steps.append({'time_s':(i+1)*base.step_dt,'height_m':float(robot.data.root_pos_w[0,2]),
       'tilt_deg':float(torch.rad2deg(torch.acos(up.clamp(-1,1)))),'done':bool(done[0]),
       'joint_rmse_rad':float(((robot.data.joint_pos[:,idx]-target)**2).mean().sqrt())})
  if bool(done[0]):break
 result['isaac_reference_pd_screen']={'note':'Reference joint targets, not policy. Root initialized from motion. Manager auto-reset means the final done-frame pose may be reset state.',
   'steps':reference_steps,'terminated':reference_steps[-1]['done']}
 checkpoint=ROOT/'logs/rsl_rl/robotcupmini_velocity/2026-11-09_16-30-32_forward_gait_v18_stance_yaw/model_15200.pt'
 if checkpoint.exists():
  cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
  expected=runner.alg.actor_critic.state_dict()
  result['checkpoint_shape_check']={'path':str(checkpoint),'iteration':cp.get('iter'),
      'mismatches':[k for k,v in cp['model_state_dict'].items() if k not in expected or v.shape!=expected[k].shape],
      'missing':[k for k in expected if k not in cp['model_state_dict']],
      'encoder_all_shapes_match':all(k in runner.alg.encoder.state_dict() and v.shape==runner.alg.encoder.state_dict()[k].shape for k,v in cp['encoder_state_dict'].items()),
      'note':'Shape compatibility only, not gait quality or recommended baseline.'}
 wrapper.close()
except Exception:
 result['status']='failed'; result['traceback']=traceback.format_exc();print(result['traceback'])
finally:
 (OUT/'isaac_runtime_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str))
 app.close()
