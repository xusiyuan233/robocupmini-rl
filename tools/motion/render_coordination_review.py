"""Source human + whole recorded robot physics, no smoothing or phase alignment.

Columns front/side/oblique, top human, bottom robot. Cameras do not follow
lateral sway. Labels always show failed gates; this renderer never certifies.
"""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
from typing import Any,cast
import numpy as np
from scipy.spatial.transform import Rotation
from bvh_motion import anatomical_frame
import mujoco as _mj
from PIL import Image,ImageDraw,ImageFont
from play_retarget_motion import build_model,MODEL_PATH,joint_mapping
from render_human_retarget import human_model
from reference_coordination_quality import from_audit
mj=cast(Any,_mj)


def plot_time_review(a, audit:Path, output:Path):
    """Whole-clip curves of the unchanged recorded states, with no phase fitting."""
    t=a['time_s'];root=Rotation.from_quat(a['root_quaternions_xyzw'])
    waist=a['joint_positions_rad'][:,a['joint_names'].tolist().index('waist_yaw_joint')]
    torso=root*Rotation.from_euler('z',waist)
    hn=a['human_joint_names'].tolist();hp=a['human_positions_m']
    at=lambda name:hp[:,hn.index(name)]
    source=Rotation.from_matrix(anatomical_frame(at('LeftShoulder'),at('RightShoulder'),at('Neck')-at('Spine2')))
    actual=np.rad2deg(np.unwrap(torso.as_euler('xyz'),axis=0))
    human=np.rad2deg(np.unwrap(source.as_euler('xyz'),axis=0))
    # Only constant orientation offsets are removed; amplitudes/phases untouched.
    actual-=actual.mean(0);human-=human.mean(0)
    p=a['root_positions_m'];sway=100*(p[:,1]-np.polyval(np.polyfit(t,p[:,1],1),t))
    separation=100*root.inv().apply(a['foot_targets_world_m'][:,0]-a['foot_targets_world_m'][:,1])[:,1]
    with np.load(audit/'physics_replay.npz',allow_pickle=False) as ar:
        clearance=ar['mesh_min_z_m']*100
    panels=[('Chest roll [deg], mean removed',[(actual[:,0],'robot','#1565c0'),(human[:,0],'human source','#e67700')]),
            ('Chest yaw [deg], mean removed',[(actual[:,2],'robot','#1565c0'),(human[:,2],'human source','#e67700')]),
            ('Pelvis lateral position [cm], linear drift removed',[(sway,'robot','#1565c0')]),
            ('Foot width [cm], measured in root coordinates',[(separation,'robot','#1565c0'),(np.full(len(t),15.),'15cm nominal lower band','#888888'),(np.full(len(t),19.),'19cm nominal upper band','#b0b0b0')]),
            ('Sole geometry minimum height [cm] (negative = penetration)',[(np.r_[clearance[0,0],clearance[:,0]],'left','#1565c0'),(np.r_[clearance[0,1],clearance[:,1]],'right','#e67700')])]
    im=Image.new('RGB',(1440,1210),'white');draw=ImageDraw.Draw(im)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    draw.text((15,7),'FULL-CLIP RECORDED PHYSICS REVIEW | no filtering, no phase shift | human vs 13-joint robot',fill='black',font=font)
    for k,(title,series) in enumerate(panels):
        top=65+k*228;left,right=85,1405;bottom=top+157
        values=np.concatenate([np.asarray(v) for v,_,_ in series]);lo,hi=float(values.min()),float(values.max())
        pad=max((hi-lo)*.12,.2);lo-=pad;hi+=pad
        draw.text((15,top-25),title,fill='black',font=font)
        for f in np.linspace(0,1,5):
            y=bottom-f*(bottom-top);draw.line([(left,y),(right,y)],fill='#dddddd')
            draw.text((4,y-10),f'{lo+f*(hi-lo):.1f}',fill='black',font=font)
        for tm in np.arange(0,float(t[-1])+1e-6,1):
            x=left+(right-left)*tm/t[-1];draw.line([(x,top),(x,bottom)],fill='#eeeeee')
            draw.text((x-7,bottom+3),f'{tm:.0f}',fill='black',font=font)
        for n,(v,label,color) in enumerate(series):
            xy=list(zip(left+(right-left)*t/t[-1],bottom-(np.asarray(v)-lo)/(hi-lo)*(bottom-top)))
            draw.line(xy,fill=color,width=2)
            draw.text((left+n*365,bottom+26),label,fill=color,font=font)
    im.save(output/'whole_clip_time_review.png')


def render(audit:Path,quality:Path,output:Path):
    for p in (audit,quality,output):
        if 'lilin' in str(p.resolve()).lower():raise ValueError('Outside user project')
    if output.exists():raise FileExistsError(output)
    a,decision=from_audit(audit);report=json.loads(quality.read_text())
    output.mkdir(parents=True,exist_ok=False)
    plot_time_review(a,audit,output)
    m=build_model(MODEL_PATH);d=mj.MjData(m);_,adr,_=joint_mapping(m,a['joint_names'].tolist())
    hm=human_model();hd=mj.MjData(hm);mj.mj_forward(hm,hd)
    opt=mj.MjvOption();mj.mjv_defaultOption(opt);opt.geomgroup[3]=0
    cam=mj.MjvCamera();mj.mjv_defaultCamera(cam);cam.distance=1.75;cam.elevation=-10
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    human=a['human_positions_m']*.6078721595500206
    # Whole-body scale, one fixed vertical floor fit, NOT per-frame foot snapping.
    names=a['human_joint_names'].tolist();footids=[names.index(x) for x in ['LeftFoot','RightFoot']]
    human[:,:,2]-=float(np.percentile(human[:,footids,2],5))
    parents=a['human_parents'];t=a['time_s'];dt=float(a['sample_dt_s'])
    colors=[np.array([.2,.6,.95,1.]) if n.startswith('Left') else np.array([.95,.55,.2,1.]) if n.startswith('Right') else np.array([.85,.85,.85,1.]) for n in names]
    target=output/'source_human_vs_recorded_robot.mp4'
    ff=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
                         '-video_size','1440x870','-framerate','25','-i','-','-c:v','libx264',
                         '-crf','19','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
    states_center=float(a['root_positions_m'][:,1].mean());human_center=float(human[:,0,1].mean())
    passed=bool(decision['dynamics_certified_in_this_mujoco_contract'])
    with mj.Renderer(m,height=360,width=480) as rr,mj.Renderer(hm,height=360,width=480) as hr:
        def panels(i):
            d.qpos[:3]=a['root_positions_m'][i];d.qpos[3:7]=a['root_quaternions_xyzw'][i,[3,0,1,2]]
            d.qpos[adr]=a['joint_positions_rad'][i];d.qvel[:]=0;mj.mj_forward(m,d)
            hp=[];rp=[]
            for az in [0.,90.,45.]:
                cam.azimuth=az;cam.lookat[:]=[human[i,0,0],human_center,.5]
                hr.update_scene(hd,camera=cam)
                for j,p in enumerate(parents):
                    if p<0:continue
                    k=hr.scene.ngeom
                    mj.mjv_initGeom(hr.scene.geoms[k],mj.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).flatten(),colors[j])
                    mj.mjv_connector(hr.scene.geoms[k],mj.mjtGeom.mjGEOM_CAPSULE,.009,human[i,p],human[i,j]);hr.scene.ngeom+=1
                hp.append(Image.fromarray(hr.render()))
                cam.lookat[:]=[d.qpos[0],states_center,.5]
                rr.update_scene(d,camera=cam,scene_option=opt);rp.append(Image.fromarray(rr.render()))
            return hp,rp
        def frame(i):
            im=Image.new('RGB',(1440,870),'white');draw=ImageDraw.Draw(im)
            draw.text((8,5),f'Human-source posture review | playback {t[i]:.2f}s | exact source {a["source_time_s"][i]:.3f}s | 1x recorded time',font=font,fill='black')
            draw.text((8,28),f'Physics: {"PASS" if passed else "FAIL / PARTIAL"} | motion-quality failed checks: {report["failed_checks"]}',font=font,fill='black')
            draw.text((8,51),'TOP: measured human, uniformly scaled. BOTTOM: recorded robot, NOT IK or a trained policy. Front / side / oblique.',font=font,fill='black')
            hs,rs=panels(i)
            for c in range(3):im.paste(hs[c],(480*c,80));im.paste(rs[c],(480*c,475))
            draw.text((8,448),'Same source timestamps; robot has no arm/spine roll-pitch/toe joints. No lateral camera tracking.',font=font,fill='black')
            draw.text((8,842),f'BLUE left / ORANGE right | source time stretch {report["measured"]["source_time_stretch"]:.2f} | NOT a human-equivalence certificate',font=font,fill='black')
            return im
        if ff.stdin is None:raise RuntimeError('Video pipe unavailable')
        try:
            for i in range(0,len(t),2):ff.stdin.write(frame(i).tobytes())
        finally:ff.stdin.close()
        if ff.wait():raise RuntimeError('Video render failed')
        # Review the ENTIRE clip in 0.4s slices, not four favourable snapshots.
        ix=np.arange(0,len(t),20)
        rows=(len(ix)+5)//6
        sheet=Image.new('RGB',(1920,rows*285),'white')
        for k,i in enumerate(ix):
            hs,rs=panels(int(i));small=Image.new('RGB',(320,285),'white');dr=ImageDraw.Draw(small)
            dr.text((5,2),f't={t[i]:.2f}s',font=font,fill='black')
            # Front + side for both human and robot in each temporal tile.
            small.paste(hs[0].resize((160,120)),(0,25));small.paste(hs[1].resize((160,120)),(160,25))
            small.paste(rs[0].resize((160,120)),(0,150));small.paste(rs[1].resize((160,120)),(160,150))
            sheet.paste(small,((k%6)*320,(k//6)*285))
        sheet.save(output/'whole_clip_review_sheet.png')
        for k,i in enumerate(np.linspace(0,len(t)-1,8,dtype=int)):
            frame(int(i)).save(output/f'review_phase_{k}.png')
    print(target)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--quality',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.audit,a.quality,a.output)
if __name__=='__main__':main()
