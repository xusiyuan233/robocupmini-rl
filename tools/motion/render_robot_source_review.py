"""Render robot-source poses vs target ACTUAL physics. Never certifies a clip.
Top uses linked source timestamps, not native source speed. Bottom uses unchanged
1kHz audited physics sampled at 50Hz. Lateral cameras are fixed, no looping.
"""
from pathlib import Path
import argparse,json,subprocess
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw,ImageFont
from reference_coordination_quality import from_audit
from play_retarget_motion import build_model,joint_mapping,MODEL_PATH
ROOT=Path(__file__).resolve().parents[2]

def render(args):
    if args.output.exists():raise FileExistsError(args.output)
    for p in [args.audit,args.output,args.source_model]:
        if not p.resolve().is_relative_to(ROOT) or 'lilin' in str(p).lower():raise ValueError('Outside user project')
    a,decision=from_audit(args.audit);args.output.mkdir(parents=True)
    m=build_model(MODEL_PATH);d=mj.MjData(m);_,adr,_=joint_mapping(m,a['joint_names'].tolist())
    sm=mj.MjModel.from_xml_path(str(args.source_model));sd=mj.MjData(sm)
    sm.vis.global_.offwidth=480;sm.vis.global_.offheight=360
    cam=mj.MjvCamera();mj.mjv_defaultCamera(cam);cam.distance=1.45;cam.elevation=-9
    opt=mj.MjvOption();mj.mjv_defaultOption(opt);opt.geomgroup[3]=1
    target_opt=mj.MjvOption();mj.mjv_defaultOption(target_opt);target_opt.geomgroup[3]=0
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    stretch=float(np.median(np.diff(a['time_s']))/np.median(np.diff(a['source_time_s'])))
    src=a['source_robot_root_positions_m'];n=len(a['time_s']);sy=float(src[:,1].mean());ty=float(a['root_positions_m'][:,1].mean())
    quality=json.loads(args.quality.read_text()) if args.quality else None
    target=args.output/'berkeley_to_robotcupmini_physics.mp4'
    process=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-video_size','1440x890','-framerate','25','-i','-','-c:v','libx264','-crf','19','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
    with mj.Renderer(m,height=360,width=480) as rr,mj.Renderer(sm,height=360,width=480) as sr:
        def frame(i):
            sd.qpos[:3]=src[i];sd.qpos[3:7]=a['source_robot_root_quaternions_xyzw'][i,[3,0,1,2]];sd.qpos[7:]=a['source_robot_joint_positions_rad'][i];sd.qvel[:]=0.;mj.mj_forward(sm,sd)
            d.qpos[:3]=a['root_positions_m'][i];d.qpos[3:7]=a['root_quaternions_xyzw'][i,[3,0,1,2]];d.qpos[adr]=a['joint_positions_rad'][i];d.qvel[:]=0.;mj.mj_forward(m,d)
            im=Image.new('RGB',(1440,890),'white');draw=ImageDraw.Draw(im)
            draw.text((10,5),f'Berkeley-policy source -> ROBOCUPMINI | recorded target t={a["time_s"][i]:.2f}s | source t={a["source_time_s"][i]:.3f}s',font=font,fill='black')
            draw.text((10,29),f'Physics: {decision["status"]} | style: {quality["status"] if quality else "NOT YET SCREENED"}',font=font,fill='black')
            draw.text((10,53),f'TOP: published Berkeley policy, simplified visuals; linked source time (stretch {stretch:.1f}x), not paper-video checkpoint.',font=font,fill='black')
            for k,az in enumerate([0.,90.,45.]):
                cam.azimuth=az;cam.lookat[:]=[src[i,0],sy,.48];sr.update_scene(sd,camera=cam,scene_option=opt);im.paste(Image.fromarray(sr.render()),(k*480,80))
                cam.lookat[:]=[d.qpos[0],ty,.50];rr.update_scene(d,camera=cam,scene_option=target_opt);im.paste(Image.fromarray(rr.render()),(k*480,490))
            draw.text((10,450),'BOTTOM: actual target-robot physics, not IK animation / not a trained policy. FRONT / SIDE / OBLIQUE.',font=font,fill='black')
            draw.text((10,858),'Fixed lateral cameras. No state filtering, no root forces. Passing this clip is NOT IsaacLab or real-robot approval.',font=font,fill='black')
            return im
        try:
            for i in range(0,n,2):process.stdin.write(frame(i).tobytes())
        finally:process.stdin.close()
        if process.wait():raise RuntimeError('ffmpeg failed')
        indices=np.arange(0,n,20);sheet=Image.new('RGB',(1440,((len(indices)+2)//3)*297),'white')
        for k,i in enumerate(indices):sheet.paste(frame(int(i)).resize((480,297)),((k%3)*480,(k//3)*297))
        sheet.save(args.output/'full_clip_review_sheet.jpg')
    print(target)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--quality',type=Path);p.add_argument('--source-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);render(p.parse_args())
