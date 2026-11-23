"""Render RECORDED physics states, not an IK animation or a policy inference.

This viewer never certifies motion. Failed/partial audit is labelled DRAFT.
Uses a display-only MJCF with the same qpos mapping; actual physics is audited
independently by audit_physical_reference.py. EGL offline rendering only.
"""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
from typing import Any,cast
import numpy as np
import mujoco as _mj
from PIL import Image,ImageDraw,ImageFont
from play_retarget_motion import build_model,MODEL_PATH
mj=cast(Any,_mj)


def render(audit:Path,output:Path):
    if output.exists():raise FileExistsError('Refusing to overwrite video output')
    if 'lilin' in str(audit.resolve()).lower() or 'lilin' in str(output.resolve()).lower():raise ValueError('Outside scope')
    decision=json.loads((audit/'decision.json').read_text());data=np.load(audit/'physics_replay.npz',allow_pickle=False)
    states=data['states_50hz'];dt=float(np.median(np.diff(states[:,0])));Q=states[:,1:21]
    output.mkdir(parents=True,exist_ok=False)
    m=build_model(MODEL_PATH);d=mj.MjData(m);opt=mj.MjvOption();mj.mjv_defaultOption(opt);opt.geomgroup[3]=0
    cam=mj.MjvCamera();mj.mjv_defaultCamera(cam);cam.distance=1.7;cam.elevation=-9
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    passed=bool(decision['dynamics_certified_in_this_mujoco_contract'])
    width,height=960,430
    path=output/'recorded_physics_front_side.mp4'
    command=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-video_size',f'{width}x{height}',
             '-framerate',str(1/(2*dt)),'-i','-','-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p',str(path)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    with mj.Renderer(m,height=360,width=480) as renderer:
        def frame(i:int):
            d.qpos[:]=Q[i];d.qvel[:]=0.;mj.mj_forward(m,d)
            canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
            title='PASSED MuJoCo clip test' if passed else 'DRAFT - not approved'
            draw.text((10,6),f'{title} | RECORDED PHYSICS | t={states[i,0]:.2f}s',font=font,fill='black')
            draw.text((10,29),'Free root, gravity + contact, native effort-limited PD. Visualization is not training.',font=font,fill='black')
            for j,az in enumerate([0.,90.]):
                cam.azimuth=az;cam.lookat[:]=Q[i,:3];cam.lookat[2]=.47
                renderer.update_scene(d,camera=cam,scene_option=opt)
                canvas.paste(Image.fromarray(renderer.render()),(480*j,65))
            return canvas
        if process.stdin is None:raise RuntimeError('ffmpeg pipe unavailable')
        try:
            for i in range(0,len(Q),2):process.stdin.write(frame(i).tobytes())
        finally:process.stdin.close()
        if process.wait():raise RuntimeError('ffmpeg rendering failed')
        indices=np.linspace(0,len(Q)-1,4,dtype=int)
        sheet=Image.new('RGB',(1920,800),'white')
        for col,i in enumerate(indices):
            im=frame(int(i));sheet.paste(im.crop((0,65,480,425)),(480*col,40));sheet.paste(im.crop((480,65,960,425)),(480*col,440))
            draw=ImageDraw.Draw(sheet);draw.text((480*col+8,8),f't={states[i,0]:.2f}s | front',font=font,fill='black');draw.text((480*col+8,408),'side | recorded physical state',font=font,fill='black')
        sheet.save(output/'contact_sheet.png')
    print(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();render(args.audit,args.output)

if __name__=='__main__':main()
