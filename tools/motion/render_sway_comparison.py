"""Compare recorded physics with fixed lateral cameras (never re-simulates).

Each view follows forward travel but NOT lateral pelvis motion. Without this,
a pelvis-following camera can hide the very lateral sway we are measuring.
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, cast
import numpy as np
import mujoco as _mj
from PIL import Image, ImageDraw, ImageFont
from play_retarget_motion import build_model, MODEL_PATH
mj = cast(Any, _mj)


def render(baseline: Path, candidate: Path, output: Path, baseline_label="V13 baseline", candidate_label="Reduced-sway candidate"):
    for path in (baseline, candidate, output):
        if 'lilin' in str(path.resolve()).lower(): raise ValueError('Outside user project scope')
    if output.exists(): raise FileExistsError('Refusing output overwrite')
    clips=[]; labels=[]
    for audit in (baseline, candidate):
        decision=json.loads((audit/'decision.json').read_text())
        with np.load(audit/'physics_replay.npz',allow_pickle=False) as a:
            states=a['states_50hz'].copy()
        clips.append(states)
        labels.append('physics checks passed' if decision['dynamics_certified_in_this_mujoco_contract'] else 'DRAFT: physics checks FAILED')
    output.mkdir(parents=True,exist_ok=False)
    m=build_model(MODEL_PATH);d=mj.MjData(m)
    opt=mj.MjvOption();mj.mjv_defaultOption(opt);opt.geomgroup[3]=0
    cam=mj.MjvCamera();mj.mjv_defaultCamera(cam);cam.distance=1.7;cam.elevation=-9
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    w,h=960,840
    path=output/'recorded_physics_comparison.mp4'
    ff=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
                         '-video_size',f'{w}x{h}','-framerate','25','-i','-',
                         '-c:v','libx264','-crf','19','-preset','fast','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)
    duration=min(float(x[-1,0]) for x in clips)
    with mj.Renderer(m,height=360,width=480) as renderer:
        def frame(t):
            canvas=Image.new('RGB',(w,h),'white');draw=ImageDraw.Draw(canvas)
            draw.text((8,5),f'RECORDED PHYSICS | t={t:.2f}s | fixed lateral camera, 1x playback',font=font,fill='black')
            for col,(states,label) in enumerate(zip(clips,labels)):
                k=int(np.argmin(abs(states[:,0]-t)));q=states[k,1:21]
                d.qpos[:]=q;d.qvel[:]=0;mj.mj_forward(m,d)
                y0=float(states[:,2].mean())
                draw.text((480*col+8,32),(baseline_label if col==0 else candidate_label),font=font,fill='black')
                draw.text((480*col+8,54),label,font=font,fill='black')
                for row,az in enumerate((0.,90.)):
                    cam.azimuth=az;cam.lookat[:]=[q[0],y0,.47]
                    renderer.update_scene(d,camera=cam,scene_option=opt)
                    canvas.paste(Image.fromarray(renderer.render()),(480*col,85+row*370))
                draw.text((480*col+8,816),f'pelvis Y relative to mean: {(q[1]-y0)*100:+.1f} cm',font=font,fill='black')
            return canvas
        if ff.stdin is None: raise RuntimeError('Video encoder pipe unavailable')
        try:
            for t in np.arange(0,duration+1e-6,.04): ff.stdin.write(frame(float(t)).tobytes())
        finally: ff.stdin.close()
        if ff.wait(): raise RuntimeError('Video encoder failed')
        times=[1.46,3.0,4.46,6.0]
        sheet=Image.new('RGB',(960,840),'white')
        for i,t in enumerate(times):
            im=frame(min(t,duration));sheet.paste(im.resize((480,420)),((i%2)*480,(i//2)*420))
        # Also retain full-resolution individual frames, for motion-phase QA.
        for i,t in enumerate(times): frame(min(t,duration)).save(output/f'phase_{i}.png')
        sheet.save(output/'comparison_sheet.png')
    print(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-audit',type=Path,required=True)
    p.add_argument('--candidate-audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline-label',default='V13 baseline');p.add_argument('--candidate-label',default='Reduced-sway candidate')
    a=p.parse_args();render(a.baseline_audit,a.candidate_audit,a.output,a.baseline_label,a.candidate_label)
if __name__=='__main__': main()
