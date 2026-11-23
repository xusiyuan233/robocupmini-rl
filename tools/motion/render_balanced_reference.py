"""离线渲染参考动作和 v3 对比；只做运动学回放，不启动策略或训练。"""
import argparse
from pathlib import Path
import subprocess
from typing import Any, cast

import mujoco as _mujoco
import numpy as np
from PIL import Image, ImageDraw

from play_retarget_motion import (build_model, MODEL_PATH, joint_mapping,
                                  load_motion, apply_frame)

mujoco = cast(Any, _mujoco)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEW = ROOT/'data/motions/robotcupmini/retargeted/walk1_candidate_v4_balanced/candidate_motion.npz'
DEFAULT_OLD = ROOT/'data/motions/robotcupmini/retargeted/walk1_candidate_v3_coordination/candidate_motion.npz'
DEFAULT_OUT = ROOT/'data/motions/robotcupmini/validated/walk1_candidate_v4_balanced/visuals'


def video(path, frames, fps):
    width, height = frames[0].size
    command = ['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo',
               '-pixel_format','rgb24','-video_size',f'{width}x{height}',
               '-framerate',str(fps),'-i','-','-c:v','libx264','-crf','20',
               '-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError('ffmpeg stdin unavailable')
    try:
        for frame in frames:
            process.stdin.write(frame.convert('RGB').tobytes())
    finally:
        process.stdin.close()
    if process.wait():
        raise RuntimeError(f'ffmpeg failed: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion',type=Path,default=DEFAULT_NEW)
    parser.add_argument('--old-motion',type=Path,default=DEFAULT_OLD)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    new,names,dt = load_motion(args.motion)
    old,oldnames,old_dt = load_motion(args.old_motion)
    if oldnames != names:
        raise ValueError('Joint names mismatch')
    model = build_model(MODEL_PATH)
    data = mujoco.MjData(model)
    _,addresses,root = joint_mapping(model,names)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 1.6
    camera.elevation = -9.0
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[3] = 0
    with mujoco.Renderer(model,height=400,width=400) as renderer:
        def panel(motion,frame,angle):
            apply_frame(model,data,motion,addresses,root,frame)
            camera.azimuth = angle
            camera.lookat[:] = data.qpos[root:root+3]
            camera.lookat[2] -= .03
            renderer.update_scene(data,camera=camera,scene_option=option)
            return Image.fromarray(renderer.render())
        # 全段、正常时间速度的新参考回放。
        frames = []
        for frame in range(0,len(new['time_s']),2):
            image = Image.new('RGB',(800,450),'white')
            image.paste(panel(new,frame,0),(0,50))
            image.paste(panel(new,frame,55),(400,50))
            draw = ImageDraw.Draw(image)
            draw.text((8,5),f'V4 balanced reference | t={frame*dt:.2f}s | front / oblique',fill='black')
            draw.text((8,22),'KINEMATIC REFERENCE - not policy inference or physics tracking',fill='black')
            frames.append(image)
        video(args.output/'reference_v4.mp4',frames,1/(2*dt))
        cycle = round(float(new['cycle_duration_s'])/dt)
        gif = [f.resize((640,360)) for f in frames[:cycle//2]]
        gif[0].save(args.output/'reference_v4_cycle.gif',save_all=True,
                    append_images=gif[1:],duration=round(2000*dt),loop=0)
        # 同一真实时间，而非把两个不同步频的视频偷偷按相位变速对齐。
        comparison = []
        length = min(len(new['time_s']),len(old['time_s']))
        for frame in range(0,length,2):
            image = Image.new('RGB',(800,860),'white')
            draw = ImageDraw.Draw(image)
            draw.text((8,5),f'V3 vs V4 | real time t={frame*dt:.2f}s | kinematic only',fill='black')
            draw.text((8,22),'TOP: V3, no COM planning | BOTTOM: V4, COM transfer + double support',fill='black')
            old_frame = min(round(frame*dt/old_dt),len(old['time_s'])-1)
            image.paste(panel(old,old_frame,0),(0,50))
            image.paste(panel(old,old_frame,90),(400,50))
            image.paste(panel(new,frame,0),(0,460))
            image.paste(panel(new,frame,90),(400,460))
            comparison.append(image)
        video(args.output/'v3_vs_v4.mp4',comparison,1/(2*dt))
        # 静态对照按各自一个步态周期的相位采样，明确标注不是同一时刻。
        sheet = Image.new('RGB',(1600,850),'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8,5),'Pose comparison at matched gait phases (NOT matched wall-clock time)',fill='black')
        for col,phase in enumerate([.0,.25,.5,.75]):
            oi=min(17+round(phase*82),len(old['time_s'])-1)
            ni=round(phase*cycle)%cycle
            sheet.paste(panel(old,oi,0),(col*400,40))
            sheet.paste(panel(new,ni,0),(col*400,450))
            draw.text((col*400+8,25),f'V3 phase={phase:.2f}',fill='black')
            draw.text((col*400+8,435),f'V4 phase={phase:.2f}',fill='black')
        sheet.save(args.output/'v3_vs_v4_pose_comparison.png')
        # 便于先目视确认的四视角海报。
        poster=Image.new('RGB',(800,860),'white')
        for i,(phase,angle) in enumerate([(.25,0),(.25,90),(.75,0),(.75,55)]):
            row,col=divmod(i,2)
            poster.paste(panel(new,round(phase*cycle),angle),(col*400,row*430+30))
            ImageDraw.Draw(poster).text((col*400+8,row*430+8),f'V4 phase={phase:.2f} view={angle} deg - kinematic',fill='black')
        poster.save(args.output/'reference_v4_poster.png')
    print(f'Rendered videos and pose comparison to {args.output}')


if __name__ == '__main__':
    main()
