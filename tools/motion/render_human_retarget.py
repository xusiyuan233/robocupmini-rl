"""Render measured human skeleton and retargeted robot at IDENTICAL source times.

No time/phase realignment, no inference, no dynamics. EGL/offline rendering.
"""
import argparse
import json
from pathlib import Path
from typing import Any, cast
import subprocess
import mujoco as _mj
import numpy as np
from PIL import Image, ImageDraw
from play_retarget_motion import build_model, joint_mapping, load_motion, apply_frame, MODEL_PATH
mj=cast(Any,_mj)
ROOT=Path(__file__).resolve().parents[2]


def human_model():
    return mj.MjModel.from_xml_string('''<mujoco><visual><global offwidth="480" offheight="360"/></visual>
    <asset><texture name="grid" type="2d" builtin="checker" width="256" height="256" rgb1=".22 .25 .29" rgb2=".36 .39 .43"/>
    <material name="floor" texture="grid" texrepeat="12 12" texuniform="true"/></asset>
    <worldbody><geom type="plane" size="6 6 .1" material="floor"/><light pos="0 -2 4" dir="0 0 -1" directional="true"/></worldbody></mujoco>''')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion',type=Path,default=ROOT/'data/motions/robotcupmini/retargeted/walk1_candidate_v5_human/candidate_motion.npz')
    parser.add_argument('--output',type=Path,default=ROOT/'data/motions/robotcupmini/validated/walk1_candidate_v5_human/visuals')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    motion,names,dt=load_motion(args.motion)
    report=json.loads((args.motion.parent/'generation_report.json').read_text())
    m=build_model(MODEL_PATH);d=mj.MjData(m);_,addresses,root=joint_mapping(m,names)
    hm=human_model();hd=mj.MjData(hm);mj.mj_forward(hm,hd)
    option=mj.MjvOption();mj.mjv_defaultOption(option);option.geomgroup[3]=0
    camera=mj.MjvCamera();mj.mjv_defaultCamera(camera);camera.distance=1.65;camera.elevation=-8
    scale=report['spatial_scale'];human=motion['human_positions_m']*scale
    human[:,:,2]+=float(np.median(motion['source_root_target_m'][:,2]-human[:,0,2]))
    parents=motion['human_parents'];hnames=motion['human_joint_names'].tolist()
    colours=[np.array([.2,.6,.95,1]) if n.startswith('Left') else np.array([.95,.55,.2,1]) if n.startswith('Right') else np.array([.85,.85,.85,1]) for n in hnames]
    process=None;thumbs=[];frame_count=len(motion['time_s'])
    width,height=960,800
    command=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
             '-framerate',str(1/(2*dt)),'-i','-','-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p',
             '-movflags','+faststart',str(args.output/'human_vs_robot_v5.mp4')]
    with mj.Renderer(m,height=360,width=480) as renderer, mj.Renderer(hm,height=360,width=480) as hr:
        def robot_panel(i,azimuth):
            apply_frame(m,d,motion,addresses,root,i);camera.azimuth=azimuth
            camera.lookat[:]=d.qpos[:3];camera.lookat[2]=.52
            renderer.update_scene(d,camera=camera,scene_option=option)
            return Image.fromarray(renderer.render())
        def human_panel(i,azimuth):
            camera.azimuth=azimuth;camera.lookat[:]=human[i,0];camera.lookat[2]=.52
            hr.update_scene(hd,camera=camera)
            for j,p in enumerate(parents):
                if p<0:continue
                scene=hr.scene;k=scene.ngeom
                mj.mjv_initGeom(scene.geoms[k],mj.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).flatten(),colours[j])
                mj.mjv_connector(scene.geoms[k],mj.mjtGeom.mjGEOM_CAPSULE,.009,human[i,p],human[i,j]);scene.ngeom+=1
            return Image.fromarray(hr.render())
        def frame(i):
            canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
            draw.text((8,5),f'V5 HUMAN RETARGET | playback t={i*dt:.2f}s | measured BVH t={motion["source_time_s"][i]:.3f}s',fill='black')
            draw.text((8,21),f'Human and robot: identical source timestamps | time stretch x{report["source_time_scale"]:.2f} | NOT dynamics or policy',fill='black')
            draw.text((8,39),'TOP: source human, uniformly scaled for display | front / side',fill='black')
            draw.text((8,425),'BOTTOM: robot retarget | front / side | BLUE left, ORANGE right',fill='black')
            for col,az in enumerate([0,90]):
                canvas.paste(human_panel(i,az),(col*480,60))
                canvas.paste(robot_panel(i,az),(col*480,440))
            return canvas
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        if process.stdin is None:raise RuntimeError('ffmpeg stdin unavailable')
        try:
            for i in range(0,frame_count,2):
                im=frame(i);process.stdin.write(im.tobytes())
                if len(thumbs)<120:thumbs.append(im.resize((600,500)))
        finally:process.stdin.close()
        if process.wait():raise RuntimeError('ffmpeg failed')
        # Four snapshots from one measured same-foot stride, no phase warping.
        intervals=report['contact_calibration'][0]['stance_intervals_frames']
        start,stop=intervals[0][0],intervals[1][0]
        indices=np.linspace(start,stop,4,endpoint=False,dtype=int)
        sheet=Image.new('RGB',(1920,800),'white')
        for col,i in enumerate(indices[:4]):
            sheet.paste(robot_panel(int(i),0),(col*480,40));sheet.paste(robot_panel(int(i),90),(col*480,440))
            ImageDraw.Draw(sheet).text((col*480+8,10),f't={i*dt:.2f}s / source={motion["source_time_s"][i]:.2f}s',fill='black')
        sheet.save(args.output/'robot_v5_contact_sheet.png')
        frame(int(frame_count//2)).save(args.output/'human_vs_robot_poster.png')
        if thumbs:thumbs[0].save(args.output/'human_vs_robot_preview.gif',save_all=True,append_images=thumbs[1:],duration=round(2000*dt),loop=0)
    print(args.output/'human_vs_robot_v5.mp4')

if __name__=='__main__':main()
