#!/usr/bin/env python3
"""One transaction-sized batch of native V8 (no loops, no output upscaling).

Run this repeatedly, one process at a time. A monotonically advancing cache
separates pressure/velocity computation from native fine transport and rendering.
Snapshots and per-frame checkpoints are persistent and independently verifiable.
"""
from pathlib import Path
import argparse, fcntl, json, os, subprocess, sys, hashlib, time

ROOT=Path(__file__).resolve().parents[1]
SCENES={
 'jet':{'warmup':24,'width':3840,'height':2160,'exposure':.009,'distance':1.25,'fov':42.,'target':.65},
 'pool':{'warmup':24,'width':3840,'height':2160,'exposure':.03,'distance':3.25,'fov':40.,'target':.96},
 'fireball':{'warmup':1,'width':2160,'height':3840,'exposure':.03,'distance':3.55,'fov':40.,'target':1.30},
}
def read(path):
 return json.loads(path.read_text()) if path.exists() else None

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()

def verify_frames(out, rec):
 frames=rec['frames']; expected=list(range(len(frames)))
 assert [f['index'] for f in frames]==expected,'Missing/disordered physical frames'
 hashes=[]; times=[]
 for f in frames:
  p=out/'capture'/'frames'/f"{f['index']:04d}.png"
  assert p.exists() and digest(p)==f['sha256'],'Frame missing or changed: '+str(p)
  hashes.append(f['sha256']);times.append(f['sim_time'])
 assert len(set(hashes))==len(hashes),'Duplicate source frames'
 assert all(t2>t1 for t1,t2 in zip(times,times[1:])),'Non-increasing simulation time'
 return len(hashes)

def partial_video(out, rec, scene):
 count=verify_frames(out,rec)
 if count<2:return None
 # Produced at 24 fps, from exactly the independently computed PNGs.
 path=out/'capture'/f'{scene}_INCOMPLETE_{count:04d}frames_4K.mp4'
 tmp=out/'capture'/'partial.tmp.mp4'
 subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','24',
  '-start_number','0','-i',str(out/'capture'/'frames'/'%04d.png'),'-frames:v',str(count),
  '-c:v','libx264','-threads','4','-preset','veryfast','-crf','18',
  '-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],check=True)
 got=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
  '-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames',
  '-of','json',str(tmp)]))['streams'][0]
 assert int(got['nb_read_frames'])==count
 assert (int(got['width']),int(got['height'])) in ((3840,2160),(2160,3840))
 tmp.replace(path)
 return str(path)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--scene',choices=SCENES,required=True)
 ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--stage',choices=['parent','capture','status','encode-partial'],required=True)
 ap.add_argument('--frames',type=int,default=120)
 ap.add_argument('--warmup',type=int,default=None)
 ap.add_argument('--batch',type=int,default=1)
 args=ap.parse_args()
 if args.frames<1 or args.batch<1:raise ValueError('Invalid batch/length')
 c=SCENES[args.scene]
 warmup=c['warmup'] if args.warmup is None else args.warmup
 if warmup<0:raise ValueError('Invalid warmup')
 out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
 (out/'parent').mkdir(exist_ok=True);(out/'capture').mkdir(exist_ok=True)
 with (out/'.work.lock').open('w') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  parent=read(out/'parent'/'manifest.json')
  rec=read(out/'capture'/'capture.json')
  print('BEFORE',args.scene,'parent',len(parent['frames']) if parent else 0,
        'fine',rec['current'] if rec else 0,'rendered',len(rec['frames']) if rec else 0,flush=True)
  if args.stage=='parent':
   if parent and len(parent['frames'])>=warmup+args.frames+1:
    print('Parent already covers requested duration')
   else:
    cmd=[sys.executable,str(ROOT/'tools'/'build_scene_cache.py'),
         '--scene',args.scene,'--quality','native-v8',
         '--out',str(out/'parent'),'--frames',str(warmup+args.frames+1),
         '--batch',str(args.batch)]
    subprocess.run(cmd,cwd=ROOT,check=True)
  elif args.stage=='capture':
   if not parent or len(parent['frames'])<2:raise RuntimeError('Build parent batches first')
   cmd=[sys.executable,str(ROOT/'tools'/'capture_scene.py'),
        '--quality','native-v8','--parent',str(out/'parent'),
        '--out',str(out/'capture'),'--warmup',str(warmup),
        '--frames',str(args.frames),'--batch',str(args.batch),
        '--width',str(c['width']),'--height',str(c['height']),
        '--exposure',str(c['exposure']),'--distance',str(c['distance']),
        '--fov',str(c['fov']),'--target-height',str(c['target'])]
   subprocess.run(cmd,cwd=ROOT,check=True)
  elif args.stage=='encode-partial':
   if not rec:raise RuntimeError('No fine frames yet')
   print('PARTIAL',partial_video(out,rec,args.scene))
  parent=read(out/'parent'/'manifest.json');rec=read(out/'capture'/'capture.json')
  n=verify_frames(out,rec) if rec else 0
  receipt={
   'scenario':args.scene,'quality':'native-v8',
   'target_frames':args.frames,'warmup':warmup,
   'parent_grid':[160,280,160], 'fine_grid':[384,672,384],
   'native_video_dimensions':[c['width'],c['height']],
   'parent_snapshots':len(parent['frames']) if parent else 0,
   'parent_target_snapshots':warmup+args.frames+1,
   'parent_complete':bool(parent and parent.get('complete')),
   'fine_steps_complete':rec['current'] if rec else 0,
   'genuinely_rendered_frames':n,
   'first_unrendered_frame':n,
   'duration_seconds_complete':n/24,
   'total_duration_requested_seconds':args.frames/24,
   'finished':bool(rec and rec.get('status')=='completed_verified'),
   'no_frame_interpolation':True,'no_loops':True,'no_upscaling':True,
  }
  tmp=out/'receipt.tmp.json';tmp.write_text(json.dumps(receipt,indent=2));tmp.replace(out/'batch_receipt.json')
  print('AFTER',json.dumps(receipt),flush=True)

if __name__=='__main__':main()
