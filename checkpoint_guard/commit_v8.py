#!/usr/bin/env python3
"""Persist one V8 batch as an immutable checkpoint generation.

Run after each successful parent or fine-step commit. A partial source snapshot
is not accepted. Store should be on an independently durable mounted volume;
copy its generation archive out of ephemeral workspace after each batch.
"""
import argparse,json,sys
from pathlib import Path
from checkpoint_guard import snapshot,verify

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--run',type=Path,required=True)
 p.add_argument('--store',type=Path,required=True)
 p.add_argument('--stage',choices=['parent','fine'],required=True)
 args=p.parse_args();r=args.run
 parent=r/'parent/manifest.json';cap=r/'capture/capture.json'
 if not parent.is_file():raise FileNotFoundError(parent)
 pm=json.loads(parent.read_text());count=len(pm.get('frames',[]))
 if count<1:raise RuntimeError('No completed parent snapshots')
 source=['parent/manifest.json','parent/restart.npz','parent/initial-q.npy',f'parent/velocity-{count-1:03d}.npy']
 if args.stage=='fine':
  if not cap.is_file():raise FileNotFoundError(cap)
  c=json.loads(cap.read_text())
  if c.get('current',0)<1 or not c.get('checkpoint_npy'):raise RuntimeError('No committed fine state')
  source+=['capture/capture.json','capture/'+c['checkpoint_npy']]
  if c.get('surface_checkpoint_npy'):source+=['capture/'+c['surface_checkpoint_npy']]
  for x in c.get('frames',[]):source.append(f'capture/frames/{x["index"]:04d}.png')
  for x in c.get('frames',[]):
   from checkpoint_guard import digest
   if digest(r/f'capture/frames/{x["index"]:04d}.png')!=x['sha256']:
    raise RuntimeError('Frame hash mismatch at index '+str(x['index']))
 generation=f'{args.stage}-p{count:04d}'+(f'-f{c["current"]:04d}' if args.stage=='fine' else '')
 m=snapshot(r,args.store,source,generation)
 verify(args.store,m['generation'])
 print('COMMITTED',m['generation'],sum(f['bytes'] for f in m['files']),'bytes')
if __name__=='__main__':main()
