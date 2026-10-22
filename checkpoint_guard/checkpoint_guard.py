#!/usr/bin/env python3
"""Crash-safe, content-verified checkpoint bundles for resumable CYBR FIRE runs.

A generation is only visible in LATEST after its immutable snapshot has been
fully written, fsynced, hashed, and atomically renamed.  Untrusted manifests
are never accepted without checking the referenced bytes.
"""
import argparse, hashlib, json, os, shutil, sys, tempfile, time, uuid, zipfile
from pathlib import Path

BLOCK=8*1024*1024

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(BLOCK),b''):h.update(chunk)
    return h.hexdigest()

def sync_dir(p):
    fd=os.open(str(p),os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def atomic_text(p,s):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp')
    with open(t,'w',encoding='utf8') as f:
        f.write(s);f.flush();os.fsync(f.fileno())
    os.replace(t,p);sync_dir(p.parent)

def protected_rel(path):
    p=Path(path)
    if p.is_absolute() or not p.parts or '..' in p.parts:raise ValueError('unsafe relative path')
    return p

def snapshot(run, store, includes, generation=None):
    run=Path(run).resolve();store=Path(store).resolve();store.mkdir(parents=True,exist_ok=True)
    generation=generation or time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:8]
    protected_rel(generation)
    dst=store/'generations'/generation
    if dst.exists():raise FileExistsError(dst)
    staging=store/'generations'/('.'+generation+'.partial')
    staging.mkdir(parents=True)
    entries=[]
    try:
        for s in includes:
            rel=protected_rel(s)
            src=run/rel
            if not src.is_file():raise FileNotFoundError('Required checkpoint missing: '+str(src))
            if src.is_symlink():raise ValueError('Symlink forbidden: '+str(src))
            target=staging/'payload'/rel;target.parent.mkdir(parents=True,exist_ok=True)
            # Independent copy, never hardlink; the live simulation may replace files.
            shutil.copyfile(src,target)
            with open(target,'rb') as f:os.fsync(f.fileno())
            source_hash=digest(src);copy_hash=digest(target)
            if source_hash!=copy_hash:raise RuntimeError('Source changed during snapshot: '+str(rel))
            entries.append(dict(path=str(rel),bytes=target.stat().st_size,sha256=copy_hash))
        manifest=dict(format='cybr-fire-checkpoint-v1',generation=generation,
                      created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),files=entries)
        atomic_text(staging/'manifest.json',json.dumps(manifest,indent=2,sort_keys=True))
        sync_dir(staging/'payload');sync_dir(staging)
        staging.rename(dst);sync_dir(dst.parent)
        # Commit pointer last: failed snapshots are never counted as completed.
        atomic_text(store/'LATEST',generation+'\n')
        return manifest
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise

def verify(store,generation=None):
    store=Path(store)
    if generation is None:
        latest=store/'LATEST'
        if not latest.exists():raise FileNotFoundError('No committed checkpoint (LATEST missing)')
        generation=latest.read_text().strip()
    protected_rel(generation)
    root=store/'generations'/generation
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['format']!='cybr-fire-checkpoint-v1' or manifest['generation']!=generation:raise ValueError('Bad manifest')
    seen=set()
    for f in manifest['files']:
        rel=protected_rel(f['path'])
        if str(rel) in seen:raise ValueError('Repeated manifest entry')
        seen.add(str(rel))
        path=root/'payload'/rel
        if not path.is_file() or path.is_symlink():raise FileNotFoundError(path)
        if path.stat().st_size!=f['bytes'] or digest(path)!=f['sha256']:
            raise ValueError('Corrupt checkpoint '+str(rel))
    if not seen:raise ValueError('Empty checkpoint is not useful')
    return manifest

def restore(store, into, generation=None):
    m=verify(store,generation); source=Path(store)/'generations'/m['generation']/'payload';into=Path(into)
    # Validate before writing into live run. Do not overwrite by default.
    for f in m['files']:
        dest=into/protected_rel(f['path'])
        if dest.exists():raise FileExistsError('Refusing to overwrite existing live file: '+str(dest))
    for f in m['files']:
        rel=protected_rel(f['path']);dest=into/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/rel,dest)
        if digest(dest)!=f['sha256']:raise RuntimeError('Restore checksum failed: '+str(rel))
    return m

def main():
    ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='cmd',required=True)
    a=sp.add_parser('commit');a.add_argument('--run',required=True);a.add_argument('--store',required=True);a.add_argument('--files',nargs='+',required=True);a.add_argument('--generation')
    a=sp.add_parser('verify');a.add_argument('--store',required=True);a.add_argument('--generation')
    a=sp.add_parser('restore');a.add_argument('--store',required=True);a.add_argument('--into',required=True);a.add_argument('--generation')
    args=ap.parse_args()
    if args.cmd=='commit':m=snapshot(args.run,args.store,args.files,args.generation)
    elif args.cmd=='verify':m=verify(args.store,args.generation)
    else:m=restore(args.store,args.into,args.generation)
    print(json.dumps(dict(result='verified',generation=m['generation'],count=len(m['files']),bytes=sum(f['bytes'] for f in m['files'])),indent=2))
if __name__=='__main__':main()
