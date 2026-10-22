#!/usr/bin/env python3
"""Restore a downloaded Library generation using a previously recorded SHA-256.

Download the exact file_id in LATEST_CHECKPOINT.json using the Files connector.
This tool does not perform network requests or pretend a local copy is uploaded.
"""
from pathlib import Path
import argparse, hashlib, json, stat, sys, zipfile
from checkpoint_guard import digest, restore, verify, atomic_text
from v8_guarded import audit

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--remote-file-id',required=True)
    p.add_argument('--into',type=Path,required=True)
    a=p.parse_args()
    if digest(a.archive)!=a.expected_sha256:raise RuntimeError('Downloaded archive hash mismatch')
    if a.into.exists() and any(a.into.iterdir()):raise FileExistsError('Restore destination must be empty')
    a.into.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.archive) as z:
        seen=set()
        for i in z.infolist():
            f=Path(i.filename)
            if f.is_absolute() or '..' in f.parts or i.filename in seen or stat.S_ISLNK(i.external_attr>>16):
                raise ValueError('Unsafe archive member')
            seen.add(i.filename)
        if z.testzip() is not None:raise ValueError('Archive CRC failed')
        z.extractall(a.into)
    m=verify(a.into/'store');restore(a.into/'store',a.into/'run')
    state=audit(a.into/'run')
    receipt={'generation':m['generation'],'status':'uploaded_and_downloaded_hash_match',
        'archive_sha256':a.expected_sha256,'remote_file_id':a.remote_file_id,
        'downloaded_bytes':a.archive.stat().st_size,
        'verification_basis':'downloaded archive matches independently retained registry hash'}
    atomic_text(a.into/'store/REMOTE_CONFIRMED.json',json.dumps(receipt,indent=2))
    print(json.dumps({'status':'restored_and_verified','run':str(a.into/'run'),
          'generation':m['generation'],'state':state},indent=2))
if __name__=='__main__':main()
