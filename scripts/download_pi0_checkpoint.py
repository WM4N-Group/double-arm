#!/usr/bin/env python3
"""SERVER ONLY: resumable public pi0_base download with per-object MD5 checks.

No authentication, gated-download acceptance, or policy terms are submitted.
Only the official public bucket and original pi0_base prefix are supported.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
import requests

PREFIX='checkpoints/pi0_base/'


def checksum(path):
    result=hashlib.md5()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):result.update(block)
    return base64.b64encode(result.digest()).decode()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/mnt/yuanyue/double-arm'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--object-base-url',default='https://storage.googleapis.com/openpi-assets')
    args=parser.parse_args()
    if args.object_base_url!='https://storage.googleapis.com/openpi-assets' and not args.object_base_url.startswith('http://127.0.0.1:'):
        raise ValueError('Only official bucket or explicitly local SSH relay allowed')
    if not str(args.root).startswith('/mnt/'):raise ValueError('Persistent remote root required')
    output=args.root/'data/pi0/checkpoints/pi0_base';output.mkdir(parents=True,exist_ok=True)
    items=[];token=None
    while True:
        params={'prefix':PREFIX,'fields':'items(name,size,md5Hash),nextPageToken'}
        if token:params['pageToken']=token
        response=requests.get('https://storage.googleapis.com/storage/v1/b/openpi-assets/o',params=params,timeout=60)
        response.raise_for_status();reply=response.json();items.extend(reply.get('items',[]));token=reply.get('nextPageToken')
        if not token:break
    if not items:raise ValueError('Empty public checkpoint listing')
    manifest={'bucket':'openpi-assets','prefix':PREFIX,'items':items,'bytes':sum(int(item['size']) for item in items),'status':'downloading'}
    (output/'download-manifest.json').write_text(json.dumps(manifest,indent=2))
    def download(item):
        relative=Path(item['name'][len(PREFIX):])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe bucket path')
        destination=output/relative;destination.parent.mkdir(parents=True,exist_ok=True);size=int(item['size']);expected=item.get('md5Hash')
        if destination.exists() and destination.stat().st_size==size and expected and checksum(destination)==expected:
            return {'name':str(relative),'status':'already_verified','bytes':size}
        partial=destination.with_name(destination.name+'.partial')
        for attempt in range(5):
            try:
                offset=partial.stat().st_size if partial.exists() else 0
                if offset>size:raise ValueError('Partial object larger than advertised object')
                if offset<size:
                    headers={'Range':f'bytes={offset}-'} if offset else {}
                    base=args.object_base_url if attempt<2 else 'https://storage.googleapis.com/openpi-assets'
                    with requests.get(base.rstrip('/')+'/'+quote(item['name'],safe='/'),headers=headers,stream=True,timeout=(30,120)) as response:
                        response.raise_for_status()
                        if offset and response.status_code!=206:offset=0
                        with partial.open('ab' if offset else 'wb') as handle:
                            for block in response.iter_content(256*1024):
                                if block:handle.write(block)
                if partial.stat().st_size!=size:raise ValueError('Object size mismatch')
                if not expected or checksum(partial)!=expected:
                    partial.rename(partial.with_name(partial.name+f'.bad-{int(time.time())}'))
                    raise ValueError('Missing or mismatched MD5')
                partial.replace(destination)
                return {'name':str(relative),'status':'downloaded_verified','bytes':size}
            except Exception:
                if attempt==4:raise
                time.sleep(2**attempt)
    completed=[]
    with ThreadPoolExecutor(max_workers=min(max(1,args.workers),16)) as pool:
        for future in as_completed([pool.submit(download,item) for item in sorted(items,key=lambda item:int(item['size']),reverse=True)]):
            item=future.result();completed.append(item);print(json.dumps(item),flush=True)
    manifest.update(status='download_verified',completed_objects=len(completed),verified_md5=True)
    (output/'download-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({key:manifest[key] for key in ['status','completed_objects','bytes','verified_md5']}),flush=True)


if __name__=='__main__':main()
