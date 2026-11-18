#!/usr/bin/env python3
"""Independent geometry cross-check of X7 base clearance in a strict carry trace.

No physics is started. A conservative swept AABB spans the entire recorded X
interval, with additional padding for measured lateral drift and yaw. This is
an independent lower bound for chassis clearance within that pose envelope,
not a continuous all-link collision certificate.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from run_x7_mobile_bimanual import PARK_Y, TABLE_Z
from x7_station_collision import CAD_MOUNT_Z, CAD_STL, STATION_CLEARANCE_M, station_parts

EXTRA_PAD_M = .002


def signed_aabb_separation(lo_a, hi_a, lo_b, hi_b):
    gap = np.maximum(lo_a-hi_b,lo_b-hi_a)
    if np.any(gap>0):
        return float(np.linalg.norm(np.maximum(gap,0.)))
    # Minimum translation to separate the boxes, including contained intervals.
    return float(np.max(gap))


def audit(bounds, trace):
    if bounds['raw_cad_stl_sha256']!=CAD_STL:
        raise ValueError('Different raw CAD geometry')
    samples=np.r_[trace['base_pre'][:,:,0],trace['base_post'][:,:,0],[[0.,0.,0.]]]
    xmin,xmax=map(float,(samples[:,0].min(),samples[:,0].max()))
    lateral,yaw=np.max(np.abs(samples[:,1:]),axis=0)
    station_b=float(trace['target_xyz'][0]-trace['initial_object_xyz'][0])
    if (not np.isfinite(samples).all() or not np.isfinite(station_b)
        or not 2.4<=station_b<=3.0):
        raise ValueError('Expected a valid mobile carry trace from the corrected task')
    results={}
    for label,station_x in zip(('A','B'),(0.,station_b)):
        nearest=(np.inf,None,None)
        for i,slab in enumerate(bounds['parts_in_cad_frame']):
            x,y,z=slab['center']; dx,dy,dz=slab['size']
            ctr=np.array([-z,y,CAD_MOUNT_Z+x]); half=.5*np.array([dz,dy,dx]); r=np.linalg.norm(abs(ctr[:2])+half[:2])
            dr=float(r*yaw)+EXTRA_PAD_M
            lo=ctr-half+[xmin-dr,-lateral-dr,0.]; hi=ctr+half+[xmax+dr,lateral+dr,0.]
            for name,center,dims,_ in station_parts(station_x,PARK_Y,TABLE_Z,[0.,0.,0.,1.]):
                b=np.asarray(center); bhalf=.5*np.asarray(dims)
                d=signed_aabb_separation(lo,hi,b-bhalf,b+bhalf)
                if d<nearest[0]: nearest=(d,i,name)
        results[label]={'lower_bound_m':nearest[0],'base_slab_index':nearest[1],'station_part':nearest[2]}
    return {'scope':'chassis only; conservative full-X sweep at measured lateral/yaw envelope plus extra horizontal padding',
            'x_interval_m':[xmin,xmax],'lateral_envelope_m':float(lateral),'yaw_envelope_rad':float(yaw),
            'extra_horizontal_padding_m':EXTRA_PAD_M,'stations':results,
            'pass':all(result['lower_bound_m']>=STATION_CLEARANCE_M for result in results.values())}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('base_envelope',type=Path)
    ap.add_argument('episode',type=Path,nargs='+')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    bounds=json.loads(args.base_envelope.read_text()); rows=[]
    for episode in args.episode:
        with np.load(episode,allow_pickle=False) as trace:
            row=audit(bounds,trace)
        rows.append(row|{'episode':episode.name,'trace_sha256':sha256(episode.read_bytes()).hexdigest()})
    report={'results':rows,'all_pass':all(r['pass'] for r in rows)}
    with args.out.open('x') as f: json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))
    return 0 if report['all_pass'] else 1


if __name__=='__main__':
    raise SystemExit(main())
