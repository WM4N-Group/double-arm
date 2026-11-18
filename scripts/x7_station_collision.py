#!/usr/bin/env python3
"""Collision geometry used by the X7 mobile carry task (raw CAD + rendered mesh)."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pybullet as p

CAD_MOUNT_Z = 1.77
CAD_STL = 'f40161a4403269fe5cc0a6977476e86b9202a996e8de52ac190a01f7338403c1'
PAD_M = .002
STATION_SCAN_M = .080
STATION_CLEARANCE_M = .005
SLABS_WORLD_M = np.array([0., .72, .84, 1.13, 1.150, 1.195, 1.254, 1.320, 1.38, 1.58, 1.74, 1.94, 2.08])
STL_DTYPE = np.dtype([('normal','<f4',3),('vertices','<f4',(3,3)),('attributes','<u2')])


def triangles(data):
    n = int.from_bytes(data[80:84], 'little')
    if not n or len(data) != 84+50*n:
        raise ValueError('Expected a nonempty binary STL')
    xyz = np.frombuffer(data, STL_DTYPE, offset=84)['vertices']
    if not np.isfinite(xyz).all():
        raise ValueError('Non-finite CAD vertex')
    return xyz


def clipped_bounds(mesh, lo, hi):
    """Exact axis-aligned bounds after clipping triangles to a local X slab.

    A clipped convex polygon's vertices are original in-slab vertices or
    intersections of an original triangle edge and either clipping plane.
    """
    x = mesh[:,:,0]
    mesh = mesh[(x.min(axis=1)<=hi) & (x.max(axis=1)>=lo)]
    if not len(mesh):
        return None
    x = mesh[:,:,0]
    points = [mesh[(x>=lo)&(x<=hi)]]
    for k in range(3):
        a,b = mesh[:,k],mesh[:,(k+1)%3]
        for level in (lo,hi):
            mask=(np.minimum(a[:,0],b[:,0])<level)&(np.maximum(a[:,0],b[:,0])>level)
            aa,bb=a[mask].astype(float),b[mask].astype(float)
            if len(aa):
                points.append(aa+(bb-aa)*((level-aa[:,0])/(bb[:,0]-aa[:,0]))[:,None])
    v=np.concatenate(points)
    return v.min(axis=0),v.max(axis=0)


def cad_envelopes(cad_zip, visible_stl):
    with ZipFile(cad_zip) as z:
        [name] = [n for n in z.namelist() if n.endswith('/meshes/base_link.STL')]
        raw=z.read(name)
    if sha256(raw).hexdigest()!=CAD_STL:
        raise ValueError('Unsupported raw X7 base CAD revision')
    visible=Path(visible_stl).read_bytes()
    datasets=[triangles(data) for data in (raw,visible)]
    cut=SLABS_WORLD_M-CAD_MOUNT_Z
    if any(t[:,:,0].min()<cut[0] or t[:,:,0].max()>cut[-1] for t in datasets):
        raise ValueError('CAD is not contained in audited slab heights')
    boxes=[]
    for lo,hi in zip(cut[:-1],cut[1:]):
        bounds=[b for t in datasets if (b:=clipped_bounds(t,lo,hi)) is not None]
        if bounds:
            lower=np.min([b[0] for b in bounds],axis=0)-PAD_M
            upper=np.max([b[1] for b in bounds],axis=0)+PAD_M
            boxes.append(dict(center=((lower+upper)/2).tolist(),size=(upper-lower).tolist()))
    return {'raw_cad_stl_sha256':CAD_STL,'visible_base_stl_sha256':sha256(visible).hexdigest(),
            'padding_m':PAD_M,'parts_in_cad_frame':boxes}


def station_parts(x, y, table_top, rgba):
    """Robot-side opening, floor legs wholly beyond the CAD base footprint.

    Every visible structural part is passed to *both* shape constructors.
    The outriggers sit on the operator/far side and top brackets cantilever
    from them; no visual-only geometry traverses the planned chassis lane.
    """
    parts=[('deck',[x,y-.016,table_top-.026],[1.32,.25,.052],rgba)]
    for name,dx in [('left',-.55),('right',.55)]:
        support_y=-.855
        parts.extend([(name+'_leg',[x+dx,support_y,(table_top-.059)/2],[.072,.076,table_top-.059],rgba),
                      (name+'_beam',[x+dx,(support_y+y)/2,table_top-.074],[.072,y-support_y+.076,.044],rgba)])
    parts.append(('rail',[x,-.855,.23],[1.16,.061,.055],rgba))
    return parts


def compound_station(parts, *, position=(0.,0.,0.), yaw=0.):
    shapes=[p.GEOM_BOX]*len(parts)
    half=(np.asarray([row[2] for row in parts],dtype=float)/2).tolist()
    centers=[row[1] for row in parts]
    col=p.createCollisionShapeArray(shapeTypes=shapes,halfExtents=half,collisionFramePositions=centers)
    vis=p.createVisualShapeArray(shapeTypes=shapes,halfExtents=half,visualFramePositions=centers,
                                rgbaColors=[row[3] for row in parts])
    return p.createMultiBody(baseMass=0,baseCollisionShapeIndex=col,baseVisualShapeIndex=vis,
                             basePosition=position,baseOrientation=p.getQuaternionFromEuler([0.,0.,yaw]))


def station_clearance(robot, stations):
    """Signed minimum per station; no normal-force gate. No hit is censored at scan radius."""
    distance=[]; contacts=[]; link=[]
    for station in stations:
        points=p.getClosestPoints(robot,station,STATION_SCAN_M)
        near=min(points,key=lambda c:c[8],default=None)
        d=float(near[8]) if near is not None else STATION_SCAN_M
        if not np.isfinite(d):
            raise RuntimeError('Non-finite signed distance')
        distance.append(d)
        contacts.append(len(p.getContactPoints(robot,station)))
        link.append(int(near[3]) if near is not None else -2)
    return distance,contacts,link
