"""World-base target <-> native local displacement, using robot odometry only.

No task-object pose, planner or trajectory phase is used. The pinned holonomic
controller transforms local XY displacement through the root rotation each step.
"""
import numpy as np


def root_rotation_xy_and_yaw(poses):
    poses=np.asarray(poses,dtype=np.float64)
    if poses.shape[-1]!=7 or not np.isfinite(poses).all():raise ValueError('Expected finite robot XYZ + XYZW odometry')
    q=poses[...,3:];norm=np.linalg.norm(q,axis=-1,keepdims=True)
    if np.any(norm<.99) or np.any(norm>1.01):raise ValueError('Invalid root quaternion')
    x,y,z,w=np.moveaxis(q/norm,-1,0)
    rotation=np.stack([np.stack([1-2*(y*y+z*z),2*(x*y-z*w)],axis=-1),
                       np.stack([2*(x*y+z*w),1-2*(x*x+z*z)],axis=-1)],axis=-2)
    if np.any(np.abs(np.linalg.det(rotation))<.98):raise ValueError('Robot root is not approximately planar')
    return rotation,np.arctan2(rotation[...,1,0],rotation[...,0,0])


def native_to_absolute_base(actions,poses):
    actions=np.asarray(actions,dtype=np.float64);poses=np.asarray(poses,dtype=np.float64)
    if actions.shape[-1]!=21 or actions.shape[:-1]!=poses.shape[:-1]:raise ValueError('Misaligned native commands and robot odometry')
    rotation,yaw=root_rotation_xy_and_yaw(poses)
    result=actions.copy()
    result[...,:2]=poses[...,:2]+np.einsum('...ij,...j->...i',rotation,actions[...,:2])
    # One fixed heading branch for ALL episodes. Per-episode unwrap would
    # assign different 2*pi aliases to identical visual/robot states, although
    # the VLA has no trajectory-history input to distinguish those aliases.
    result[...,2]=(yaw+actions[...,2]+np.pi)%(2*np.pi)-np.pi
    return result.astype(np.float32)


def absolute_base_to_native(target,pose):
    target=np.asarray(target,dtype=np.float64);pose=np.asarray(pose,dtype=np.float64)
    if target.shape[-1]!=21 or target.shape[:-1]!=pose.shape[:-1]:raise ValueError('Misaligned absolute targets and robot odometry')
    rotation,yaw=root_rotation_xy_and_yaw(pose)
    result=target.copy()
    result[...,:2]=np.linalg.solve(rotation,(target[...,:2]-pose[...,:2])[...,None])[...,0]
    result[...,2]=(target[...,2]-yaw+np.pi)%(2*np.pi)-np.pi
    if not np.isfinite(result).all():raise ValueError('Invalid target conversion')
    return result.astype(np.float32)
