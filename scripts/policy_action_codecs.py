"""Reversible learned-action encodings, not task planners or grasp rules."""
import numpy as np


def xy_rotation(quaternion):
    q=np.asarray(quaternion,dtype=np.float64)
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    x,y,z,w=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*y+z*w),1-2*(x*x+z*z)),axis=-1).reshape(q.shape[:-1]+(2,2))


def encode_world_base_goals(actions,base_poses,base_headings):
    """21 motor commands -> 22 goals: world XY, sin/cos joint yaw, 18 other commands."""
    actions=np.asarray(actions);poses=np.asarray(base_poses)
    xy=poses[:,:2]+np.einsum('nij,nj->ni',xy_rotation(poses[:,3:7]),actions[:,:2])
    angle=np.asarray(base_headings)+actions[:,2]
    return np.concatenate((xy,np.sin(angle)[:,None],np.cos(angle)[:,None],actions[:,3:]),axis=1).astype(np.float32)


def decode_world_base_goals(goals,base_pose,base_heading):
    """Goal -> native local position command, using only measured current pose.

    This is the inverse of the native holonomic controller's coordinate change.
    There are no object names, task phases, navigation waypoints or grasp rules.
    """
    goals=np.asarray(goals);pose=np.asarray(base_pose)
    rotation=xy_rotation(pose[3:7])
    if abs(np.linalg.det(rotation))<1e-3:raise ValueError('Robot base is too tilted for planar goal control')
    xy=np.linalg.solve(rotation,(goals[:,:2]-pose[:2]).T).T
    heading=np.arctan2(goals[:,2],goals[:,3]);difference=heading-base_heading
    yaw=np.arctan2(np.sin(difference),np.cos(difference))[:,None]
    return np.concatenate((xy,yaw,goals[:,4:]),axis=1).astype(np.float32)


def world_goal_anchor(base_poses, base_headings, joint_anchors):
    """Current measured motor state in the 22D goal encoding, without task logic.

    Gripper classifier baselines are zero. Heading is kept as sin/cos so the
    neural residual has no +/-pi discontinuity. No future observation is used.
    """
    poses=np.asarray(base_poses);heading=np.asarray(base_headings)
    joints=np.asarray(joint_anchors)
    if poses.shape[-1]!=7 or joints.shape[-1]!=21:
        raise ValueError('Expected 7D root pose and 21D motor-layout joint anchors')
    zeros=np.zeros(poses.shape[:-1]+(1,),dtype=np.float32)
    return np.concatenate((poses[...,:2],np.sin(heading)[...,None],np.cos(heading)[...,None],
                           joints[...,3:13],zeros,joints[...,14:20],zeros),axis=-1).astype(np.float32)
