"""R1 native observations / commands for pi0. No privileged object-pose input."""
import numpy as np

CAMERA_SUFFIXES={
    'head_image':':eyes:Camera:0::rgb',
    'left_wrist_image':':left_eef_link:Camera:0::rgb',
    'right_wrist_image':':right_eef_link:Camera:0::rgb',
}
PROMPT='Pick up the cup from the countertop, move to the sink, and place the cup in the sink.'
STATE_NAMES=('base_vx','base_vy','base_vyaw')+tuple(f'trunk_q{i}' for i in range(4))+tuple(f'left_q{i}' for i in range(6))+tuple(f'right_q{i}' for i in range(6))+('left_gripper_width','right_gripper_width')
ACTION_NAMES=('base_local_dx','base_local_dy','base_delta_yaw')+tuple(f'trunk_target_q{i}' for i in range(4))+tuple(f'left_target_q{i}' for i in range(6))+('left_gripper_command',)+tuple(f'right_target_q{i}' for i in range(6))+('right_gripper_command',)


def camera_keys(observations):
    result={}
    for target,suffix in CAMERA_SUFFIXES.items():
        matches=[name for name in observations if name.endswith(suffix)]
        if len(matches)!=1:raise ValueError(f'Need exactly one real camera for {target}, found {matches}')
        result[target]=matches[0]
    return result


def rgb_image(image):
    x=np.asarray(image)
    if x.ndim!=3:raise ValueError('Image must be HWC or CHW')
    if x.shape[0] in (3,4) and x.shape[-1] not in (3,4):x=np.transpose(x,(1,2,0))
    if x.shape[-1] not in (3,4):raise ValueError('Image must have RGB or RGBA channels')
    if np.issubdtype(x.dtype,np.floating):
        if not np.isfinite(x).all() or x.min()<0 or x.max()>1.0001:raise ValueError('Float camera images must be in [0,1]')
        x=np.rint(np.clip(x,0,1)*255).astype(np.uint8)
    if x.dtype!=np.uint8:raise ValueError('Camera images must be uint8 or normalized float')
    return np.ascontiguousarray(x[...,:3])


def native_actions(actions):
    x=np.asarray(actions,dtype=np.float32)
    if x.shape[-1]!=21 or not np.isfinite(x).all():raise ValueError('Expected finite 21D native R1 motor commands')
    return x.copy()


def make_pi0_inputs(data):
    state=np.asarray(data['state'],dtype=np.float32)
    if state.shape!=(21,) or not np.isfinite(state).all():raise ValueError('Expected R1 prop_state, not action-layout joint targets')
    image={target:rgb_image(data[source]) for target,source in {
        'base_0_rgb':'head_image','left_wrist_0_rgb':'left_wrist_image','right_wrist_0_rgb':'right_wrist_image'}.items()}
    output={'state':state,'image':image,'image_mask':{key:np.True_ for key in image}}
    if 'actions' in data:output['actions']=native_actions(data['actions'])
    if 'prompt' in data:output['prompt']=data['prompt']
    return output


def unpad_pi0_actions(actions):
    x=np.asarray(actions,dtype=np.float32)
    if x.ndim!=2 or x.shape[-1]<21:raise ValueError('Expected an action chunk with >=21 motor coordinates')
    return native_actions(x[...,:21])
