"""Read-only evaluation telemetry; NEVER policy inputs or task acceptance gates."""
import numpy as np
from state_policy_features import as_numpy


def read_grasp_telemetry(robot,cup):
    """EEF frames and geometric finger contacts, NOT calibrated force evidence.

    Unknown reads stay explicitly NaN/-1 and carry errors, not invented zeros.
    Queries do not move objects, create constraints or change control commands.
    """
    poses=np.full((2,7),np.nan,np.float32);counts=np.full(2,-1,np.int32);errors=[]
    cup_paths={cup.root_link.prim_path}
    try:cup_paths.update(link.prim_path for link in cup.links.values())
    except AttributeError:pass
    for index,arm in enumerate(('left','right')):
        try:
            position,quaternion=robot.eef_links[arm].get_position_orientation()
            pose=np.concatenate([as_numpy(position),as_numpy(quaternion)])
            if pose.shape!=(7,) or not np.isfinite(pose).all():raise ValueError('Invalid EEF pose')
            poses[index]=pose
        except Exception as error:errors.append(f'{arm} EEF telemetry unavailable: {type(error).__name__}')
        try:
            _,contacts=robot._find_gripper_contacts(arm=arm)
            links=set()
            for path in cup_paths:links.update(contacts.get(path,set()))
            counts[index]=len(links)
        except Exception as error:errors.append(f'{arm} contact telemetry unavailable: {type(error).__name__}')
    return poses,counts,errors
